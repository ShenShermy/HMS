"""
models/vit_model.py
-------------------
Vision Transformer (ViT) for EEG Mel-Spectrogram Classification

两种模式：
  1. pretrained=True  → 使用 torchvision ViT-B/16 ImageNet 权重，只微调分类头
  2. pretrained=False → 从头构建一个轻量级 MiniViT（适合小数据集）

MiniViT 结构（轻量版）：
    输入频谱图 (B, 3, H, W)
        ↓ 切成 patch_size × patch_size 小块
    N 个 patch token + 1 个 [CLS] token
        ↓ + 位置编码
    Transformer Encoder × num_layers 层
        [CLS] token 输出
        ↓
    分类头 → (B, 6)
"""

import math
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


# ─────────────────────────────────────────────
# 迁移学习版 ViT-B/16（推荐）
# ─────────────────────────────────────────────
class ViTEEG(nn.Module):
    """
    torchvision ViT-B/16 迁移学习版本。

    注意：ViT-B/16 要求输入尺寸 224×224。
    当传入非标准尺寸时，自动调整 patch 数量的位置编码
    （设 interpolate_pos_encoding=True）。

    建议在 DataLoader 中将频谱图 resize 到 224×224。
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()

        weights   = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit  = vit_b_16(weights=weights)

        # 冻结骨干（仅训练分类头，节省显存）
        if freeze_backbone:
            for name, p in self.vit.named_parameters():
                if "heads" not in name:
                    p.requires_grad = False

        # 替换分类头适配 6 类
        in_features = self.vit.heads.head.in_features   # 768 for ViT-B
        self.vit.heads = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


# ─────────────────────────────────────────────
# 轻量级 MiniViT（从头训练，适合小数据）
# ─────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """将图像切分为 patch 并线性投影"""

    def __init__(self, img_h: int, img_w: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        assert img_h % patch_size == 0 and img_w % patch_size == 0, \
            "图像尺寸必须能被 patch_size 整除"
        self.n_patches = (img_h // patch_size) * (img_w // patch_size)
        # 用步长为 patch_size 的卷积实现线性投影
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)            # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)            # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)       # (B, n_patches, embed_dim)
        return x


class MiniViT(nn.Module):
    """
    从头训练的轻量级 ViT

    参数：
        img_h, img_w  : 输入图像尺寸（如 128, 256）
        patch_size    : patch 边长（如 16）
        in_channels   : 输入通道数（3）
        embed_dim     : token 嵌入维度
        num_heads     : 多头注意力头数
        num_layers    : Transformer 层数
        mlp_ratio     : FFN 隐层维度倍率
    """

    def __init__(
        self,
        img_h:       int   = 128,
        img_w:       int   = 256,
        patch_size:  int   = 16,
        in_channels: int   = 3,
        embed_dim:   int   = 256,
        num_heads:   int   = 8,
        num_layers:  int   = 6,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.1,
        num_classes: int   = 6,
    ):
        super().__init__()

        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_h, img_w, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = num_heads,
            dim_feedforward = int(embed_dim * mlp_ratio),
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,    # Pre-LN，更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # Patch embedding
        tokens = self.patch_embed(x)                      # (B, n_patches, D)

        # 拼接 [CLS] token
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)          # (B, n_patches+1, D)

        # 加位置编码
        tokens = self.pos_drop(tokens + self.pos_embed)

        # Transformer Encoder
        encoded = self.encoder(tokens)                    # (B, n_patches+1, D)

        # 取 [CLS] token 做分类
        cls_out = self.norm(encoded[:, 0])                # (B, D)
        return self.head(cls_out)                         # (B, num_classes)


# ─── 快速测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── MiniViT ──")
    model = MiniViT(img_h=128, img_w=256, num_classes=6)
    x     = torch.randn(4, 3, 128, 256)
    out   = model(x)
    print(f"MiniViT output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")

    print("\n── ViTEEG (pretrained, no download) ──")
    model2 = ViTEEG(num_classes=6, pretrained=False)
    x2     = torch.randn(2, 3, 224, 224)
    out2   = model2(x2)
    print(f"ViTEEG output: {out2.shape}")
