"""
models/efficientnet_model.py
----------------------------
基于 torchvision 预训练 EfficientNet-B0 的迁移学习模型
输入：(B, 3, 128, 256) 梅尔频谱图
输出：(B, 6) 6类脑活动 logits
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetEEG(nn.Module):
    """
    EfficientNet-B0 迁移学习版本

    结构：
        EfficientNet-B0 骨干（预训练 ImageNet 权重）
            ↓  features 输出 (B, 1280, H', W')
        Adaptive Average Pooling → (B, 1280)
            ↓
        Dropout(0.3)
            ↓
        Linear(1280 → 256)  + ReLU
            ↓
        Linear(256 → 6)
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # 取特征提取部分，去掉原分类头
        self.features = backbone.features       # 输出通道 1280
        self.pool     = nn.AdaptiveAvgPool2d(1)

        # 冻结骨干参数（仅微调分类头）
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        in_features = 1280
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        feat = self.features(x)          # (B, 1280, H', W')
        feat = self.pool(feat)            # (B, 1280, 1, 1)
        feat = feat.flatten(1)            # (B, 1280)
        out  = self.classifier(feat)      # (B, num_classes)
        return out


# ─── 快速测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = EfficientNetEEG(num_classes=6, pretrained=False)
    x     = torch.randn(4, 3, 128, 256)
    out   = model(x)
    print(f"EfficientNetEEG output shape: {out.shape}")   # (4, 6)
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}  |  Trainable: {train:,}")
