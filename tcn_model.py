"""
models/tcn_model.py
-------------------
Temporal Convolutional Network (TCN) for EEG mel-spectrogram classification

结构思路：
    输入频谱图 (B, 3, freq=128, time=256)
        ↓ 先用 CNN 提取频率特征，压缩 freq 维度
    (B, C, time=256)
        ↓ TCN：多层膨胀因果卷积，捕获时间上下文
    (B, C, time=256)
        ↓ Global Average Pooling → (B, C)
        ↓ 分类头 → (B, 6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 膨胀因果卷积 残差块
# ─────────────────────────────────────────────
class TCNResidualBlock(nn.Module):
    """
    单个 TCN 残差块，包含两层膨胀因果卷积。

    因果填充 (Causal Padding)：
        只在左侧填充，保证 t 时刻只能看到 t 及之前的信息。
        padding = (kernel_size - 1) * dilation
        卷积后裁剪右侧多余部分。
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        dilation:     int = 1,
        dropout:      float = 0.2,
    ):
        super().__init__()
        # 左侧因果填充量
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size = kernel_size,
            dilation    = dilation,
            padding     = self.padding,   # 仅左侧有效，需裁剪右侧
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size = kernel_size,
            dilation    = dilation,
            padding     = self.padding,
        )
        self.bn2     = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU(inplace=True)

        # 残差连接的通道对齐
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def _causal_trim(self, x: torch.Tensor, original_len: int) -> torch.Tensor:
        """裁剪右侧多余的填充，保证输出长度 = 输入长度（因果性）"""
        return x[:, :, :original_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(2)   # 输入时间长度

        # 第一层
        out = self.conv1(x)
        out = self._causal_trim(out, T)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # 第二层
        out = self.conv2(out)
        out = self._causal_trim(out, T)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # 残差
        res = self.downsample(x)
        return self.relu(out + res)


# ─────────────────────────────────────────────
# 完整 TCN 模型
# ─────────────────────────────────────────────
class TCNEEG(nn.Module):
    """
    TCN for EEG Mel-Spectrogram Classification

    Stage 1 — 频率编码器 (CNN 2D)
        将 (3, freq, time) 映射到 (channels, time)

    Stage 2 — 时序编码器 (TCN)
        膨胀因果卷积，感受野随层数指数增长

    Stage 3 — 分类头
        Global Average Pooling + Linear
    """

    def __init__(
        self,
        num_classes:   int   = 6,
        freq_channels: int   = 64,    # 频率编码后通道数
        tcn_channels:  int   = 128,   # TCN 隐层通道数
        num_tcn_layers:int   = 6,     # TCN 层数（膨胀率 1,2,4,8,16,32）
        kernel_size:   int   = 3,
        dropout:       float = 0.2,
    ):
        super().__init__()

        # ── Stage 1: 频率轴 CNN ─────────────────────────────────────
        # 输入 (B, 3, 128, 256) → (B, freq_channels, 1, 256) → squeeze → (B, freq_channels, 256)
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(3, 32,  kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),   # (B, 64, 1, time) 压缩频率维
        )

        # ── Stage 2: TCN ────────────────────────────────────────────
        # 膨胀率：1, 2, 4, 8, 16, 32, …
        tcn_layers = []
        in_ch = freq_channels
        for i in range(num_tcn_layers):
            dilation = 2 ** i
            tcn_layers.append(
                TCNResidualBlock(
                    in_channels  = in_ch,
                    out_channels = tcn_channels,
                    kernel_size  = kernel_size,
                    dilation     = dilation,
                    dropout      = dropout,
                )
            )
            in_ch = tcn_channels
        self.tcn = nn.Sequential(*tcn_layers)

        # ── Stage 3: 分类头 ─────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool1d(1)   # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(tcn_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, freq, time)

        # Stage 1: 频率编码
        freq_feat = self.freq_encoder(x)         # (B, 64, 1, time)
        freq_feat = freq_feat.squeeze(2)          # (B, 64, time)

        # Stage 2: TCN 时序建模
        tcn_out = self.tcn(freq_feat)             # (B, tcn_channels, time)

        # Stage 3: 分类
        pooled = self.gap(tcn_out).squeeze(-1)    # (B, tcn_channels)
        out    = self.classifier(pooled)           # (B, num_classes)
        return out

    def receptive_field(self, kernel_size: int = 3, num_layers: int = 6) -> int:
        """计算 TCN 的理论感受野"""
        rf = 1
        for i in range(num_layers):
            dilation = 2 ** i
            rf += (kernel_size - 1) * dilation * 2
        return rf


# ─── 快速测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = TCNEEG(num_classes=6)
    x     = torch.randn(4, 3, 128, 256)
    out   = model(x)
    print(f"TCNEEG output shape: {out.shape}")     # (4, 6)
    rf = model.receptive_field()
    print(f"TCN Receptive Field: {rf} time steps")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")
