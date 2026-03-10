"""
dataset.py
----------
EEG 梅尔频谱图数据集加载与预处理
支持 HMS Harmful Brain Activity Classification 数据集
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────
# 标签映射
# ─────────────────────────────────────────────
LABEL_MAP = {
    "seizure":  0,
    "lpd":      1,
    "gpd":      2,
    "lrda":     3,
    "grda":     4,
    "other":    5,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)


# ─────────────────────────────────────────────
# 真实 HMS 频谱图数据集
# ─────────────────────────────────────────────
class HMSSpectrogramDataset(Dataset):
    """
    读取 HMS Kaggle 数据集的频谱图 (.npy 或 .png)
    CSV 需包含列：spectrogram_id, expert_consensus (标签字符串)

    目录结构示例：
        data/
          train.csv
          train_spectrograms/   ← 每个 .npy 文件是一张频谱图
    """

    def __init__(self, csv_path: str, spec_dir: str, transform=None, use_npy: bool = True):
        self.df = pd.read_csv(csv_path)
        self.spec_dir = spec_dir
        self.transform = transform
        self.use_npy = use_npy

        # 统一小写标签
        self.df["label"] = (
            self.df["expert_consensus"].str.lower().map(LABEL_MAP)
        )
        # 去掉无效标签行
        self.df = self.df.dropna(subset=["label"]).reset_index(drop=True)
        self.df["label"] = self.df["label"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spec_id = row["spectrogram_id"]
        label   = row["label"]

        if self.use_npy:
            fpath = os.path.join(self.spec_dir, f"{spec_id}.npy")
            spec  = np.load(fpath).astype(np.float32)           # (freq, time)
            spec  = np.log1p(spec)                              # 对数压缩
            spec  = (spec - spec.mean()) / (spec.std() + 1e-6)  # 归一化
            # → (1, freq, time)
            spec  = torch.tensor(spec).unsqueeze(0)
            # 复制到 3 通道供 CNN / ViT
            spec  = spec.repeat(3, 1, 1)
        else:
            fpath = os.path.join(self.spec_dir, f"{spec_id}.png")
            img   = Image.open(fpath).convert("RGB")
            spec  = transforms.ToTensor()(img)

        if self.transform:
            spec = self.transform(spec)

        return spec, label


# ─────────────────────────────────────────────
# 演示用随机假数据集（无需下载，直接跑通代码）
# ─────────────────────────────────────────────
class DummyEEGDataset(Dataset):
    """
    生成随机频谱图，用于快速验证代码流程
    shape: (3, 128, 256)  模拟 (通道, 频率, 时间)
    """

    def __init__(self, size: int = 500, transform=None):
        self.size      = size
        self.transform = transform
        # 固定随机种子保证可复现
        rng = np.random.default_rng(42)
        self.data   = rng.random((size, 3, 128, 256)).astype(np.float32)
        self.labels = rng.integers(0, NUM_CLASSES, size=size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])
        y = int(self.labels[idx])
        if self.transform:
            x = self.transform(x)
        return x, y


# ─────────────────────────────────────────────
# DataLoader 工厂函数
# ─────────────────────────────────────────────
def get_dataloaders(
    dataset,
    batch_size:  int   = 32,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    num_workers: int   = 2,
    seed:        int   = 42,
):
    """
    将数据集按比例随机分为 train / val / test，
    返回三个 DataLoader。
    """
    n       = len(dataset)
    n_test  = int(n * test_ratio)
    n_val   = int(n * val_ratio)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = True,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)

    print(f"[Dataset] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    return train_loader, val_loader, test_loader
