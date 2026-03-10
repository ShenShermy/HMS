"""
train_tcn.py
------------
TCN (Temporal Convolutional Network) 完整训练脚本

自动完成以下实验：
  实验1：CrossEntropyLoss      → results/tcn/exp1_ce/
  实验2：FocalLoss             → results/tcn/exp2_focal/
  实验3：不同学习率对比          → results/tcn/exp3_lr/
  实验4：不同 batch size 对比   → results/tcn/exp4_bs/
  最终：预测前100个测试样本      → results/tcn/predictions.png
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import DummyEEGDataset, get_dataloaders
from models import TCNEEG
from utils import (
    get_device,
    train_model,
    plot_training_curves,
    plot_hyperparameter_comparison,
    visualize_predictions,
)

RESULTS_DIR = "results/tcn"
NUM_CLASSES = 6
NUM_EPOCHS  = 30
BASE_LR     = 1e-3
BASE_BS     = 32


# ─────────────────────────────────────────────
# Focal Loss（适合类别不均衡的 EEG 数据）
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p) = -α(1-p)^γ · log(p)
    γ > 0 降低易分类样本权重，聚焦难分类样本
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt      = torch.exp(-ce_loss)                  # 预测概率
        fl      = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return fl.mean()


def build_model(device):
    model = TCNEEG(
        num_classes    = NUM_CLASSES,
        freq_channels  = 64,
        tcn_channels   = 128,
        num_tcn_layers = 6,
        kernel_size    = 3,
        dropout        = 0.2,
    )
    return model.to(device)


def run_experiment(
    model, train_loader, val_loader, test_loader,
    criterion, lr, device, save_dir, exp_name
):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    history = train_model(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        optimizer    = optimizer,
        criterion    = criterion,
        device       = device,
        num_epochs   = NUM_EPOCHS,
        scheduler    = scheduler,
        model_name   = exp_name,
        save_dir     = save_dir,
    )
    plot_training_curves(
        history,
        title     = f"TCN | {exp_name}",
        save_path = os.path.join(save_dir, f"{exp_name}_curves.png"),
    )
    return history


def main():
    device  = get_device()
    dataset = DummyEEGDataset(size=800)   # 替换为真实数据集

    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size=BASE_BS)

    # ═══════════════════════════════════════════════
    # 实验 1 & 2：CrossEntropy vs FocalLoss
    # ═══════════════════════════════════════════════
    print("\n" + "━"*55)
    print("  EXP 1 & 2: CrossEntropy vs FocalLoss")
    print("━"*55)

    ce_loss    = nn.CrossEntropyLoss()
    focal_loss = FocalLoss(gamma=2.0, alpha=0.25)

    model1  = build_model(device)
    hist_ce = run_experiment(
        model1, train_loader, val_loader, test_loader,
        ce_loss, BASE_LR, device,
        save_dir = os.path.join(RESULTS_DIR, "exp1_ce"),
        exp_name = "ce_loss",
    )

    model2       = build_model(device)
    hist_focal   = run_experiment(
        model2, train_loader, val_loader, test_loader,
        focal_loss, BASE_LR, device,
        save_dir = os.path.join(RESULTS_DIR, "exp2_focal"),
        exp_name = "focal_loss",
    )

    for metric, ylabel in [("train_loss", "Loss"), ("test_acc", "Accuracy (%)")]:
        plot_hyperparameter_comparison(
            histories = {"CrossEntropy": hist_ce, "FocalLoss(γ=2)": hist_focal},
            metric    = metric,
            title     = f"TCN | Loss Function Comparison | {metric}",
            ylabel    = ylabel,
            save_path = os.path.join(RESULTS_DIR, f"loss_compare_{metric}.png"),
        )

    # ═══════════════════════════════════════════════
    # 实验 3：不同学习率
    # ═══════════════════════════════════════════════
    print("\n" + "━"*55)
    print("  EXP 3: Learning Rate Comparison")
    print("━"*55)

    lr_list  = [0.1, 0.01, 0.001, 0.0001]
    lr_hists = {}

    for lr in lr_list:
        model_lr = build_model(device)
        hist = run_experiment(
            model_lr, train_loader, val_loader, test_loader,
            ce_loss, lr, device,
            save_dir = os.path.join(RESULTS_DIR, f"exp3_lr_{lr}"),
            exp_name = f"lr_{lr}",
        )
        lr_hists[f"lr={lr}"] = hist

    for metric, ylabel in [("train_loss", "Loss"), ("test_acc", "Accuracy (%)")]:
        plot_hyperparameter_comparison(
            histories = lr_hists,
            metric    = metric,
            title     = f"TCN | LR Comparison | {metric}",
            ylabel    = ylabel,
            save_path = os.path.join(RESULTS_DIR, f"lr_compare_{metric}.png"),
        )

    # ═══════════════════════════════════════════════
    # 实验 4：不同 Batch Size
    # ═══════════════════════════════════════════════
    print("\n" + "━"*55)
    print("  EXP 4: Batch Size Comparison")
    print("━"*55)

    bs_list  = [8, 16, 32, 64, 128]
    bs_hists = {}

    for bs in bs_list:
        train_l, val_l, test_l = get_dataloaders(dataset, batch_size=bs)
        model_bs = build_model(device)
        hist = run_experiment(
            model_bs, train_l, val_l, test_l,
            ce_loss, BASE_LR, device,
            save_dir = os.path.join(RESULTS_DIR, f"exp4_bs_{bs}"),
            exp_name = f"bs_{bs}",
        )
        bs_hists[f"bs={bs}"] = hist

    for metric, ylabel in [("train_loss", "Loss"), ("test_acc", "Accuracy (%)")]:
        plot_hyperparameter_comparison(
            histories = bs_hists,
            metric    = metric,
            title     = f"TCN | Batch Size Comparison | {metric}",
            ylabel    = ylabel,
            save_path = os.path.join(RESULTS_DIR, f"bs_compare_{metric}.png"),
        )

    # ═══════════════════════════════════════════════
    # 预测可视化
    # ═══════════════════════════════════════════════
    best_ckpt = os.path.join(RESULTS_DIR, "exp1_ce", "ce_loss_best.pt")
    if os.path.exists(best_ckpt):
        model1.load_state_dict(torch.load(best_ckpt, map_location=device))

    visualize_predictions(
        model       = model1,
        test_loader = test_loader,
        device      = device,
        save_path   = os.path.join(RESULTS_DIR, "predictions.png"),
        n           = 100,
    )

    print(f"\n✅ All TCN experiments done! Results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
