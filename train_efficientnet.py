"""
train_efficientnet.py
---------------------
EfficientNetB0 完整训练脚本

自动完成以下实验（对应作业要求）：
  实验1：CrossEntropyLoss     → results/efficientnet/exp1_ce/
  实验2：LabelSmoothingLoss   → results/efficientnet/exp2_ls/
  实验3：不同学习率对比         → results/efficientnet/exp3_lr/
  实验4：不同 batch size 对比  → results/efficientnet/exp4_bs/
  最终：预测前100个测试样本     → results/efficientnet/predictions.png
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from dataset import DummyEEGDataset, get_dataloaders   # 换真实数据集时替换 DummyEEGDataset
from models import EfficientNetEEG
from utils import (
    get_device,
    train_model,
    plot_training_curves,
    plot_hyperparameter_comparison,
    visualize_predictions,
)

# ─────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────
RESULTS_DIR  = "results/efficientnet"
NUM_CLASSES  = 6
NUM_EPOCHS   = 30          # 正式训练建议 50-100
BASE_LR      = 1e-3
BASE_BS      = 32
IMG_SIZE     = (128, 256)  # (freq, time)


def get_transform():
    """数据增强（训练时）"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def build_model(device):
    model = EfficientNetEEG(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False)
    return model.to(device)


def run_experiment(
    model, train_loader, val_loader, test_loader,
    criterion, lr, device, save_dir, exp_name
):
    """运行一次完整训练实验"""
    os.makedirs(save_dir, exist_ok=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

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
        title     = f"EfficientNetB0 | {exp_name}",
        save_path = os.path.join(save_dir, f"{exp_name}_curves.png"),
    )
    return history


def main():
    device = get_device()

    # ─── 数据集（替换为真实数据集）─────────────────────────────────────────
    # 真实数据：
    # from dataset import HMSSpectrogramDataset
    # dataset = HMSSpectrogramDataset("data/train.csv", "data/train_spectrograms/")
    dataset = DummyEEGDataset(size=800)

    # ═══════════════════════════════════════════════
    # 实验 1 & 2：不同 Loss Function
    # ═══════════════════════════════════════════════
    print("\n" + "━"*55)
    print("  EXP 1 & 2: CrossEntropy vs LabelSmoothing")
    print("━"*55)

    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size=BASE_BS)

    ce_loss = nn.CrossEntropyLoss()
    ls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label Smoothing

    # 实验1：CrossEntropy
    model1  = build_model(device)
    hist_ce = run_experiment(
        model1, train_loader, val_loader, test_loader,
        ce_loss, BASE_LR, device,
        save_dir = os.path.join(RESULTS_DIR, "exp1_ce"),
        exp_name = "ce_loss",
    )

    # 实验2：LabelSmoothing
    model2  = build_model(device)
    hist_ls = run_experiment(
        model2, train_loader, val_loader, test_loader,
        ls_loss, BASE_LR, device,
        save_dir = os.path.join(RESULTS_DIR, "exp2_ls"),
        exp_name = "label_smoothing",
    )

    # 对比图（作业 Figure 1 & 2 对应）
    for metric, ylabel in [
        ("train_loss", "Loss"),
        ("test_acc",   "Test Accuracy (%)"),
    ]:
        plot_hyperparameter_comparison(
            histories = {"CrossEntropy": hist_ce, "LabelSmoothing(0.1)": hist_ls},
            metric    = metric,
            title     = f"EfficientNetB0 | Loss Function Comparison | {metric}",
            ylabel    = ylabel,
            save_path = os.path.join(RESULTS_DIR, f"loss_compare_{metric}.png"),
        )

    # ═══════════════════════════════════════════════
    # 实验 3：不同学习率（作业要求 Figure 3 & 4）
    # ═══════════════════════════════════════════════
    print("\n" + "━"*55)
    print("  EXP 3: Learning Rate Comparison")
    print("━"*55)

    lr_list    = [0.1, 0.01, 0.001, 0.0001]
    lr_hists   = {}

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
            title     = f"EfficientNetB0 | LR Comparison | {metric}",
            ylabel    = ylabel,
            save_path = os.path.join(RESULTS_DIR, f"lr_compare_{metric}.png"),
        )

    # ═══════════════════════════════════════════════
    # 实验 4：不同 Batch Size（作业要求 Figure 5 & 6）
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
            title     = f"EfficientNetB0 | Batch Size Comparison | {metric}",
            ylabel    = ylabel,
            save_path = os.path.join(RESULTS_DIR, f"bs_compare_{metric}.png"),
        )

    # ═══════════════════════════════════════════════
    # 最终预测：前 100 个测试样本（作业 Figure 7）
    # ═══════════════════════════════════════════════
    print("\n" + "━"*55)
    print("  Prediction Visualization (first 100 test samples)")
    print("━"*55)

    # 加载最优模型权重
    best_ckpt = os.path.join(RESULTS_DIR, "exp1_ce", "ce_loss_best.pt")
    if os.path.exists(best_ckpt):
        model1.load_state_dict(torch.load(best_ckpt, map_location=device))

    visualize_predictions(
        model    = model1,
        test_loader = test_loader,
        device   = device,
        save_path = os.path.join(RESULTS_DIR, "predictions.png"),
        n        = 100,
    )

    print(f"\n✅ All EfficientNet experiments done! Results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
