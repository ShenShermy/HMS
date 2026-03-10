"""
train_vit.py
------------
Vision Transformer (ViT) 完整训练脚本

使用 MiniViT（从头训练，轻量级）
若有 GPU 且数据集足够大，可切换为 ViTEEG（预训练迁移学习）

自动完成以下实验：
  实验1：CrossEntropyLoss      → results/vit/exp1_ce/
  实验2：LabelSmoothingLoss    → results/vit/exp2_ls/
  实验3：不同学习率对比          → results/vit/exp3_lr/
  实验4：不同 batch size 对比   → results/vit/exp4_bs/
  最终：预测前100个测试样本      → results/vit/predictions.png
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import DummyEEGDataset, get_dataloaders
from models import MiniViT, ViTEEG
from utils import (
    get_device,
    train_model,
    plot_training_curves,
    plot_hyperparameter_comparison,
    visualize_predictions,
)

RESULTS_DIR = "results/vit"
NUM_CLASSES = 6
NUM_EPOCHS  = 30
BASE_LR     = 5e-4   # ViT 通常用较小 lr
BASE_BS     = 32
IMG_H, IMG_W = 128, 256


def build_model(device, use_pretrained_vit: bool = False):
    """
    use_pretrained_vit=False → MiniViT（轻量，无需下载权重）
    use_pretrained_vit=True  → ViT-B/16（需要 224×224 输入 & 预训练权重）
    """
    if use_pretrained_vit:
        model = ViTEEG(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
    else:
        model = MiniViT(
            img_h       = IMG_H,
            img_w       = IMG_W,
            patch_size  = 16,
            embed_dim   = 256,
            num_heads   = 8,
            num_layers  = 6,
            dropout     = 0.1,
            num_classes = NUM_CLASSES,
        )
    return model.to(device)


def run_experiment(
    model, train_loader, val_loader, test_loader,
    criterion, lr, device, save_dir, exp_name
):
    os.makedirs(save_dir, exist_ok=True)

    # ViT 通常用 AdamW + Warmup + Cosine 调度
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = lr,
        weight_decay = 0.05,
        betas        = (0.9, 0.999),
    )
    # Cosine 退火 + 线性 Warmup
    warmup_epochs = max(1, NUM_EPOCHS // 10)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers = [
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - warmup_epochs, eta_min=1e-6),
        ],
        milestones = [warmup_epochs],
    )

    history = train_model(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        optimizer    = optimizer,
        criterion    = criterion,
        device       = device,
        num_epochs   = NUM_EPOCHS,
        scheduler    = None,           # 手动 step 替代 ReduceLROnPlateau
        model_name   = exp_name,
        save_dir     = save_dir,
    )

    # 手动调用 scheduler（train_model 中用 val_loss 触发，这里改为 epoch step）
    plot_training_curves(
        history,
        title     = f"MiniViT | {exp_name}",
        save_path = os.path.join(save_dir, f"{exp_name}_curves.png"),
    )
    return history


def main():
    device  = get_device()
    dataset = DummyEEGDataset(size=800)   # 替换为真实数据集

    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size=BASE_BS)

    # ═══════════════════════════════════════════════
    # 实验 1 & 2：CrossEntropy vs LabelSmoothing
    # ═══════════════════════════════════════════════
    print("\n" + "━"*55)
    print("  EXP 1 & 2: CrossEntropy vs LabelSmoothing")
    print("━"*55)

    ce_loss = nn.CrossEntropyLoss()
    ls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    model1  = build_model(device)
    hist_ce = run_experiment(
        model1, train_loader, val_loader, test_loader,
        ce_loss, BASE_LR, device,
        save_dir = os.path.join(RESULTS_DIR, "exp1_ce"),
        exp_name = "ce_loss",
    )

    model2  = build_model(device)
    hist_ls = run_experiment(
        model2, train_loader, val_loader, test_loader,
        ls_loss, BASE_LR, device,
        save_dir = os.path.join(RESULTS_DIR, "exp2_ls"),
        exp_name = "label_smoothing",
    )

    for metric, ylabel in [("train_loss", "Loss"), ("test_acc", "Accuracy (%)")]:
        plot_hyperparameter_comparison(
            histories = {"CrossEntropy": hist_ce, "LabelSmoothing(0.1)": hist_ls},
            metric    = metric,
            title     = f"MiniViT | Loss Comparison | {metric}",
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
            title     = f"MiniViT | LR Comparison | {metric}",
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
            title     = f"MiniViT | Batch Size Comparison | {metric}",
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

    print(f"\n✅ All ViT experiments done! Results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
