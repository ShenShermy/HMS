"""
compare_models.py
-----------------
三模型横向对比脚本

功能：
  1. 在相同数据集 & 超参数下各训练一次三个模型
  2. 生成三模型对比曲线图（Loss / Accuracy）
  3. 打印模型参数量 / 训练时间 / 最终测试精度对比表
  4. 生成综合对比报告图（雷达图 + 柱状图）

运行方式：
  python compare_models.py
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from dataset import DummyEEGDataset, get_dataloaders
from models import EfficientNetEEG, TCNEEG, MiniViT
from utils import get_device, train_model, plot_hyperparameter_comparison

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
RESULTS_DIR = "results/comparison"
NUM_CLASSES = 6
NUM_EPOCHS  = 20       # 对比实验用较少 epochs，节省时间
LR          = 1e-3
BATCH_SIZE  = 32

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 参数量统计
# ─────────────────────────────────────────────
def count_params(model: nn.Module):
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─────────────────────────────────────────────
# 单模型对比训练（固定超参数）
# ─────────────────────────────────────────────
def benchmark_model(model, model_name, train_loader, val_loader, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    t0 = time.time()
    history = train_model(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        optimizer    = optimizer,
        criterion    = criterion,
        device       = device,
        num_epochs   = NUM_EPOCHS,
        model_name   = model_name,
        save_dir     = RESULTS_DIR,
        early_stop_patience = NUM_EPOCHS,   # 对比实验不提前停止
    )
    elapsed = time.time() - t0

    return history, elapsed


# ─────────────────────────────────────────────
# 雷达图（综合能力对比）
# ─────────────────────────────────────────────
def plot_radar(scores: dict, save_path: str):
    """
    scores: {model_name: {"accuracy": x, "speed": x, "efficiency": x, "stability": x}}
    所有指标均归一化到 [0, 1]
    """
    categories = ["Test Acc", "Speed\n(1/time)", "Param\nEfficiency", "Loss\nStability"]
    N = len(categories)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]   # 闭合

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for (name, vals), color in zip(scores.items(), colors):
        v = list(vals.values()) + [list(vals.values())[0]]
        ax.plot(angles, v, "o-", linewidth=2, color=color, label=name)
        ax.fill(angles, v, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Model Capability Radar Chart", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ↳ Saved radar chart → {save_path}")


# ─────────────────────────────────────────────
# 柱状图对比
# ─────────────────────────────────────────────
def plot_bar_comparison(summary: dict, save_path: str):
    """
    summary: {model_name: {"test_acc": x, "train_time_min": x, "params_M": x}}
    """
    names = list(summary.keys())
    test_accs  = [summary[n]["test_acc"] * 100 for n in names]
    train_time = [summary[n]["train_time_min"] for n in names]
    params_m   = [summary[n]["params_M"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Three Models — Final Performance Comparison", fontsize=14, fontweight="bold")
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    # 1) Test Accuracy
    bars = axes[0].bar(names, test_accs, color=colors, alpha=0.85, edgecolor="white")
    axes[0].set_title("Test Accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy (%)")
    for bar, val in zip(bars, test_accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    # 2) Training Time
    bars = axes[1].bar(names, train_time, color=colors, alpha=0.85, edgecolor="white")
    axes[1].set_title("Training Time (min)")
    axes[1].set_ylabel("Minutes")
    for bar, val in zip(bars, train_time):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    # 3) Parameter Count
    bars = axes[2].bar(names, params_m, color=colors, alpha=0.85, edgecolor="white")
    axes[2].set_title("Parameter Count (M)")
    axes[2].set_ylabel("Millions")
    for bar, val in zip(bars, params_m):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.1f}M", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ↳ Saved bar comparison → {save_path}")


# ─────────────────────────────────────────────
# 训练曲线三合一对比图
# ─────────────────────────────────────────────
def plot_triple_curves(histories: dict, save_path: str):
    """将三个模型的曲线画在同一张 2×2 图上"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Three Models — Training Curves Comparison", fontsize=14, fontweight="bold")
    colors = {"EfficientNetB0": "#2196F3", "TCN": "#FF9800", "MiniViT": "#4CAF50"}

    metrics = [
        ("train_loss", "Train Loss",     axes[0, 0]),
        ("train_acc",  "Train Acc (%)",  axes[0, 1]),
        ("val_acc",    "Val Acc (%)",    axes[1, 0]),
        ("test_acc",   "Test Acc (%)",   axes[1, 1]),
    ]

    for key, ylabel, ax in metrics:
        for name, hist in histories.items():
            y = hist[key]
            if "acc" in key:
                y = [v * 100 for v in y]
            ax.plot(range(1, len(y)+1), y, marker="o", markersize=3,
                    color=colors[name], label=name)
        ax.set_title(ylabel)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ↳ Saved triple curve → {save_path}")


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────
def main():
    device  = get_device()
    dataset = DummyEEGDataset(size=800)   # 替换真实数据集
    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size=BATCH_SIZE)

    # ── 三个模型 ────────────────────────────────────────────────────────────
    models_cfg = {
        "EfficientNetB0": EfficientNetEEG(num_classes=NUM_CLASSES, pretrained=False).to(device),
        "TCN":            TCNEEG(num_classes=NUM_CLASSES).to(device),
        "MiniViT":        MiniViT(img_h=128, img_w=256, num_classes=NUM_CLASSES).to(device),
    }

    histories = {}
    summary   = {}

    for name, model in models_cfg.items():
        total_p, train_p = count_params(model)
        print(f"\n{'='*55}")
        print(f"  Model: {name}")
        print(f"  Total params: {total_p:,}  |  Trainable: {train_p:,}")
        print(f"{'='*55}")

        hist, elapsed = benchmark_model(
            model, name, train_loader, val_loader, test_loader, device
        )
        histories[name] = hist

        best_test = max(hist["test_acc"])
        final_loss = hist["train_loss"][-1]
        summary[name] = {
            "test_acc":        best_test,
            "final_train_loss": final_loss,
            "train_time_min":  elapsed / 60,
            "params_M":        total_p / 1e6,
        }

        print(f"  ✓ Best Test Acc: {best_test*100:.2f}%  |  Time: {elapsed/60:.1f} min")

    # ── 保存 summary JSON ─────────────────────────────────────────────────
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── 打印对比表 ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"  {'Model':<20} {'Test Acc':>10} {'Time(min)':>12} {'Params(M)':>12}")
    print("="*65)
    for name, s in summary.items():
        print(
            f"  {name:<20} "
            f"{s['test_acc']*100:>9.2f}% "
            f"{s['train_time_min']:>11.1f}m "
            f"{s['params_M']:>11.1f}M"
        )
    print("="*65)

    # ── 可视化 ─────────────────────────────────────────────────────────────
    plot_triple_curves(
        histories,
        save_path = os.path.join(RESULTS_DIR, "triple_curves.png"),
    )

    plot_bar_comparison(
        summary,
        save_path = os.path.join(RESULTS_DIR, "bar_comparison.png"),
    )

    # 雷达图（归一化各指标）
    max_acc   = max(s["test_acc"]     for s in summary.values())
    max_eff   = max(s["params_M"]     for s in summary.values())
    max_time  = max(s["train_time_min"] for s in summary.values())
    max_loss  = max(s["final_train_loss"] for s in summary.values())

    radar_scores = {}
    for name, s in summary.items():
        radar_scores[name] = {
            "accuracy":   s["test_acc"] / (max_acc + 1e-9),
            "speed":      1 - s["train_time_min"] / (max_time + 1e-9),     # 越快越好
            "efficiency": 1 - s["params_M"] / (max_eff + 1e-9),            # 越少越好
            "stability":  1 - s["final_train_loss"] / (max_loss + 1e-9),   # loss 越低越好
        }

    plot_radar(
        radar_scores,
        save_path = os.path.join(RESULTS_DIR, "radar_chart.png"),
    )

    print(f"\n✅ Comparison complete! All figures saved to: {RESULTS_DIR}/")
    print("\nGenerated files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"  📄 {f}")


if __name__ == "__main__":
    main()
