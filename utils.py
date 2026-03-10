"""
utils.py
--------
通用训练循环、评估函数、曲线绘制、结果保存
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from dataset import IDX_TO_LABEL, NUM_CLASSES


# ─────────────────────────────────────────────
# 设备自动检测
# ─────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


# ─────────────────────────────────────────────
# 单 epoch 训练
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += X.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# 评估（val 或 test）
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out  = model(X)
        loss = criterion(out, y)

        total_loss += loss.item() * X.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += X.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# 完整训练流程
# ─────────────────────────────────────────────
def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    num_epochs:    int = 30,
    scheduler      = None,
    early_stop_patience: int = 7,
    model_name:    str = "model",
    save_dir:      str = "results",
):
    """
    训练模型并记录每个 epoch 的 loss / accuracy。
    支持 Early Stopping 和学习率调度器。
    返回 history dict。
    """
    os.makedirs(save_dir, exist_ok=True)
    best_val_acc  = 0.0
    patience_cnt  = 0
    best_ckpt     = os.path.join(save_dir, f"{model_name}_best.pt")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "test_acc":   [],
    }

    print(f"\n{'='*55}")
    print(f"  Training: {model_name}  |  epochs={num_epochs}")
    print(f"  Criterion: {criterion.__class__.__name__}")
    print(f"{'='*55}")

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, val_loader,  criterion, device)
        _,       ts_acc = evaluate(model, test_loader, criterion, device)

        if scheduler:
            scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["test_acc"].append(ts_acc)

        # Early stopping
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_cnt = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_cnt += 1

        print(
            f"Epoch [{epoch:03d}/{num_epochs}] "
            f"Loss: {tr_loss:.4f} | "
            f"TrainAcc: {tr_acc*100:.2f}% | "
            f"ValAcc: {vl_acc*100:.2f}% | "
            f"TestAcc: {ts_acc*100:.2f}%"
        )

        if patience_cnt >= early_stop_patience:
            print(f"  ↳ Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} min | Best Val Acc: {best_val_acc*100:.2f}%")

    # 保存 history
    hist_path = os.path.join(save_dir, f"{model_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


# ─────────────────────────────────────────────
# 绘制训练曲线
# ─────────────────────────────────────────────
def plot_training_curves(history: dict, title: str, save_path: str):
    """
    绘制 Loss 和 Accuracy 双图，保存到文件。
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Loss 曲线
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   "r-s", markersize=3, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy 曲线
    ax = axes[1]
    ax.plot(epochs, [a*100 for a in history["train_acc"]], "b-o", markersize=3, label="Train Acc")
    ax.plot(epochs, [a*100 for a in history["val_acc"]],   "r-s", markersize=3, label="Val Acc")
    ax.plot(epochs, [a*100 for a in history["test_acc"]],  "g-^", markersize=3, label="Test Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ↳ Saved curve → {save_path}")


# ─────────────────────────────────────────────
# 多曲线对比图（不同 lr / batch_size）
# ─────────────────────────────────────────────
def plot_hyperparameter_comparison(
    histories:   dict,   # {label_str: history_dict}
    metric:      str,    # "train_loss" | "train_acc" | "test_acc"
    title:       str,
    ylabel:      str,
    save_path:   str,
):
    plt.figure(figsize=(9, 5))
    for label, hist in histories.items():
        y = hist[metric]
        if "acc" in metric:
            y = [v * 100 for v in y]
        plt.plot(range(1, len(y)+1), y, marker="o", markersize=3, label=label)

    plt.title(title, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ↳ Saved comparison → {save_path}")


# ─────────────────────────────────────────────
# 预测前 100 个测试样本，输出结果表格图
# ─────────────────────────────────────────────
@torch.no_grad()
def visualize_predictions(model, test_loader, device, save_path: str, n: int = 100):
    model.eval()
    all_preds, all_true = [], []

    for X, y in test_loader:
        X = X.to(device)
        preds = model(X).argmax(1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_true.extend(y.numpy().tolist())
        if len(all_preds) >= n:
            break

    all_preds = all_preds[:n]
    all_true  = all_true[:n]

    # 混淆矩阵
    cm     = confusion_matrix(all_true, all_preds, labels=list(range(NUM_CLASSES)))
    labels = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左：混淆矩阵热力图
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=axes[0]
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title(f"Confusion Matrix (first {n} test samples)")

    # 右：前 30 条预测结果文字对比
    ax = axes[1]
    ax.axis("off")
    cols   = ["#", "Actual", "Predicted", "Correct"]
    rows   = []
    for i in range(min(30, n)):
        correct = "✓" if all_true[i] == all_preds[i] else "✗"
        rows.append([
            str(i+1),
            IDX_TO_LABEL[all_true[i]],
            IDX_TO_LABEL[all_preds[i]],
            correct,
        ])
    tbl = ax.table(
        cellText   = rows,
        colLabels  = cols,
        cellLoc    = "center",
        loc        = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.4)
    axes[1].set_title("First 30 Predictions")

    plt.suptitle("Prediction Visualization", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ↳ Saved predictions → {save_path}")

    # 打印分类报告
    print("\n[Classification Report]")
    print(classification_report(all_true, all_preds, target_names=labels))
