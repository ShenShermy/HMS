# 🧠 CDS525 Group Project — Harmful Brain Activity Classification

使用深度学习对 EEG（脑电图）梅尔频谱图进行 **6 类脑活动分类**，
包含 EfficientNetB0、TCN、MiniViT 三个模型的完整训练、超参数对比与结果可视化。

---

## 📁 项目结构

```
eeg_project/
│
├── dataset.py              # 数据集加载（HMS真实数据 & 随机演示数据）
├── utils.py                # 训练循环、评估、绘图工具函数
│
├── models/
│   ├── __init__.py
│   ├── efficientnet_model.py   # EfficientNet-B0 迁移学习模型
│   ├── tcn_model.py            # TCN 时序卷积网络
│   └── vit_model.py            # Vision Transformer (MiniViT / ViT-B16)
│
├── train_efficientnet.py   # EfficientNet 完整训练 + 超参数实验
├── train_tcn.py            # TCN 完整训练 + 超参数实验
├── train_vit.py            # ViT 完整训练 + 超参数实验
├── compare_models.py       # 三模型横向对比
│
├── requirements.txt        # 依赖包列表
└── README.md               # 本文档
```

---

## 🏷️ 分类目标（6类）

| 标签 | 全称 | 含义 |
|------|------|------|
| `seizure` | Seizure | 癫痫发作 |
| `lpd` | Lateralized Periodic Discharge | 侧向性周期性放电 |
| `gpd` | Generalized Periodic Discharge | 广泛性周期性放电 |
| `lrda` | Lateralized Rhythmic Delta Activity | 侧向性节律性δ活动 |
| `grda` | Generalized Rhythmic Delta Activity | 广泛性节律性δ活动 |
| `other` | Other | 其他 |

---

## 🤖 三个模型说明

### 1. EfficientNetB0（推荐 Baseline）

- **原理**：复合缩放 CNN，同时在深度、宽度、分辨率三个维度按比例扩展
- **核心积木**：MBConv（Mobile Inverted Bottleneck）+ SE（Squeeze-and-Excitation）注意力
- **优势**：ImageNet 预训练权重，迁移学习效果好，参数少速度快
- **输入**：`(B, 3, 128, 256)` 梅尔频谱图

```
EfficientNet-B0 Backbone (ImageNet pretrained)
    ↓ AdaptiveAvgPool2d
    ↓ Dropout(0.3) → Linear(1280→256) → ReLU
    ↓ Linear(256→6)
```

### 2. TCN（Temporal Convolutional Network）

- **原理**：因果膨胀卷积，沿时间轴捕获长距离时序依赖，感受野随层数指数增长
- **核心积木**：Dilated Causal Conv1D + 残差连接
- **优势**：专为时序设计，可并行训练，不存在 LSTM 的梯度消失问题
- **感受野**：6层时约 `127` 个时间帧

```
输入 (B,3,128,256)
    ↓ 频率编码 CNN（压缩 freq 维度）→ (B,64,256)
    ↓ TCN × 6层（dilation=1,2,4,8,16,32）
    ↓ Global Average Pooling → (B,128)
    ↓ Linear → 6类
```

### 3. MiniViT（Vision Transformer，轻量版）

- **原理**：将频谱图切成 16×16 的 Patch，每个 Patch 作为 Token，用 Multi-Head Self-Attention 建模全局依赖
- **核心积木**：Patch Embedding + Positional Encoding + Transformer Encoder
- **优势**：一步获得全局感受野，Attention Map 可可视化解释
- **注意**：数据量小时比 CNN 难训练，建议用 `ViTEEG`（预训练版本）

```
输入 (B,3,128,256)
    ↓ 切成 8×16=128 个 16×16 patch
    ↓ Linear Embedding → 128 个 Token (dim=256)
    ↓ + [CLS] Token + 位置编码
    ↓ Transformer Encoder × 6层（8头注意力）
    ↓ [CLS] Token → LayerNorm → Linear → 6类
```

---

## ⚙️ 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. GPU 支持（自动检测）

代码会自动检测 CUDA → Apple MPS → CPU，**无需手动配置**：

```python
# utils.py 中的 get_device()
device = torch.device("cuda")   # NVIDIA GPU
device = torch.device("mps")    # Apple Silicon
device = torch.device("cpu")    # 无 GPU 时自动回退
```

---

## 🚀 运行方法

### 快速验证（使用随机假数据，无需下载数据集）

```bash
# 分别训练三个模型
python train_efficientnet.py
python train_tcn.py
python train_vit.py

# 三模型横向对比
python compare_models.py
```

### 使用真实 HMS 数据集

1. 从 Kaggle 下载数据：https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data

2. 目录结构：
```
data/
├── train.csv
└── train_spectrograms/
    ├── 1000086677.npy
    ├── 1000186671.npy
    └── ...
```

3. 修改各训练文件中的数据集加载部分：
```python
# 将这行：
dataset = DummyEEGDataset(size=800)

# 替换为：
from dataset import HMSSpectrogramDataset
dataset = HMSSpectrogramDataset(
    csv_path = "data/train.csv",
    spec_dir = "data/train_spectrograms/",
)
```

---

## 📊 实验内容（对应课程作业要求）

每个模型脚本自动运行以下实验：

| 实验 | 内容 | 对应作业 |
|------|------|----------|
| Exp 1 | CrossEntropyLoss 训练曲线 | Figure 1 |
| Exp 2 | 不同 Loss Function 对比 | Figure 2 |
| Exp 3 | 不同学习率 (0.1, 0.01, 0.001, 0.0001) | Figure 3 & 4 |
| Exp 4 | 不同 Batch Size (8, 16, 32, 64, 128) | Figure 5 & 6 |
| Final | 前100个测试样本预测可视化 | Figure 7 |

---

## 📂 结果文件结构

运行后自动生成：

```
results/
├── efficientnet/
│   ├── exp1_ce/          # CrossEntropy 实验
│   ├── exp2_ls/          # LabelSmoothing 实验
│   ├── exp3_lr_*/        # 不同学习率
│   ├── exp4_bs_*/        # 不同 batch size
│   ├── loss_compare_*.png
│   ├── lr_compare_*.png
│   ├── bs_compare_*.png
│   └── predictions.png   # 前100预测可视化
│
├── tcn/                  # 同上
├── vit/                  # 同上
│
└── comparison/
    ├── triple_curves.png  # 三模型训练曲线对比
    ├── bar_comparison.png # 精度/时间/参数柱状图
    ├── radar_chart.png    # 雷达图（综合能力）
    └── summary.json       # 数值结果汇总
```

---

## 📈 三模型对比概览

| 维度 | EfficientNetB0 | TCN | MiniViT |
|------|:--------------:|:---:|:-------:|
| 参数量 | ~5.3M | ~1-3M | ~3M |
| 时序建模 | 弱 | ✅ 强 | ✅ 强 |
| 迁移学习 | ✅ | ❌ | ✅（ViTEEG版）|
| 训练速度 | 快 | 最快 | 慢 |
| 数据需求 | 中 | 少 | 大 |
| 推荐场景 | 快速Baseline | 小数据时序 | 大数据全局建模 |

---

## 🔑 关键超参数说明

```python
# EfficientNet
optimizer = AdamW(lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(patience=4, factor=0.5)

# TCN
optimizer = Adam(lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(T_max=epochs)

# MiniViT
optimizer = AdamW(lr=5e-4, weight_decay=0.05, betas=(0.9, 0.999))
scheduler = LinearWarmup(10%) + CosineAnnealing
```
---

## 📚 参考文献

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML*.
2. Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv*.
3. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.
4. HMS Harmful Brain Activity Classification. Kaggle Competition, 2024.
