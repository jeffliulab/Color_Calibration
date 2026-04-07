<div align="center">

[![en](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![zh](https://img.shields.io/badge/lang-中文-red.svg)](README_zh.md)

<h1>Card Calibration</h1>

<p>
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-ultralytics-purple?logo=yolo&logoColor=white" alt="YOLOv8">
  <img src="https://img.shields.io/badge/XGBoost-1.x-blue?logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/Gradio-5.x-orange?logo=gradio&logoColor=white" alt="Gradio">
  <img src="https://img.shields.io/badge/状态-已完成-brightgreen" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p>
  <strong>自动化颜色校准：检测照片中的校准卡，提取参考色块，预测目标区域在标准光照下的真实颜色。</strong>
</p>

<p>
  <a href="https://huggingface.co/spaces/jeffliulab/card-calibration-v1"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20在线演示-Card%20Calibration-blue" alt="Demo"></a>
  <a href="https://huggingface.co/jeffliulab/card-calibration-v1"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20模型权重-Weights-yellow" alt="Model"></a>
  <a href="https://huggingface.co/datasets/jeffliulab/card-calibration-v1-data"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20数据集-v1--data-green" alt="Dataset"></a>
</p>

</div>

---

## 项目亮点

- **Lab Mean ΔE = 4.59** — 达到商业印刷标准（XGBoost + 贝叶斯超参数优化）
- **两阶段 YOLO 检测** — 卡片定位 + 四色块识别，全自动流水线
- **12 维特征工程** — 9 个参考色差值 + 3 个目标 RGB 通道，输入树模型回归
- **一键 HuggingFace 演示** — 上传照片即可获取预测真实颜色
- **可打印校准卡** — 下载模板卡片，打印后即可使用

---

## 目录

- [在线演示](#在线演示)
- [项目存储拓扑](#项目存储拓扑)
- [任务定义](#任务定义)
- [系统架构](#系统架构)
- [实验结果](#实验结果)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [数据](#数据)
- [相关链接](#相关链接)
- [致谢](#致谢)

---

## 项目存储拓扑

```
                    GitHub
              jeffliulab/Color_Calibration
              (源代码 + 脚本 + 文档)
                        │
                        │  引用
                        ▼
            HuggingFace (jeffliulab)
            ┌──────────────────────────────────────┐
            │ Dataset  card-calibration-v1-data    │  ← 原始照片 + 增强 + 训练集
            │ Model    card-calibration-v1         │  ← YOLO + XGBoost + RF 推理权重
            │ Space    card-calibration-v1         │  ← Gradio 在线演示
            └──────────────────────────────────────┘
```

| 资源 | 平台 | URL |
|---|---|---|
| 代码 | GitHub | [`jeffliulab/Color_Calibration`](https://github.com/jeffliulab/Color_Calibration) |
| 数据 | HF Dataset | [`jeffliulab/card-calibration-v1-data`](https://huggingface.co/datasets/jeffliulab/card-calibration-v1-data) |
| 模型 | HF Model | [`jeffliulab/card-calibration-v1`](https://huggingface.co/jeffliulab/card-calibration-v1) |
| 演示 | HF Space | [`jeffliulab/card-calibration-v1`](https://huggingface.co/spaces/jeffliulab/card-calibration-v1) |

---

## 在线演示

在浏览器中直接试用 — 上传一张包含校准卡的照片：

**[HuggingFace Spaces 在线演示](https://huggingface.co/spaces/jeffliulab/card-calibration-v1)**

<img src="docs/readme/card/card.png" width="300">

> 打印此卡片，放置在彩色物体上，拍照后上传即可。

---

## 任务定义

**输入：** 一张包含颜色校准卡的照片（卡上有红色圆形、绿色三角形、蓝色五边形、黑色方框四个色块）。

**输出：** 目标色块在标准 D65 光照下的真实 RGB 颜色。

**为什么需要：** 相机在不同光照条件下拍摄的颜色会偏移。本系统利用卡片上已知的参考色块来修正这种偏移。

---

## 系统架构

```
照片 ──▶ YOLO 第一阶段 ──▶ 卡片裁剪 ──▶ YOLO 第二阶段 ──▶ 4 个色块
                                                              │
                                          ┌───────────────────┘
                                          ▼
                                       特征工程
                                   (9 个差值 + 3 个 RGB)
                                          │
                                          ▼
                                   XGBoost / RF 模型
                                          │
                                          ▼
                                     预测真实 RGB
```

| 阶段 | 模型 | 说明 |
|------|------|------|
| 卡片检测 | YOLOv8-nano (`yolo_first.pt`) | 定位校准卡的边界框 |
| 色块检测 | YOLOv8-nano (`yolo_second.pt`) | 识别 `red_circle`、`green_triangle`、`blue_pentagon`、`black_box` |
| 颜色预测 | XGBoost / Random Forest | 根据 12 维特征向量预测真实 RGB |

<details>
<summary><strong>特征工程详情</strong>（点击展开）</summary>

提取每个检测色块中心 1/3 区域的平均 RGB，然后计算：

```
特征向量 (12 维):
  Delta_RR_red   = Rp_R - 255    Delta_RG_red   = Rp_G - 0      Delta_RB_red   = Rp_B - 0
  Delta_RR_green = Gp_R - 0      Delta_RG_green = Gp_G - 255    Delta_RB_green = Gp_B - 0
  Delta_RR_blue  = Bp_R - 0      Delta_RG_blue  = Bp_G - 0      Delta_RB_blue  = Bp_B - 255
  Cp_R, Cp_G, Cp_B  (目标色块拍摄 RGB)

预测目标: Cs_R, Cs_G, Cs_B  (标准光照下的真实 RGB)
```

其中 `Rp/Gp/Bp` = 红/绿/蓝参考色块的拍摄 RGB；`Cp` = 目标色块的拍摄值。

</details>

---

## 实验结果

### 模型对比

| 模型 | R² | RMSE | Lab Mean ΔE | Lab Median ΔE |
|------|---:|-----:|------------:|--------------:|
| **XGBoost（调优后）** | **0.8280** | **11.76** | **4.59** | **3.61** |
| Random Forest | 0.8225 | 12.10 | 5.20 | 3.96 |
| Linear Regression | 0.7113 | 14.98 | 6.63 | 5.53 |
| MLP | 0.7068 | 14.22 | 7.39 | 6.70 |

> ΔE < 3：专业校准级别 · ΔE < 5：商业印刷标准 · ΔE < 10：可接受

### 训练数据

- 约 2,000 张原始图片，增强后扩展至约 9,000+ 样本（亮度、色调、模糊、噪声、旋转等）
- 训练/测试划分：70/30，`random_state=42`
- 数据通过 DVC 管理，存储在 GCP（`gs://color_calibration`）

<details>
<summary><strong>各模型详细配置</strong>（点击展开）</summary>

**XGBoost（最佳）：**
- 通过 `tune_xgboost.py` 进行贝叶斯超参数优化
- Boosting 轮数：500，学习率和树深度经调优

**Random Forest：**
- `n_estimators=500`，`random_state=42`

**Linear Regression：**
- 标准最小二乘法，作为 baseline

**MLP：**
- 2 个隐藏层（各 64 神经元，ReLU），Adam 优化器，500 epochs

</details>

---

## 项目结构

```
card-calibration/
├── space/                        # HuggingFace Space（Gradio 演示）
│   ├── app.py                    # Gradio UI 入口
│   ├── model_utils.py            # 模型下载（HF Hub）& 推理管线
│   ├── requirements.txt          # Space Python 依赖
│   └── README.md                 # HF Space YAML 元数据
│
├── src/                          # 核心源代码
│   ├── detect/yolo.py            # 两阶段 YOLO 检测（PatternDetector）
│   ├── predict/predict_rf.py     # ColorPredictionSystem — 完整推理
│   ├── feature_extraction/       # 从裁剪色块提取 RGB 特征
│   ├── train/                    # 模型训练脚本
│   │   ├── pre_train.py          # 数据加载 & 特征工程
│   │   ├── train_xgboost.py      # XGBoost 训练
│   │   ├── train_rf.py           # Random Forest 训练
│   │   ├── train_linear_regression.py
│   │   └── train_mlp.py          # MLP 训练
│   ├── tune/tune_xgboost.py     # 贝叶斯超参数优化
│   ├── data_processing/          # 数据清理 & 预处理
│   ├── detect_processing/        # YOLO 训练数据准备 & 增强
│   └── api/main.py               # 旧版 FastAPI（Google Cloud Run）
│
├── scripts/                      # 部署 & 数据脚本
│   ├── deploy_space.py           # 推送 space/ → HF Spaces
│   ├── hf_upload.py              # 上传模型 → HF Model Hub
│   ├── dataset_pack.py           # 打包 data/ → tar.gz
│   ├── dataset_upload.py         # 上传数据 → HF Dataset Hub
│   └── dataset_download.py       # 从 HF 下载并解压数据
│
├── configs/detect/               # YOLOv8 训练配置
├── notebooks/                    # Jupyter 实验（探索性）
├── tests/                        # 泛化测试
├── docs/                         # 图片 & 文档资源
└── data/                         # 托管在 HF Dataset (gitignored)
```

---

## 快速开始

### 试用演示（无需安装）

访问 **[在线演示](https://huggingface.co/spaces/jeffliulab/card-calibration-v1)**，上传包含校准卡的照片即可获取结果。

### 本地运行

```bash
# 克隆仓库
git clone https://github.com/jeffliulab/Color_Calibration.git
cd Color_Calibration

# 安装
pip install -e .

# 本地运行 Gradio 演示
cd space && pip install -r requirements.txt
python app.py
```

### 训练

```bash
# 从 HuggingFace 下载数据集（一行命令，无需任何凭证）
python scripts/dataset_download.py

# 训练 XGBoost
python src/train/train_xgboost.py

# 超参数调优
python src/tune/tune_xgboost.py
```

### 部署到 HuggingFace

```bash
# 上传模型到 HF Hub
python scripts/hf_upload.py --all

# 部署 Space
python scripts/deploy_space.py --space_id jeffliulab/card-calibration-v1

# 打包并上传数据集
python scripts/dataset_pack.py
python scripts/dataset_upload.py
```

---

## 数据

完整数据集（255 张原始照片 + 增强变体 + YOLO 标注 + 色块裁剪，约 360 MB）托管在 **[HuggingFace Datasets](https://huggingface.co/datasets/jeffliulab/card-calibration-v1-data)**。

### 下载

```bash
python scripts/dataset_download.py
```

下载约 360 MB 数据并解压到 `data/`，目录结构与所有训练脚本期望的完全一致。**无需 GCP 凭证、无需 DVC、无需任何外部工具**。

### 内容

| 压缩包 | 文件数 | 说明 |
|---|---|---|
| `raw_photos.tar.gz` | 255 | 原始采集照片 |
| `augmented_images.tar.gz` | 2,294 | Albumentations 增强变体 |
| `feature_crops.tar.gz` | 9,154 | 色块中心区域裁剪 |
| `yolo_card_dataset.tar.gz` | 510 | YOLO 第一阶段 train/val/test |
| `yolo_pattern_dataset.tar.gz` | 510 | YOLO 第二阶段 train/val/test |
| `yolo_labeled_card.tar.gz` | 510 | YOLO 标注源（卡片） |
| `yolo_labeled_patterns.tar.gz` | 510 | YOLO 标注源（色块） |
| `generalization_test.png` | 1 | 泛化测试图 |
| `features/*.csv` | 3 | 特征 CSV（训练 + 中间） |

---

## 相关链接

| 资源 | 链接 |
|------|------|
| 在线演示 | [HuggingFace Space](https://huggingface.co/spaces/jeffliulab/card-calibration-v1) |
| 模型权重 | [HuggingFace Model Hub](https://huggingface.co/jeffliulab/card-calibration-v1) |
| 数据集 | [HuggingFace Datasets](https://huggingface.co/datasets/jeffliulab/card-calibration-v1-data) |
| GitHub | [jeffliulab/Color_Calibration](https://github.com/jeffliulab/Color_Calibration) |

---

## 致谢

- 开发于 Brandeis University CS149 Practical Machine Learning 课程（2025 Spring）
- YOLOv8 来自 [Ultralytics](https://github.com/ultralytics/ultralytics)
- 托管于 [HuggingFace](https://huggingface.co)
