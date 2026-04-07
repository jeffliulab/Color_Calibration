---
title: Card Calibration
emoji: "🎨"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.23.0"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
models:
  - jeffliulab/card-calibration-v1
datasets:
  - jeffliulab/card-calibration-v1-data
tags:
  - color-calibration
  - yolo
  - computer-vision
---

# Card Calibration — Live Demo

Upload a photo containing a color calibration card. The system detects the card,
extracts color reference patches, and predicts the true color of the target
patch under standard lighting.

## How It Works

1. **Card Detection** — YOLOv8 locates the calibration card in your photo
2. **Pattern Detection** — A second YOLOv8 identifies 4 patches: red circle, green triangle, blue pentagon, black box (target)
3. **Feature Engineering** — Extracts 12-D feature vector (9 reference deltas + 3 target RGB)
4. **Color Prediction** — XGBoost or Random Forest predicts the true RGB under standard lighting

**Best result:** XGBoost (Bayesian-tuned) — Lab Mean ΔE = 4.59 (commercial printing standard)

## Models

| File | Description |
|------|-------------|
| `yolo_first.pt` | YOLOv8-nano — card detection |
| `yolo_second.pt` | YOLOv8-nano — pattern detection |
| `xgboost_v1.pkl` | XGBoost calibration (best, ΔE=4.59) |
| `random_forest_v1.pkl` | Random Forest calibration |

Models are hosted on **[`jeffliulab/card-calibration-v1`](https://huggingface.co/jeffliulab/card-calibration-v1)** and downloaded automatically on first request via `huggingface_hub`.

## Training Data

255 hand-collected photos, augmented to 2,294 samples. Full dataset available on **[`jeffliulab/card-calibration-v1-data`](https://huggingface.co/datasets/jeffliulab/card-calibration-v1-data)**.

## Links

- **GitHub:** https://github.com/jeffliulab/Color_Calibration
- **Model Weights:** https://huggingface.co/jeffliulab/card-calibration-v1
- **Dataset:** https://huggingface.co/datasets/jeffliulab/card-calibration-v1-data
