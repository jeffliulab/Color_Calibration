"""
Upload model weights and Model Card to HuggingFace Model Hub.

Usage:
    # Upload all models + Model Card
    python scripts/hf_upload.py --all

    # Upload only the Model Card README
    python scripts/hf_upload.py --card-only

    # Upload a specific model file
    python scripts/hf_upload.py --file data/models/xgboost/model_v1.pkl --dest xgboost_v1.pkl

HuggingFace repo: https://huggingface.co/jeffliulab/card-calibration-v1
Token: 由 huggingface-cli login 自动管理 (~/.cache/huggingface/token)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

HF_REPO_ID = "jeffliulab/card-calibration-v1"
ROOT = Path(__file__).resolve().parent.parent

# 模型文件映射 (--all 时使用)
ALL_MODELS = {
    "yolo_first.pt":          "data/models/detect/yolo_0214_first.pt",
    "yolo_second.pt":         "data/models/detect/yolo_0214_second.pt",
    "xgboost_v1.pkl":         "data/models/xgboost/model_v1.pkl",
    "random_forest_v1.pkl":   "data/models/random_forest/model_v1.pkl",
}

# Model Card 内容,带 YAML frontmatter
MODEL_CARD = """\
---
license: mit
language: en
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - color-calibration
  - yolo
  - xgboost
  - random-forest
  - computer-vision
datasets:
  - jeffliulab/card-calibration-v1-data
---

# Card Calibration v1

Inference weights for the [Card Calibration](https://github.com/jeffliulab/Color_Calibration) project — automated color calibration via two-stage YOLO detection + tree-based regression.

**Best result:** XGBoost with Bayesian-tuned hyperparameters — **Lab Mean ΔE = 4.59** (meets commercial printing standards).

## Live Demo

Try it directly in your browser: **[HuggingFace Space](https://huggingface.co/spaces/jeffliulab/card-calibration-v1)**

## Files

| File | Description | Size |
|---|---|---|
| `yolo_first.pt` | YOLOv8-nano — calibration card detector | 6 MB |
| `yolo_second.pt` | YOLOv8-nano — 4-pattern detector (red/green/blue/black box) | 6 MB |
| `xgboost_v1.pkl` | XGBoost calibration model (best) | 3.5 MB |
| `random_forest_v1.pkl` | Random Forest calibration model | 45 MB |

## Quick Start

```python
from huggingface_hub import hf_hub_download
import joblib
from ultralytics import YOLO

REPO = "jeffliulab/card-calibration-v1"

# Download weights (cached in ~/.cache/huggingface/)
yolo_card_path    = hf_hub_download(repo_id=REPO, filename="yolo_first.pt")
yolo_pattern_path = hf_hub_download(repo_id=REPO, filename="yolo_second.pt")
xgb_path          = hf_hub_download(repo_id=REPO, filename="xgboost_v1.pkl")

# Load
yolo_card    = YOLO(yolo_card_path)
yolo_pattern = YOLO(yolo_pattern_path)
xgb_model    = joblib.load(xgb_path)
```

For the full inference pipeline (detect card → detect patterns → extract RGB → predict true color), see [`space/model_utils.py`](https://github.com/jeffliulab/Color_Calibration/blob/main/space/model_utils.py).

## Inference Pipeline

```
Photo ──▶ YOLO Stage 1 ──▶ Card Crop ──▶ YOLO Stage 2 ──▶ 4 Patches
                                                              │
                                          ┌───────────────────┘
                                          ▼
                                   Feature Engineering
                                   (9 deltas + 3 RGB)
                                          │
                                          ▼
                                   XGBoost / RF Model
                                          │
                                          ▼
                                   Predicted True RGB
```

## Performance

| Model | R² | RMSE | Lab Mean ΔE | Lab Median ΔE |
|---|---|---|---|---|
| **XGBoost (tuned)** | **0.8280** | **11.76** | **4.59** | **3.61** |
| Random Forest | 0.8225 | 12.10 | 5.20 | 3.96 |

> ΔE < 3: professional · ΔE < 5: commercial printing · ΔE < 10: acceptable

## Training Data

Dataset: **[`jeffliulab/card-calibration-v1-data`](https://huggingface.co/datasets/jeffliulab/card-calibration-v1-data)**

- 255 hand-collected photos of color calibration cards
- Augmented to 2,294 samples (brightness/hue/blur/noise/rotation)
- 12-D feature vectors extracted from center 1/3 of each detected patch
- 70/30 train/test split, `random_state=42`

## Links

- **GitHub:** https://github.com/jeffliulab/Color_Calibration
- **Live Demo:** https://huggingface.co/spaces/jeffliulab/card-calibration-v1
- **Dataset:** https://huggingface.co/datasets/jeffliulab/card-calibration-v1-data

## License

MIT — both data and model weights are freely available for research and commercial use.
"""


def upload_file(api, local_path: Path, dest_name: str, note: str = "") -> bool:
    """上传单个文件到 HF Model Hub。"""
    if not local_path.exists():
        print(f"  WARNING: 文件不存在: {local_path},跳过")
        return False

    timestamp = datetime.now().strftime("%Y%m%d")
    commit_msg = f"Upload {dest_name} [{timestamp}]"
    if note:
        commit_msg += f" — {note}"

    print(f"  Uploading: {local_path}")
    print(f"         To: {HF_REPO_ID}/{dest_name}")

    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=dest_name,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message=commit_msg,
    )
    print(f"  Done!")
    return True


def upload_model_card(api) -> bool:
    """上传 Model Card README.md 到 HF Model Hub。"""
    print(f"\n  Uploading Model Card to {HF_REPO_ID}/README.md")
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message=f"Update Model Card [{datetime.now().strftime('%Y%m%d')}]",
    )
    print(f"  Done!")
    return True


def main():
    parser = argparse.ArgumentParser(description="上传模型权重和 Model Card 到 HF Hub")
    parser.add_argument("--all", action="store_true",
                        help="上传所有模型 + Model Card")
    parser.add_argument("--card-only", action="store_true",
                        help="只更新 Model Card README")
    parser.add_argument("--file", type=str,
                        help="上传单个模型文件 (相对项目根)")
    parser.add_argument("--dest", type=str,
                        help="HF repo 中的目标文件名")
    parser.add_argument("--note", type=str, default="",
                        help="commit message 备注")
    args = parser.parse_args()

    if not (args.all or args.card_only or args.file):
        parser.error("指定 --all / --card-only / --file 之一")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()  # 使用 huggingface-cli login 保存的 token

    # 确保 repo 存在
    print(f"确保 model repo 存在: {HF_REPO_ID}")
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)

    if args.card_only:
        upload_model_card(api)
    elif args.all:
        print(f"\n上传所有模型到 {HF_REPO_ID}...")
        success = 0
        for dest_name, local_rel in ALL_MODELS.items():
            local_path = ROOT / local_rel
            if upload_file(api, local_path, dest_name, args.note):
                success += 1
        print(f"\n{success}/{len(ALL_MODELS)} 个模型上传成功")
        upload_model_card(api)
    else:
        local_path = ROOT / args.file if not Path(args.file).is_absolute() else Path(args.file)
        dest = args.dest or local_path.name
        upload_file(api, local_path, dest, args.note)

    print(f"\nView: https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
