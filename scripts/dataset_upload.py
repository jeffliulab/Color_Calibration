"""
将打包好的 _dataset_archive/ 上传到 HuggingFace Datasets。

会自动创建 dataset repo (如果不存在)，并写入 Dataset Card README.md。

用法:
    # 先运行 dataset_pack.py 生成 _dataset_archive/
    python scripts/dataset_pack.py

    # 然后上传
    python scripts/dataset_upload.py
    python scripts/dataset_upload.py --repo_id jeffliulab/card-calibration-v1-data
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REPO = "jeffliulab/card-calibration-v1-data"

DATASET_CARD = """\
---
license: mit
language: en
tags:
  - image
  - color-calibration
  - object-detection
  - yolo
size_categories:
  - 1K<n<10K
---

# Card Calibration Dataset (v1)

255 hand-collected photos of color calibration cards, with augmentations,
YOLO annotations, and extracted feature crops.

Used to train [`jeffliulab/card-calibration-v1`](https://huggingface.co/jeffliulab/card-calibration-v1).
Live demo: [`jeffliulab/card-calibration-v1` Space](https://huggingface.co/spaces/jeffliulab/card-calibration-v1).

## Contents

| Archive | Files | Description |
|---|---|---|
| `raw_photos.tar.gz` | 255 | Original hand-collected photos |
| `augmented_images.tar.gz` | 2,294 | Albumentations-augmented variants (brightness/hue/blur/etc) |
| `feature_crops.tar.gz` | 9,154 | Center-region patch crops (4 patches × ~2,294 images) |
| `yolo_card_dataset.tar.gz` | 510 | YOLO stage-1 training set (card detection) |
| `yolo_pattern_dataset.tar.gz` | 510 | YOLO stage-2 training set (pattern detection) |
| `yolo_labeled_card.tar.gz` | 510 | Raw YOLO annotations (stage 1) |
| `yolo_labeled_patterns.tar.gz` | 510 | Raw YOLO annotations (stage 2) |
| `generalization_test.png` | 1 | Held-out generalization test photo |
| `features/feature_0216.csv` | 1 | Final training feature CSV (12-D + target) |
| `features/GridMean_0216.csv` | 1 | Intermediate grid mean values |
| `features/unique_colors.csv` | 1 | Unique color reference values |

## Usage

The companion repo provides a one-command download script:

```bash
git clone https://github.com/jeffliulab/Color_Calibration.git
cd Color_Calibration
python scripts/dataset_download.py
```

This downloads all archives and extracts them into `data/` matching the layout
expected by the training scripts in `src/`.

## Manual download

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="jeffliulab/card-calibration-v1-data",
    repo_type="dataset",
    local_dir="./dataset_cache",
)
```

## Known issues

6 augmented feature crop files are missing (gsutil transfer failures during
the migration from GCS). These are non-critical center-region crops and do not
affect training or evaluation.

## License

MIT — both data and code are freely available for research and commercial use.
"""


def main():
    parser = argparse.ArgumentParser(description="上传 _dataset_archive/ 到 HF Datasets")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO,
                        help=f"HF dataset repo id (默认: {DEFAULT_REPO})")
    parser.add_argument("--archive_dir", type=str, default="_dataset_archive",
                        help="打包好的目录")
    parser.add_argument("--message", type=str, default="Upload dataset archives",
                        help="commit message")
    args = parser.parse_args()

    archive_dir = ROOT / args.archive_dir
    if not archive_dir.exists():
        print(f"ERROR: 打包目录不存在: {archive_dir}")
        print(f"先运行: python scripts/dataset_pack.py")
        sys.exit(1)

    files = sorted(archive_dir.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    if file_count == 0:
        print(f"ERROR: {archive_dir} 中没有文件")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    # 1. 创建 repo (幂等)
    print(f"\n[1/3] 确保 dataset repo 存在: {args.repo_id}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
    )
    print(f"      OK")

    # 2. 写入 README (Dataset Card)
    print(f"\n[2/3] 写入 Dataset Card")
    readme_path = archive_dir / "README.md"
    readme_path.write_text(DATASET_CARD, encoding="utf-8")
    print(f"      {readme_path}")

    # 3. 上传整个目录
    print(f"\n[3/3] 上传 {file_count} 个文件到 HF...")
    api.upload_folder(
        folder_path=str(archive_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.message,
    )

    print(f"\n=== 完成 ===")
    print(f"View: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
