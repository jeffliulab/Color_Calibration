"""
从 HuggingFace Datasets 下载并解压数据到本地 data/。

这是用户/未来自己复现项目的入口。下载完成后,本地 data/ 结构与
src/ 中所有路径引用完全一致,可以直接运行训练脚本。

用法:
    # 默认: 下载到 ./data
    python scripts/dataset_download.py

    # 指定目标目录
    python scripts/dataset_download.py --data_dir my_data

    # 强制覆盖已有目录
    python scripts/dataset_download.py --force
"""

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REPO = "jeffliulab/card-calibration-v1-data"

# 解压映射: (HF 上的压缩包名, 解压到的本地目录的父目录)
# tar 内部已经包含一层目录,所以这里指定的是父目录
EXTRACT_MAP = [
    ("raw_photos.tar.gz",            "data/processed/patterns"),
    ("augmented_images.tar.gz",      "data/processed/aug_patterns"),
    ("feature_crops.tar.gz",         "data/processed/features"),
    ("yolo_card_dataset.tar.gz",     "data/train/detect"),
    ("yolo_pattern_dataset.tar.gz",  "data/train/detect"),
    ("yolo_labeled_card.tar.gz",     "data/processed/detect"),
    ("yolo_labeled_patterns.tar.gz", "data/processed/detect"),
]

# 单文件复制映射: (HF 上的相对路径, 本地目标相对路径)
SINGLE_FILE_MAP = [
    ("generalization_test.png",         "data/images/generalization/test.png"),
    ("features/feature_0216.csv",       "data/features/feature_0216.csv"),
    ("features/GridMean_0216.csv",      "data/features/middle_file/GridMean_0216.csv"),
    ("features/unique_colors.csv",      "data/features/middle_file/unique_colors.csv"),
]


def _human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def extract_archive(archive_path: Path, target_parent: Path) -> int:
    """解压 tar.gz 到目标父目录,返回解压的文件数。"""
    target_parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        tar.extractall(target_parent)
    return sum(1 for m in members if m.isfile())


def main():
    parser = argparse.ArgumentParser(description="从 HF Datasets 下载 card calibration 数据")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO,
                        help=f"HF dataset repo id (默认: {DEFAULT_REPO})")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="本地数据目录 (相对项目根)")
    parser.add_argument("--cache_dir", type=str, default="_dataset_cache",
                        help="HF 下载缓存目录 (相对项目根)")
    parser.add_argument("--force", action="store_true",
                        help="即使 data/ 已存在也强制重新下载")
    parser.add_argument("--keep_cache", action="store_true",
                        help="解压后保留下载缓存 (默认会删除节省空间)")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    cache_dir = ROOT / args.cache_dir

    # 检查是否已存在
    if data_dir.exists() and not args.force:
        existing = sum(1 for _ in data_dir.rglob("*") if _.is_file())
        if existing > 0:
            print(f"data/ 已存在 ({existing} 个文件)。使用 --force 强制重新下载。")
            sys.exit(0)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        sys.exit(1)

    # 1. 下载整个 dataset repo
    print(f"\n[1/3] 从 HF 下载 dataset: {args.repo_id}")
    print(f"      缓存目录: {cache_dir}")
    snapshot_path = Path(snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(cache_dir),
    ))
    print(f"      OK")

    # 2. 解压所有 tar.gz
    print(f"\n[2/3] 解压压缩包到 {data_dir}")
    total_extracted = 0
    for archive_name, target_rel in EXTRACT_MAP:
        archive_path = snapshot_path / archive_name
        target_parent = ROOT / target_rel
        if not archive_path.exists():
            print(f"  [skip] {archive_name} 不存在")
            continue
        size = archive_path.stat().st_size
        print(f"  [extract] {archive_name} ({_human_size(size)}) -> {target_rel}/")
        count = extract_archive(archive_path, target_parent)
        print(f"            -> {count} files")
        total_extracted += count

    # 3. 复制单个文件
    print(f"\n[3/3] 复制单个文件")
    copied = 0
    for src_rel, dst_rel in SINGLE_FILE_MAP:
        src = snapshot_path / src_rel
        dst = ROOT / dst_rel
        if not src.exists():
            print(f"  [skip] {src_rel} 不存在")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  [copy] {src_rel} -> {dst_rel}")
        copied += 1

    # 清理缓存
    if not args.keep_cache and cache_dir.exists():
        print(f"\n清理下载缓存: {cache_dir}")
        shutil.rmtree(cache_dir)

    # 统计最终结果
    final_count = sum(1 for _ in data_dir.rglob("*") if _.is_file())
    final_size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())

    print(f"\n=== 完成 ===")
    print(f"解压: {total_extracted} 个文件 (来自 tar.gz)")
    print(f"复制: {copied} 个文件 (单文件)")
    print(f"data/ 总计: {final_count} 个文件, {_human_size(final_size)}")


if __name__ == "__main__":
    main()
