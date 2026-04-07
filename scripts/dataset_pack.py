"""
打包本地 data/ 为 tar.gz，准备上传到 HuggingFace Datasets。

按"逻辑数据阶段"分组打包，避免上传 13k+ 小文件。
打包结果输出到 _dataset_archive/，由 dataset_upload.py 上传。

用法:
    python scripts/dataset_pack.py
    python scripts/dataset_pack.py --data_dir data --output _dataset_archive
"""

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 打包映射: (本地源目录或文件, 压缩包名, 描述)
ARCHIVES = [
    ("data/processed/patterns/images_0215",        "raw_photos.tar.gz",          "原始照片 (255 张)"),
    ("data/processed/aug_patterns/20250215",       "augmented_images.tar.gz",    "增强图片 (2,294 张)"),
    ("data/processed/features/0215",               "feature_crops.tar.gz",       "色块裁剪 (9,154 个)"),
    ("data/train/detect/yolo_first_0214",          "yolo_card_dataset.tar.gz",   "YOLO 第一阶段训练集"),
    ("data/train/detect/yolo_second_0214",         "yolo_pattern_dataset.tar.gz","YOLO 第二阶段训练集"),
    ("data/processed/detect/yolo_labeled_card_0214", "yolo_labeled_card.tar.gz", "YOLO 标注源 (卡片)"),
    ("data/processed/detect/yolo_labeled_patterns_0214", "yolo_labeled_patterns.tar.gz", "YOLO 标注源 (色块)"),
]

# 直接复制的单个文件: (源, 目标相对路径)
SINGLE_FILES = [
    ("data/images/generalization/test.png",         "generalization_test.png"),
    ("data/features/feature_0216.csv",              "features/feature_0216.csv"),
    ("data/features/middle_file/GridMean_0216.csv", "features/GridMean_0216.csv"),
    ("data/features/middle_file/unique_colors.csv", "features/unique_colors.csv"),
]


def _human_size(num_bytes: int) -> str:
    """将字节数转为人类可读格式。"""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def pack_directory(src_dir: Path, archive_path: Path, description: str) -> bool:
    """将目录打包为 tar.gz。"""
    if not src_dir.exists():
        print(f"  [skip] {src_dir} 不存在")
        return False

    file_count = sum(1 for _ in src_dir.rglob("*") if _.is_file())
    if file_count == 0:
        print(f"  [skip] {src_dir} 为空")
        return False

    print(f"  [pack] {description}")
    print(f"         {src_dir} ({file_count} files)")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        # arcname 设为目录名,解压时保留顶层目录
        tar.add(src_dir, arcname=src_dir.name)

    size = archive_path.stat().st_size
    print(f"         -> {archive_path.name} ({_human_size(size)})")
    return True


def copy_single_file(src_file: Path, dst_file: Path) -> bool:
    """复制单个文件到输出目录。"""
    if not src_file.exists():
        print(f"  [skip] {src_file} 不存在")
        return False

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)
    size = dst_file.stat().st_size
    print(f"  [copy] {src_file.name} -> {dst_file.relative_to(dst_file.parent.parent)} ({_human_size(size)})")
    return True


def main():
    parser = argparse.ArgumentParser(description="打包 data/ 为 tar.gz 用于 HF Datasets 上传")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="本地数据目录 (相对项目根)")
    parser.add_argument("--output", type=str, default="_dataset_archive",
                        help="输出目录 (相对项目根)")
    parser.add_argument("--clean", action="store_true",
                        help="先清空输出目录再打包")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    output_dir = ROOT / args.output

    if not data_dir.exists():
        print(f"ERROR: 数据目录不存在: {data_dir}")
        sys.exit(1)

    if args.clean and output_dir.exists():
        print(f"清空输出目录: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== 打包目录到 tar.gz ===")
    packed = 0
    for rel_src, archive_name, description in ARCHIVES:
        src = ROOT / rel_src
        archive = output_dir / archive_name
        if pack_directory(src, archive, description):
            packed += 1

    print(f"\n=== 复制单个文件 ===")
    copied = 0
    for rel_src, rel_dst in SINGLE_FILES:
        src = ROOT / rel_src
        dst = output_dir / rel_dst
        if copy_single_file(src, dst):
            copied += 1

    # 统计
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    total_files = sum(1 for f in output_dir.rglob("*") if f.is_file())

    print(f"\n=== 完成 ===")
    print(f"打包: {packed}/{len(ARCHIVES)} 个 tar.gz")
    print(f"复制: {copied}/{len(SINGLE_FILES)} 个单文件")
    print(f"输出: {output_dir}")
    print(f"总计: {total_files} 个文件, {_human_size(total_size)}")


if __name__ == "__main__":
    main()
