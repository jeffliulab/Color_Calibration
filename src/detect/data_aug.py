import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 输入图像目录（待增强）
INPUT_DIR = ROOT_DIR / "data/processed/patterns/images_0215"

# 输出目录（增强后）
DATE = datetime.now().strftime("%Y%m%d")
OUTPUT_DIR = ROOT_DIR / f"outputs/aug/{DATE}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 颜色编号的增强计数，确保同一颜色编号的不同图像接续编号
augmentation_counters = defaultdict(int)

def add_shadow(image):
    h, w = image.shape[:2]
    top_x, top_y = np.random.randint(0, w // 2), np.random.randint(0, h // 2)
    bot_x, bot_y = np.random.randint(w // 2, w), np.random.randint(h // 2, h)

    shadow_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.rectangle(shadow_mask, (top_x, top_y), (bot_x, bot_y), (50, 50, 50), -1)
    alpha = np.random.uniform(0.5, 0.7)
    return cv2.addWeighted(image, 1 - alpha, shadow_mask, alpha, 0)

def get_base_name(file_path):
    """提取文件的基础名称（去除序号）"""
    name = file_path.stem
    # 假设格式为 "base_name_123" 或 "base-name-123"
    parts = name.replace('-', '_').split('_')
    
    # 如果最后一部分是数字，则移除
    if parts[-1].isdigit():
        return '_'.join(parts[:-1])
    return name

# 数据增强 Pipeline
augmentations = [
    ("brightness", A.RandomBrightnessContrast(p=1.0)),
    ("hue", A.HueSaturationValue(p=1.0)),
    ("gamma", A.RandomGamma(p=1.0)),
    ("motion_blur", A.MotionBlur(blur_limit=5, p=1.0)),
    ("gaussian_blur", A.GaussianBlur(blur_limit=5, p=1.0)),
    ("clahe", A.CLAHE(clip_limit=4.0, p=1.0)),
    ("noise", A.ISONoise(p=1.0)),
    ("rotate", A.Rotate(limit=10, p=1.0, border_mode=cv2.BORDER_REFLECT)),
]

def process_image(img_path):
    """处理单张图片"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Warning: Unable to read {img_path}")
        return False

    # 获取基础文件名
    base_name = get_base_name(img_path)
    print(f"Processing: {img_path.name} → base_name: {base_name}")  # 调试信息

    # 确保编号从上次基础上递增
    start_idx = augmentation_counters[base_name] + 1

    # 进行数据增强
    for i, (aug_type, aug) in enumerate(augmentations):
        try:
            aug_img = aug(image=img)["image"]
            # 更有意义的文件名格式：原始名称_增强类型_序号.jpg
            output_path = OUTPUT_DIR / f"{base_name}_{aug_type}_{start_idx + i}.jpg"
            cv2.imwrite(str(output_path), aug_img)
            print(f"  ✓ Created: {output_path.name}")
        except Exception as e:
            print(f"  ✗ Error processing {aug_type}: {e}")

    # 添加阴影版本
    try:
        shadow_img = add_shadow(img)
        shadow_path = OUTPUT_DIR / f"{base_name}_shadow_{start_idx + len(augmentations)}.jpg"
        cv2.imwrite(str(shadow_path), shadow_img)
        print(f"  ✓ Created shadow: {shadow_path.name}")
    except Exception as e:
        print(f"  ✗ Error adding shadow: {e}")

    # 更新编号记录
    augmentation_counters[base_name] += len(augmentations) + 1
    return True

def main():
    print(f"🔍 Input directory: {INPUT_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    success_count = 0
    total_files = 0
    
    for img_path in sorted(INPUT_DIR.glob("*.jpg")):
        total_files += 1
        if process_image(img_path):
            success_count += 1
            
    print(f"\n🎉 Augmentation complete!")
    print(f"📊 Processed {success_count}/{total_files} images")
    print(f"💾 Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()