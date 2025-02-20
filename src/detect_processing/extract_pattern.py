from ultralytics import YOLO
import cv2
import os
import numpy as np
from pathlib import Path

# 定义ROOT_DIR
ROOT_DIR = Path(__file__).resolve().parent.parent.parent 

# 加载YOLOv8模型
model1_path = str(ROOT_DIR / "data/models/detect/yolo_0214_first.pt")
model1 = YOLO(model1_path)

model2_path = str(ROOT_DIR / "data/models/detect/yolo_0214_second.pt")
model2 = YOLO(model2_path)

# 处理图片的文件夹
images_path = Path(ROOT_DIR / "data/processed/aug_patterns/20250215/")
save_path = Path(ROOT_DIR / "outputs/aug_patterns/processed/")
save_path.mkdir(parents=True, exist_ok=True)  # 确保保存路径存在

def crop_card(image):
    """使用model1检测并裁剪校准卡"""
    results1 = model1(image)
    boxes = results1[0].boxes if results1 else []
    
    if len(boxes) > 0:
        box = boxes[0]  # 取第一个检测到的框
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w, _ = image.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        return image[y1:y2, x1:x2]  # 裁剪区域
    return None

def extract_patterns(image_crop, image_file):
    """使用model2识别patterns并保存每个pattern的裁剪图片"""
    results2 = model2(image_crop)
    boxes = results2[0].boxes if results2 else []
    detected_patterns = []
    saved_files = 0
    
    for box in boxes:
        cls_idx = int(box.cls[0])
        pattern_name = results2[0].names[cls_idx]  # 获取YOLO识别的pattern名称
        detected_patterns.append(pattern_name)
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w, _ = image_crop.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        pattern_crop = image_crop[y1:y2, x1:x2]
        
        save_filename = f"{image_file.stem}_{pattern_name}{image_file.suffix}"
        cv2.imwrite(str(save_path / save_filename), pattern_crop)
        saved_files += 1
        print(f"保存图像: {save_filename}")
    
    return detected_patterns, saved_files

def process_images():
    """遍历文件夹中的所有图片，并处理pattern"""
    total_images = 0
    total_saved = 0
    failed_images = 0
    
    for image_file in images_path.glob("*.jpg"):
        total_images += 1
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"无法读取图像: {image_file}")
            failed_images += 1
            continue
        
        # 先裁剪校准卡
        image_crop = crop_card(image)
        if image_crop is None:
            print(f"未识别到校准卡: {image_file}")
            failed_images += 1
            continue
        
        # 提取patterns并保存每个pattern的裁剪图片
        detected_patterns, saved_files = extract_patterns(image_crop, image_file)
        total_saved += saved_files
        
        # 只有识别到4个patterns才保存
        if len(set(detected_patterns)) == 4:
            print(f"{image_file} 识别到完整patterns: {detected_patterns}")
        else:
            print(f"{image_file} 识别不完整, 跳过")
            failed_images += 1
    
    print(f"处理完成: 总图片数 {total_images}, 生成文件数 {total_saved}, 失败图片数 {failed_images}")

if __name__ == "__main__":
    process_images()
