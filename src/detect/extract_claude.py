from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple

class PatternProcessor:
    def __init__(self):
        self.ROOT_DIR = Path(__file__).resolve().parent.parent.parent
        
        # 加载模型
        self.model1 = YOLO(str(self.ROOT_DIR / "data/models/detect/yolo_0214_first.pt"))
        self.model2 = YOLO(str(self.ROOT_DIR / "data/models/detect/yolo_0214_second.pt"))
        
        # 输入输出路径
        self.input_dir = self.ROOT_DIR / "data/processed/aug_patterns/20250215"
        self.output_dir = self.ROOT_DIR / "outputs/patterns/20250215"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pattern 类型映射
        self.pattern_names = ["black_box", "red_circle", "green_triangle", "blue_pentagon"]

    def crop_card(self, image: np.ndarray) -> Optional[np.ndarray]:
        """裁剪校准卡片"""
        # 使用 model1 识别卡片区域
        results1 = self.model1(image)
        boxes = results1[0].boxes if results1 else []

        if len(boxes) > 0:
            box = boxes[0]  # 取第一个目标框
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 确保裁剪范围在图像边界内
            h, w, _ = image.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            # 裁剪卡片
            image_crop = image[y1:y2, x1:x2]
            print(f"Card cropped. Size: {image_crop.shape}")
            return image_crop
        else:
            print("Card Not Recognized")
            return None

    def extract_patterns(self, image_crop: np.ndarray) -> Optional[List[Dict]]:
        """提取并识别图案"""
        if image_crop is None:
            return None

        # 使用 model2 识别图案
        results2 = self.model2(image_crop)
        boxes = results2[0].boxes if results2 else []
        
        # 如果没有识别到恰好4个图案，返回None
        if len(boxes) != 4:
            print(f"Expected 4 patterns, but found {len(boxes)}")
            return None

        patterns = []
        # 获取所有检测框和类别
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # 裁剪图案
            pattern_crop = image_crop[y1:y2, x1:x2]
            
            # 获取图案类型名称
            pattern_type = self.pattern_names[class_id]
            
            # 计算中心点（用于排序）
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            patterns.append({
                'image': pattern_crop,
                'type': pattern_type,
                'confidence': confidence,
                'center': (center_x, center_y),
                'bbox': (x1, y1, x2, y2)
            })

        # 按位置排序（从左到右，从上到下）
        patterns.sort(key=lambda x: (x['center'][1], x['center'][0]))
        
        return patterns

    def save_patterns(self, patterns: List[Dict], source_path: Path) -> bool:
        """保存识别到的图案"""
        try:
            # 获取原始文件名（不含扩展名）
            base_name = source_path.stem
            
            # 验证是否有全部4种图案
            pattern_types = [p['type'] for p in patterns]
            if len(set(pattern_types)) != 4:
                print(f"Warning: Duplicate pattern types found in {pattern_types}")
                return False
            
            # 构建新的文件名（包含所有pattern类型）
            pattern_suffix = '_'.join(pattern_types)
            output_name = f"{base_name}_{pattern_suffix}.jpg"
            output_path = self.output_dir / output_name
            
            # 保存包含标注的图像
            annotated_image = None  # 这里可以添加标注逻辑
            cv2.imwrite(str(output_path), annotated_image if annotated_image is not None else patterns[0]['image'])
            
            print(f"Saved patterns to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving patterns: {e}")
            return False

    def process_image(self, image_path: Path) -> bool:
        """处理单张图片"""
        try:
            # 读取图片
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not read image: {image_path}")
                return False

            # 裁剪卡片
            cropped_card = self.crop_card(image)
            if cropped_card is None:
                return False

            # 提取图案
            patterns = self.extract_patterns(cropped_card)
            if patterns is None or len(patterns) != 4:
                return False

            # 保存图案
            return self.save_patterns(patterns, image_path)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return False

    def process_directory(self):
        """处理整个目录"""
        total = 0
        successful = 0
        
        for image_path in self.input_dir.glob("*.jpg"):
            total += 1
            if self.process_image(image_path):
                successful += 1
                
        print(f"\nProcessing complete: {successful}/{total} images processed successfully")
        return successful, total

def main():
    processor = PatternProcessor()
    processor.process_directory()

if __name__ == "__main__":
    main()