# LOAD ALL NECESSARY LIBRARIES AND MODELS

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent 
# print(f"ROOT_DIR: {ROOT_DIR}") # Keep this line for copying and test; Using: str(ROOT_DIR / "data/..")

# Load the first YOLOv8 model
model1_path = str(ROOT_DIR / "data/models/detect/yolo_0214_first.pt")
model1 = YOLO(model1_path)

# Load the second YOLOv8 model
model2_path = str(ROOT_DIR / "data/models/detect/yolo_0214_second.pt")
model2 = YOLO(model2_path)

# Images with augmentation folder
images_path = str(ROOT_DIR / "data/processed/aug_patterns/20250215/")

"""
crop calibration card firstly
"""
def crop_card(init):
    # Use model1 to narrow the recognizing area
    results1 = model1(image)

    # Get the detection frame
    boxes = results1[0].boxes if results1 else []

    if len(boxes) > 0:
        box = boxes[0]  # Take the first target box
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integer

        # Make sure the cropping range is within the image boundaries
        h, w, _ = image.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        # Cut the card
        image_crop = image[y1:y2, x1:x2]

        print(f"Image cropped. Size: {image_crop.shape}")
    else:
        print("Card Not Recognized")
        image_crop = None

"""
extract patterns
"""
def extract_patterns(init):
    # Recognize patterns based on cropped image
    if image_crop is not None:
        # use model2 to recognize pattern
        results2 = model2(image_crop)

        # show target box of each pattern
        annotated_crop = results2[0].plot()
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(annotated_crop, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("YOLOv8 Pattern Recognized")
        plt.show()
    else:
        print("Failed")



# 程序逻辑：

1、 遍历文件夹，对于每个图片进行pattern处理
2、 处理pattern的时候，先crop，crop后用model2（yolov8）进行识别，会识别到四个pattern：
pattern的内容如下：
names: ["black_box", "red_circle", "green_triangle", "blue_pentagon"]
把识别到的pattern作为后缀加在文件名上
3、然后要注意，如果没有识别到4个patterns，那么就不要存储那张照片了，一定要保证4个pattern都识别到再保存