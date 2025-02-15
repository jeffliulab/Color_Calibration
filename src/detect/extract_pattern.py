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

