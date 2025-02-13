import torch
import os
import cv2
from ultralytics import YOLO

# use terminal to detect >> yolo task=detect mode=predict model=best.pt source=input/

# 定义路径
input_folder = "input"  # 输入图片文件夹
output_folder = "output0_detect"  # 结果输出文件夹
model_path = "best.pt"  # 训练好的模型

# 加载 YOLO 模型
model = YOLO(model_path)

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历 input 文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):  # 仅处理图片文件
        image_path = os.path.join(input_folder, filename)
        
        # 运行 YOLO 目标检测
        results = model(image_path)

        # 获取检测结果并可视化
        for result in results:
            image_with_boxes = result.plot()  # 绘制检测框

            # 生成输出路径
            output_path = os.path.join(output_folder, filename)
            
            # 保存检测后的图片
            cv2.imwrite(output_path, image_with_boxes)

print("✅ 目标检测完成，结果已保存到:", output_folder)
