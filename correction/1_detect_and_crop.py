import os
import cv2
from ultralytics import YOLO

# YOU DON"T NEED TO RUN 0_object_detect BEFORE RUNNING THIS SCRIPT!!
# DIRECTLY RUN THIS SCRIPT IS FINE!!

# (terminal) detect & crop >> yolo task=detect mode=predict model=best.pt source=input/ save_crop=True


# 定义路径
input_folder = "input"  # 输入图片文件夹
output_folder = "output1_crop"  # 结果输出文件夹
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

        # 读取原始图片（用于裁剪）
        image = cv2.imread(image_path)

        # 处理每个检测结果
        for i, result in enumerate(results):
            for j, box in enumerate(result.boxes.xyxy):  # 遍历所有检测框
                x1, y1, x2, y2 = map(int, box)  # 获取检测框的坐标
                cropped = image[y1:y2, x1:x2]  # 裁剪目标区域

                # 生成输出路径
                crop_output_path = os.path.join(output_folder, f"{filename}_crop_{j}.jpg")
                
                # 保存裁剪后的图片
                cv2.imwrite(crop_output_path, cropped)

print("✅ 目标裁剪完成，裁剪后的图片已保存到:", output_folder)
