import os
import cv2
import numpy as np

# 输入和输出路径
input_folder = "output1_crop"  # 经过YOLO裁剪后的颜色卡
output_folder = "output2_card"  # 仅检测内黑框颜色区域

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

def detect_color_region(image):
    """ 通过中心点颜色，检测连续的色块区域 """

    h, w = image.shape[:2]
    
    # **获取图片中心点颜色**
    cx, cy = w // 2, h // 2
    center_color = image[cy, cx]  # 取中心点 BGR 颜色

    # **转换到 HSV 颜色空间**
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    center_hsv = cv2.cvtColor(np.uint8([[center_color]]), cv2.COLOR_BGR2HSV)[0][0]

    # **设定颜色范围**
    h, s, v = center_hsv
    lower_bound = np.array([max(0, h - 20), max(30, s - 40), max(30, v - 50)])  # 允许颜色波动
    upper_bound = np.array([min(180, h + 20), min(255, s + 40), min(255, v + 50)])

    # **创建颜色掩码**
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # **形态学处理，填补色块缺失部分**
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # **轮廓检测，找到最大色块**
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # 选择最大连续色块
    max_contour = max(contours, key=cv2.contourArea)

    return max_contour, mask

# 遍历所有图片
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # ❶ **检测中心颜色对应的色块区域**
        color_region, mask = detect_color_region(image)

        if color_region is None:
            print(f"⚠️ {filename} 未检测到稳定的颜色区域，跳过")
            continue

        # ❷ **绘制检测到的色块**
        result = image.copy()
        cv2.drawContours(result, [color_region], -1, (0, 255, 0), 3)

        # ❸ **保存最终图片**
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result)

print("✅ 颜色区域检测完成，结果已保存到:", output_folder)
