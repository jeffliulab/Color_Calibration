import os
import cv2
import numpy as np

# 输入和输出路径
input_folder = "0_input"  # 颜色卡的裁剪图片
output_folder = "5_pattern"  # 处理后的颜色卡，仅保留内框、三角形和五边形

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

def detect_shapes(image):
    """ 检测颜色卡中的内黑框、三角形、五边形 """

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # **步骤 1：使用 Canny 进行边缘检测**
    edges = cv2.Canny(gray, 50, 150)

    # **步骤 2：轮廓检测**
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_shapes = []  # 存储所有检测到的形状

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # 过滤掉太小的噪声
            continue

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_sides = len(approx)

        if num_sides == 4:
            detected_shapes.append(approx)  # 可能是内黑框
        elif num_sides == 3:
            detected_shapes.append(approx)  # 可能是三角形
        elif num_sides == 5:
            detected_shapes.append(approx)  # 可能是五边形

    return detected_shapes

# 遍历所有图片
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # ❶ **检测颜色卡的关键形状**
        shapes = detect_shapes(image)

        if len(shapes) < 2:  # 需要至少检测到内黑框 + 一个形状
            print(f"⚠️ {filename} CANNOT FIND KEY PATTERN")
            continue

        # ❷ **生成白色掩码（仅保留内黑框 + 三角形 + 五边形）**
        mask = np.ones_like(image, dtype=np.uint8) * 255  # 生成全白背景
        cv2.fillPoly(mask, shapes, (0, 0, 0))  # 只保留关键形状

        # **应用掩码**
        result = cv2.bitwise_or(image, mask)

        # ❸ **保存最终图片**
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result)

print("✅ 颜色卡检测完成，仅保留内黑框 + 三角形 + 五边形，结果已保存到:", output_folder)
