import os
import cv2
import numpy as np

# 输入和输出路径
input_folder = "0_input"  # 颜色卡的裁剪图片
output_folder = "4_tradition"  # 处理后的颜色卡

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

def detect_card(image):
    """ 使用边缘检测 + 霍夫变换 + 轮廓检测 找到颜色卡 """

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # **步骤 1：使用 Canny 进行边缘检测**
    edges = cv2.Canny(gray, 50, 150)

    # **步骤 2：使用霍夫变换找直线边界**
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=5)

    if lines is None:
        return None  # 如果没有检测到直线，则放弃处理

    # **步骤 3：找到所有轮廓**
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # **步骤 4：遍历轮廓，找到近似矩形**
    best_rect = None
    max_area = 0

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # 只考虑四边形
            area = cv2.contourArea(approx)
            if area > max_area:  # 找到最大面积的矩形
                best_rect = approx
                max_area = area

    return best_rect

# 遍历所有图片
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # ❶ **检测颜色卡的矩形边界**
        card_rect = detect_card(image)

        if card_rect is None:
            print(f"⚠️ {filename} CANNOT RECOGNIZE")
            continue

        # ❷ **生成白色掩码（仅保留颜色卡内部区域）**
        mask = np.ones_like(image, dtype=np.uint8) * 255  # 生成全白背景
        cv2.fillPoly(mask, [card_rect], (0, 0, 0))  # 只保留矩形区域

        # **应用掩码**
        result = cv2.bitwise_or(image, mask)

        # ❸ **保存最终图片**
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result)

print("✅ 颜色卡检测完成，结果已保存到:", output_folder)
