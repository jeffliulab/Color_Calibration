import os
import cv2
import numpy as np

# 输入和输出路径
input_folder = "0_input"  # 已裁剪的颜色卡
output_folder = "3_white_plus"  # 处理后的颜色卡

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

def detect_white_region(image):
    """ 识别白色区域，并返回轮廓 """

    # 转换到 HSV 颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # **扩展白色范围，适应不同光照**
    lower_white = np.array([0, 0, 140])
    upper_white = np.array([180, 50, 255])  # 扩展 S 让灰白色也能被检测

    # **创建白色掩码**
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # **形态学操作增强白色区域**
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 关闭运算填补小缺口
    mask = cv2.dilate(mask, kernel, iterations=1)  # 膨胀让白色区域更连贯

    # **找到所有白色区域轮廓**
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, mask

def get_largest_rectangle(contours):
    """ 获取最大白色区域的最小外接矩形 """
    if len(contours) == 0:
        return None
    
    max_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    return approx if len(approx) == 4 else None

# 遍历所有图片
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(input_folder, filename)
        
        # 读取图片
        image = cv2.imread(image_path)

        # ❶ **检测白色区域**
        contours, white_mask = detect_white_region(image)

        # ❷ **找到最大白色区域的矩形**
        white_rect = get_largest_rectangle(contours)

        if white_rect is None:
            print(f"⚠️ {filename} CANNOT FIND STABLE WHITE AREA")
            continue

        # ❸ **生成白色掩码（仅保留白色区域）**
        mask = np.ones_like(image, dtype=np.uint8) * 255  # 生成全白背景
        cv2.fillPoly(mask, [white_rect], (0, 0, 0))  # 只保留白色区域

        # **应用掩码**
        result = cv2.bitwise_or(image, mask)

        # ❹ **保存最终图片**
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result)

print("✅ 颜色卡处理完成，结果已保存到:", output_folder)
