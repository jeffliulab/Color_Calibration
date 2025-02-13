import os
import cv2
import numpy as np

# 输入和输出路径
input_folder = "output1_crop"  # 已经裁剪好的颜色卡图片
output_folder = "output2_corner"  # 角点处理后的图片存放路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历所有裁剪后的目标图片
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(input_folder, filename)
        
        # 读取裁剪后的图像
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ❶ **二值化图像，提取黑色区域**
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)  # 50以下的视为黑色
        
        # ❷ **找到所有轮廓**
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # **筛选最大轮廓（假设颜色卡的黑色轮廓是最大的）**
        if len(contours) == 0:
            continue

        max_contour = max(contours, key=cv2.contourArea)

        # ❸ **拟合四边形**
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)

        if len(approx) != 4:
            print(f"⚠️ {filename} 可能不是规则四边形，跳过处理")
            continue
        
        # ❹ **生成掩码（Mask），填充四边形外的区域为白色**
        mask = np.ones_like(image, dtype=np.uint8) * 255  # 全白背景
        cv2.fillPoly(mask, [approx], (0, 0, 0))  # 只在黑色外轮廓内保留原图

        # ❺ **应用掩码**
        result = cv2.bitwise_or(image, mask)  # 只显示颜色卡，外部变白

        # ❻ **保存最终处理的图片**
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result)

print("✅ 颜色卡处理完成，结果已保存到:", output_folder)
