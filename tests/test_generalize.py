from pathlib import Path
import joblib
import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_cie76

# 导入封装的 ColorPredictionSystem
from src.predict.predict_rf import ColorPredictionSystem 

# 设置 ROOT 目录
ROOT_DIR = Path(__file__).resolve().parent.parent

# 加载模型
model_path = ROOT_DIR / "data/models/random_forest/model_v1.pkl"
model = joblib.load(model_path)

# 真实测试颜色
true_rgb = (248, 212, 198)

# 初始化预测系统
predictor = ColorPredictionSystem(model)

# 预测颜色
test_image_dir = ROOT_DIR / "data/images/generalization/test.png"
test_image_dir_str = str(test_image_dir)
captured_color, predicted_color = predictor.predict_true_color(test_image_dir_str)

# 打印预测结果
print("\nResults:")
print("拍摄到的目标色 (Cp):", captured_color)
print("模型实际上的真实色：", true_rgb)  # 确保 true_rgb 定义
print("模型预测的真实色 (Cs_pred):", predicted_color)

# 颜色误差计算函数
def rgb_to_lab(rgb: tuple) -> np.ndarray:
    """
    将 RGB 颜色转换为 Lab 颜色空间。
    
    Args:
        rgb: 颜色的 (R, G, B) 元组，范围 [0, 255]
    
    Returns:
        np.ndarray: Lab 颜色空间的值
    """
    rgb = np.array(rgb, dtype=np.float32) / 255.0  # 归一化到 [0,1]
    return rgb2lab(rgb.reshape(1, 1, 3)).reshape(3)

# 计算 Lab 颜色空间值
true_lab = rgb_to_lab(true_rgb)
captured_lab = rgb_to_lab(captured_color)
predicted_lab = rgb_to_lab(predicted_color)

# 计算 Delta E 颜色误差
delta_e_captured = deltaE_cie76(true_lab, captured_lab)
delta_e_predicted = deltaE_cie76(true_lab, predicted_lab)

# 打印颜色误差分析
print("\n颜色误差分析:")
print(f"拍摄色 (Cp) vs 真实色: ΔE = {delta_e_captured:.2f}")
print(f"预测色 (Cs_pred) vs 真实色: ΔE = {delta_e_predicted:.2f}")
