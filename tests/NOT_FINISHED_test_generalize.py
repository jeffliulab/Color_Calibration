from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import ast
import pandas as pd

class ColorPredictionSystem:
    def __init__(self, rf_model: RandomForestRegressor):
        """
        初始化颜色预测系统
        
        Args:
            rf_model: 已训练好的RandomForestRegressor模型实例
        """
        self.pattern_detector = PatternDetector()
        self.rf_model = rf_model
        
        # 定义标准参考色值
        self.reference_colors = {
            'Rs': (255, 0, 0),
            'Gs': (0, 255, 0),
            'Bs': (0, 0, 255)
        }
        
        # 定义形状到颜色参考的映射
        self.pattern_mapping = {
            'red_circle': 'Rp',    # 红色参考物
            'green_triangle': 'Gp', # 绿色参考物
            'blue_pentagon': 'Bp',  # 蓝色参考物
            'black_box': 'Cp'      # 目标物
        }

    def predict_true_color(self, image) -> Tuple[tuple, tuple]:
        """
        预测图像中目标物的真实颜色
        
        Args:
            image: 输入图像（numpy数组）或图像路径
            
        Returns:
            Tuple[tuple, tuple]: (拍摄到的目标色, 预测的真实色)
        """
        # 读取图像（如果输入是路径）
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Cannot read image at {image}")

        # 获取各个pattern的RGB值
        pattern_results = self.pattern_detector.getRGB(image)
        
        # 转换为字典
        pattern_dict = {class_name: rgb for rgb, class_name in pattern_results if rgb is not None}
        
        # 检查所需的patterns
        required_patterns = ['red_circle', 'green_triangle', 'blue_pentagon', 'black_box']
        if not all(pattern in pattern_dict for pattern in required_patterns):
            detected_patterns = list(pattern_dict.keys())
            raise ValueError(f"Missing patterns. Required: {required_patterns}, Detected: {detected_patterns}")

        # 提取RGB值
        Rp = pattern_dict['red_circle']     # 红色参考物
        Gp = pattern_dict['green_triangle']  # 绿色参考物
        Bp = pattern_dict['blue_pentagon']   # 蓝色参考物
        Cp = pattern_dict['black_box']       # 目标物

        delta_Rp_R = Rp[0] - 255
        delta_Rp_G = Rp[1] - 0
        delta_Rp_B = Rp[2] - 0

        delta_Gp_R = Gp[0] - 0
        delta_Gp_G = Gp[1] - 255
        delta_Gp_B = Gp[2] - 0

        delta_Bp_R = Gp[0] - 0
        delta_Bp_G = Gp[1] - 0
        delta_Bp_B = Gp[2] - 255



        # 4. 最终的特征矩阵 X
        #    - 包含：参考色差值(9列) + 目标物拍摄值(3列)
        # X = data[
        #     [
        #         "Delta_RR_red", "Delta_RG_red", "Delta_RB_red",
        #         "Delta_RR_green", "Delta_RG_green", "Delta_RB_green",
        #         "Delta_RR_blue", "Delta_RG_blue", "Delta_RB_blue",
        #         "Cp_R", "Cp_G", "Cp_B",
        #     ]
        # ]



        # 构造特征向量
        input_features = [
            delta_Rp_R, delta_Gp_R, delta_Bp_R,
            delta_Rp_G, delta_Gp_G, delta_Bp_G,
            delta_Rp_B, delta_Gp_B, delta_Bp_B,
            Cp[0], Cp[1], Cp[2]
        ]



        # 预测
        # input_features_arr = np.array(input_features).reshape(1, -1)
        # predicted_rgb = tuple(self.rf_model.predict(input_features_arr)[0].astype(int))

        # 定义特征名称 (确保顺序与训练时一致)
        feature_names = [
            "Delta_RR_red", "Delta_RG_red", "Delta_RB_red",
            "Delta_RR_green", "Delta_RG_green", "Delta_RB_green",
            "Delta_RR_blue", "Delta_RG_blue", "Delta_RB_blue",
            "Cp_R", "Cp_G", "Cp_B"
        ]

        # **转换为 DataFrame**
        input_features_df = pd.DataFrame([input_features], columns=feature_names)

        # **进行预测**
        predicted_rgb = tuple(self.rf_model.predict(input_features_df)[0].astype(int))


        # 可视化结果
        self._visualize_results(Cp, predicted_rgb, pattern_dict)

        return Cp, predicted_rgb

    def _visualize_results(self, captured_color: tuple, predicted_color: tuple, pattern_dict: dict = None):
        """
        可视化拍摄色和预测色的对比
        
        Args:
            captured_color: 拍摄到的颜色
            predicted_color: 预测的颜色
            pattern_dict: 所有检测到的pattern颜色（可选）
        """
        if pattern_dict:
            plt.figure(figsize=(15, 4))
            
            # 显示所有参考色和目标色
            for i, (pattern_name, color) in enumerate(pattern_dict.items()):
                plt.subplot(1, 7, i+1)
                color_display = np.full((100, 100, 3), color, dtype=np.uint8)
                plt.imshow(color_display)
                plt.title(f'{pattern_name}\nRGB{color}')
                plt.axis('off')

            # **新增代码: 显示参考颜色 (真实色)**
            plt.subplot(1, 7, 6)  # 修改：索引改为 6
            true_color = self.reference_colors.get('Real Color', (248, 212, 198))  
            color_display = np.full((100, 100, 3), true_color, dtype=np.uint8)
            plt.imshow(color_display)
            plt.title(f'True Color\nRGB{true_color}')
            plt.axis('off')
            
            # 显示预测的真实颜色
            plt.subplot(1, 7, 7)
            color_display = np.full((100, 100, 3), predicted_color, dtype=np.uint8)
            plt.imshow(color_display)
            plt.title(f'Predicted True Color\nRGB{predicted_color}')
            plt.axis('off')
            
        else:
            plt.figure(figsize=(10, 4))
            
            # 显示拍摄到的颜色
            plt.subplot(1, 2, 1)
            color_display = np.full((100, 100, 3), captured_color, dtype=np.uint8)
            plt.imshow(color_display)
            plt.title(f'Captured Color\nRGB{captured_color}')
            plt.axis('off')
            
            # 显示预测的真实颜色
            plt.subplot(1, 2, 2)
            color_display = np.full((100, 100, 3), predicted_color, dtype=np.uint8)
            plt.imshow(color_display)
            plt.title(f'Predicted True Color\nRGB{predicted_color}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# 使用示例：
try:
    # 初始化系统
    system = ColorPredictionSystem(rf_model=rf_model)
    
    # 预测颜色
    captured_color, predicted_color = system.predict_true_color('demo_image/test_image.png')
    
    print("\nResults:")
    print("拍摄到的目标色 (Cp):", captured_color)
    print("模型实际上的真实色：",(248, 212, 198) ) # 修改这里的时候记得把上面作图的部分的对应代码也改了
    print("模型预测的真实色 (Cs_pred):", predicted_color)
    
except ValueError as e:
    print(f"\n错误: {e}")
except Exception as e:
    print(f"\n发生未预期的错误: {e}")



from skimage.color import rgb2lab, deltaE_cie76

# INPUT TRUE RGB OF TEST COLOR CALIBRATION HERE
true_rgb = (248, 212, 198)


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

# 确保 `true_rgb` 变量已经在内存中定义
true_lab = rgb_to_lab(true_rgb)
captured_lab = rgb_to_lab(captured_color)
predicted_lab = rgb_to_lab(predicted_color)

# 计算 Delta E
delta_e_captured = deltaE_cie76(true_lab, captured_lab)
delta_e_predicted = deltaE_cie76(true_lab, predicted_lab)

print("\n颜色误差分析:")
print(f"拍摄色 (Cp) vs 真实色: ΔE = {delta_e_captured:.2f}")
print(f"预测色 (Cs_pred) vs 真实色: ΔE = {delta_e_predicted:.2f}")


