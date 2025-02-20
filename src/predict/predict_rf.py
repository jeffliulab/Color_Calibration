from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ultralytics import YOLO
import joblib

from src.detect.yolo import PatternDetector

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

class getRGB:
    def __init__(self):
        self.pattern_detector = PatternDetector()

    def getRGB(self, image) -> List[Tuple[Optional[Tuple[int, int, int]], Optional[str]]]:
        """
        Extract RGB values from the center of each pattern using 3x3 grid method.
        
        Args:
            image: Input image (numpy array in BGR format)
            
        Returns:
            list: List of tuples (RGB_value, class_name) or (None, None) for missing patterns
        """
        patterns = self.pattern_detector.getPattern(image, show_results=True)
        results = []

        for i, (pattern, class_name) in enumerate(patterns, start=1):
            if pattern is None:
                results.append((None, None))
                continue

            # Convert to RGB
            pattern_rgb = cv2.cvtColor(pattern, cv2.COLOR_BGR2RGB)
            
            # Get center region
            h, w, _ = pattern_rgb.shape
            row_start = h // 3
            row_end = 2 * (h // 3)
            col_start = w // 3
            col_end = 2 * (w // 3)
            
            center_region = pattern_rgb[row_start:row_end, col_start:col_end]
            
            # Calculate average color
            avg_color = tuple(np.mean(center_region, axis=(0, 1)).astype(int))
            results.append((avg_color, class_name))

            # Visualize the center region and its average color
            plt.figure(figsize=(10, 3))
            
            # Original pattern with center region highlighted
            plt.subplot(1, 3, 1)
            plt.imshow(pattern_rgb)
            plt.gca().add_patch(plt.Rectangle((col_start, row_start), 
                                            col_end-col_start, 
                                            row_end-row_start, 
                                            fill=False, 
                                            color='red', 
                                            linewidth=2))
            plt.title(f'{class_name}\nwith Center Region')
            plt.axis('off')
            
            # Center region
            plt.subplot(1, 3, 2)
            plt.imshow(center_region)
            plt.title('Center Region')
            plt.axis('off')
            
            # Average color
            plt.subplot(1, 3, 3)
            color_display = np.full((100, 100, 3), avg_color, dtype=np.uint8)
            plt.imshow(color_display)
            plt.title(f'Average Color\nRGB{avg_color}')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()

        return results



class ColorPredictionSystem:
    def __init__(self, model_dir_str): # 预留：self.true_color parameter
        """
        初始化颜色预测系统
        
        Args:
            model: 已训练好的RandomForestRegressor模型实例
        """
        self.get_rgb = getRGB()
        self.model = model_dir_str
        self.true_color = (248, 212, 198)
        
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
        pattern_results = self.get_rgb.getRGB(image)
        
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
        # predicted_rgb = tuple(self.model.predict(input_features_arr)[0].astype(int))

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
        predicted_rgb = tuple(self.model.predict(input_features_df)[0].astype(int))


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
            true_color = self.reference_colors.get('Real Color', self.true_color)  
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


if __name__ == "__main__":
    try:
        # 初始化系统
        model_path = ROOT_DIR / "data/models/random_forest/model_v1.pkl"
        model = joblib.load(model_path)  # 这里定义 rf_model

        system = ColorPredictionSystem(model)
        
        # 预测颜色
        test_image_dir = ROOT_DIR / "data/images/generalization/test.png"
        test_image_dir_str = str(test_image_dir)
        captured_color, predicted_color = system.predict_true_color(test_image_dir_str)
        
        print("\nResults:")
        print("拍摄到的目标色 (Cp):", captured_color)
        print("模型实际上的真实色：",(248, 212, 198) ) # 修改这里的时候记得把上面作图的部分的对应代码也改了
        print("模型预测的真实色 (Cs_pred):", predicted_color)
        
    except ValueError as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n发生未预期的错误: {e}")

