import pandas as pd

# 读取两个CSV文件
unique_colors_df = pd.read_csv('unique_colors.csv')
dataset_df = pd.read_csv('GridMean_0216.csv')

# 创建一个字典，将color_code映射到RGB元组
color_rgb_dict = unique_colors_df.set_index('color_code')[['real_r', 'real_g', 'real_b']].apply(tuple, axis=1).to_dict()

# 定义一个函数，使用color_code从字典中提取RGB值并转换为字符串
def get_real_rgb(color_code):
    if color_code in color_rgb_dict:
        r, g, b = color_rgb_dict[color_code]
        return f"({r}, {g}, {b})"
    return None  # 如果没有找到匹配的color_code，可以返回None或其它值

# 创建新的'real_rgb'列
dataset_df['real_rgb'] = dataset_df['color_code'].apply(get_real_rgb)

# 保存修改后的dataset为新的CSV文件
dataset_df.to_csv('updated_dataset.csv', index=False)

# 打印结果查看
print(dataset_df)
