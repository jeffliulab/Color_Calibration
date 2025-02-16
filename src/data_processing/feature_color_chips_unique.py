from pathlib import Path
import pandas as pd

# Define ROOT_DIR
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = ROOT_DIR / "data/features/"
OUTPUT_DIR = ROOT_DIR / "outputs/features/unique_color_strip/"

# 文件路径
input_file = FEATURES_DIR / "GridMean_0216.csv"
output_file = OUTPUT_DIR / "unique_colors.csv"

# 读取 CSV 文件的第一列并去重
df = pd.read_csv(input_file, usecols=[0], names=["color_code"])
df_unique = df.drop_duplicates()

# 处理 search_code 列
def generate_search_code(color_code):
    if "-" in color_code:
        return color_code.split("_")[0] if "_" in color_code else color_code
    elif color_code.startswith("P"):
        return "none"
    return color_code

df_unique["search_code"] = df_unique["color_code"].apply(generate_search_code)

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 保存去重后的数据
df_unique.to_csv(output_file, index=False)

print(f"去重后的数据已保存到 {output_file}")