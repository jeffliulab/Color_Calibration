import os
import shutil

# 定义源文件夹和目标文件夹
source_folder = "data_raw/obj_train_data"
image_folder = "dataset/images"
label_folder = "dataset/labels"

# 确保目标文件夹存在
os.makedirs(image_folder, exist_ok=True)
os.makedirs(label_folder, exist_ok=True)

# 处理文件重命名并移动
for file in os.listdir(source_folder):
    source_path = os.path.join(source_folder, file)

    if file.count(".") >= 2:  # 确保文件名中有两个点
        parts = file.split(".")
        new_name = parts[0] + "." + parts[-1]  # 只保留前后部分，去掉中间部分

        # 处理 .jpg 文件
        if file.endswith(".jpg"):
            target_path = os.path.join(image_folder, new_name)
            shutil.move(source_path, target_path)

        # 处理 .txt 文件
        elif file.endswith(".txt"):
            target_path = os.path.join(label_folder, new_name)
            shutil.move(source_path, target_path)

print("✅ 图片和标签已整理完毕！")
