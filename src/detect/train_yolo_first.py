from ultralytics import YOLO
import torch
import time

# MANUALLY TEST GUIDE: yolo predict model=best.pt source=dataset/test/images save=True

"""
MODIFY Ultralytics (YOLOv8) JSON SETTINGS HERE
"""
import json
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
print()
print(f"ROOT_DIR: {ROOT_DIR}")
print()

# Define the path to Ultralytics settings.json
settings_path = Path.home() / ".config/Ultralytics/settings.json"

# Get the current date in YYYYMMDD format
current_date = datetime.now().strftime("%Y%m%d")  # e.g., '20250215'

def get_unique_date_path(base_path: Path, base_date: str):
    """
    Generate a unique date-based directory name by appending a counter if the directory already exists.

    Args:
        base_path (Path): The base directory where the date folder should be created.
        base_date (str): The base date string (YYYYMMDD).

    Returns:
        str: A unique date string (e.g., '20250215', '20250215_1', '20250215_2').
    """
    counter = 1
    new_date = base_date
    while (base_path / new_date).exists():
        new_date = f"{base_date}_{counter}"
        counter += 1
    return new_date

# Compute a unique date-based directory name
unique_date = get_unique_date_path(ROOT_DIR / "outputs/runs", current_date)

def update_ultralytics_settings():
    """
    Update the Ultralytics settings.json file to ensure the correct paths for dataset, weights, and runs.

    - `datasets_dir`: Directory for dataset storage.
    - `weights_dir`: Directory for trained model weights (organized by date).
    - `runs_dir`: Directory for experiment logs (organized by date).
    """
    # Load existing settings.json if available, otherwise initialize an empty dictionary
    if settings_path.exists():
        with open(settings_path, "r") as f:
            settings = json.load(f)
    else:
        settings = {}

    # Update the settings with the new paths
    settings.update({
        "datasets_dir": str(ROOT_DIR / "data/train/detect/yolo_first_0214/"),  # Dataset storage path
        "weights_dir": str(ROOT_DIR / f"weights/detect/yolo_first/{unique_date}/"),  # Model weights storage path
        "runs_dir": str(ROOT_DIR / f"outputs/runs/detect/yolo_first/{unique_date}/")  # Experiment logs storage path
    })

    # Save the updated settings.json
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)

    # Print confirmation messages
    print(f"✅ Updated Ultralytics settings.json: {settings_path}")
    print(f"📂 Training data will be stored in: {settings['runs_dir']}")
    print(f"📂 Model weights will be stored in: {settings['weights_dir']}")

# **Run the update function**
update_ultralytics_settings()

"""
MODIFY TRAINING CONFIGS HERE
"""
def train():
    # 固定参数
    CONFIG = {
        'data': str(ROOT_DIR / "configs/detect/train_yolo_first.yaml"),        # 数据配置文件
        'epochs': 100,              # 训练轮数
        'imgsz': 640,              # 图像大小
        'batch': 16,               # 批次大小
        'device': 0 if torch.cuda.is_available() else 'cpu',  # 使用GPU如果可用
        'project': str(ROOT_DIR / f"outputs/runs/detect/yolo_first/{unique_date}/"),    # auto save in YOLO
        'name': 'exp',             # 实验名称
    }
    
    print("Start Training...")
    print(f"Using GPU/CPU: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # record start time
    start_time = time.time()
    
    try:
        # LOAD MODEL
        model_path = str(ROOT_DIR / "weights/YOLOv8/yolov8n.pt")
        model = YOLO(model_path)
        
        results = model.train(**CONFIG)
        
        train_time = time.time() - start_time
        hours = int(train_time // 3600)
        minutes = int((train_time % 3600) // 60)
        
        print("\n" + "="*50)
        print("训练完成！结果摘要：")
        print("="*50)
        print(f"训练时长: {hours}小时 {minutes}分钟")
        print(f"最佳mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"最佳mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
        print(f"最终目标检测损失: {results.results_dict['train/box_loss'][-1]:.4f}")
        print(f"最终分类损失: {results.results_dict['train/cls_loss'][-1]:.4f}")
        print(f"模型保存路径: {CONFIG['project']}/{CONFIG['name']}")
        print("="*50)
        
        # export model
        print("\export...")
        model.export()
        print(f"export to: {CONFIG['project']}/{CONFIG['name']}/weights/best.pt")
        
    except Exception as e:
        print(f"\Error: {str(e)}")
        raise

if __name__ == '__main__':
    train()