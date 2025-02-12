from ultralytics import YOLO
import torch
import time

# NOTE: 需要修改settings.json来正确使用YOLO
# terminal >> open "/Users/macbookpro/Library/Application Support/Ultralytics/settings.json"

def train():
    # 固定参数
    CONFIG = {
        'data': 'dataset.yaml',        # 数据配置文件
        'epochs': 100,              # 训练轮数
        'imgsz': 640,              # 图像大小
        'batch': 16,               # 批次大小
        'device': 0 if torch.cuda.is_available() else 'cpu',  # 使用GPU如果可用
        'project': 'runs/train',    # 保存目录
        'name': 'exp',             # 实验名称
    }
    
    print("开始训练...")
    print(f"使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 加载模型
        model = YOLO('yolov8n.pt')
        
        # 训练模型
        results = model.train(**CONFIG)
        
        # 计算训练时间
        train_time = time.time() - start_time
        hours = int(train_time // 3600)
        minutes = int((train_time % 3600) // 60)
        
        # 打印训练结果摘要
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
        
        # 导出模型
        print("\n导出模型中...")
        model.export()
        print(f"模型已导出到: {CONFIG['project']}/{CONFIG['name']}/weights/best.pt")
        
    except Exception as e:
        print(f"\n训练过程中出现错误: {str(e)}")
        raise

if __name__ == '__main__':
    train()