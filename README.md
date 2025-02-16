# Color Calibration Card Detection Project

## Project Overview and Objectives
This project aims to develop an automated system for detecting and standardizing color calibration cards in images. The system is designed to accurately locate color calibration cards and apply image processing techniques to achieve standardized presentation, laying the foundation for subsequent color calibration work.

The objective of this project is to use a calibration card to detect color, in any kind of circumstances. The detecting model can detect all kind of colors, which means it has the capability of generalization.


## Project Framework

The whole pipeline:

![alt text](docs/readme/pipeline.png)

The card this project use:

<img src="docs/readme/card.png" width="400">


## Project Environment and Structrue

### Environment

### Structure

```
Color_Calibration
│
├── notebooks/                     # Explanation and Demos. If you want to know
│                                  # roughly details, you can directly see here.
│
├── .dvc/                          # DVC configuration
├── data/                          # Data management
│   ├── raw/                       # Raw Data, most are not pushed to repo
│   ├── processed/                 # Processed Data, all data are here
│   ├── train/                     # training dataset
│   ├── test/                      # test dataset
│   └── models/                    # stored models
│
├── configs/                       # Configurations
│   ├── detect/                    # YOLOv8 configs
│   ├── feature/                
│   └── train/              
│
├── src/                           # Core application code
│   ├── detect/                    # YOLOv8 - detect calibration card patterns
│   ├── feature/                   # extract features
│   ├── train/                     # train model
│   └── /
│
├── docs/                          # docs, logs, readme files, etc.
├── outputs/                       # running files tracking, git ignored
└── README.md
```

<!-- 
```
color_calibration/
├── .github/                        # CI/CD workflows
│   └── workflows/
│       ├── ci.yml                 # Continuous Integration
│       ├── cd.yml                 # Continuous Deployment
│       └── model-training.yml     # Model training pipeline
│


│
├── deployment/                    # Deployment configurations
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.training
│   │   └── docker-compose.yml
│   ├── kubernetes/
│   │   ├── api-deployment.yaml
│   │   ├── training-job.yaml
│   │   └── monitoring/
│   └── terraform/                # Infrastructure as Code
│       ├── main.tf
│       └── variables.tf
│
├── mlops/                        # MLOps specific code
│   ├── monitoring/
│   │   ├── metrics.py
│   │   └── alerts.py
│   ├── pipelines/
│   │   ├── training_pipeline.py
│   │   └── deployment_pipeline.py
│   └── serving/
│       ├── api.py
│       └── middleware.py
│
│
│
├── tests/                       # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── tools/                       # Development tools
│   ├── quality_checks/
│   └── model_analysis/
│
├── .env.example                 # Environment variables template
├── Makefile                     # Build automation
├── requirements/
│   ├── requirements.txt         # Base requirements
│   ├── requirements-dev.txt     # Development requirements
│   └── requirements-prod.txt    # Production requirements
│
└── scripts/
    ├── setup_monitoring.sh
    ├── deploy_model.sh
    └── run_tests.sh
```

├── notebooks/
│   ├── 01-data-exploration.ipynb      # 数据探索 & 可视化
│   ├── 02-data-cleaning.ipynb         # 数据清理
│   ├── 03-feature-engineering.ipynb   # 特征工程
│   ├── 04-model-training.ipynb        # 初步模型训练
│   ├── 05-hyperparameter-tuning.ipynb # 超参数优化
│   ├── 06-model-evaluation.ipynb      # 评估模型效果
│   ├── 07-inference.ipynb             # 预测 & 结果分析
│   ├── 08-visualization.ipynb         # 可视化最终结果
│   ├── experiments/                    # 额外实验
│   │   ├── experiment-baseline.ipynb   # 基线模型实验
│   │   ├── experiment-new-method.ipynb # 新方法测试
│   │   └── experiment-debug.ipynb      # 调试 Notebook
│   ├── reports/                         # 可选，存放最终报告
│   │   ├── final-report.ipynb
│   │   └── presentation.ipynb
│   ├── README.md                        # 目录说明
-->


## Phase 1: ETL Processing

### Find Patterns


**Step1. Annotation**

Implement YOLOv8 for color calibration card detection.

Firstly recognize the card:

<img src="docs/readme/annotation_1.png" width="400">

Secondly recognize the patterns:

<img src="docs/readme/annotation_2.png" width="400">



**Step2. Training YOLOv8 model**

Create a GCP VM to train the model:


<img src="docs/readme/gcp.png" width="400">


**Step3. Use YOLOv8 to detect patterns**

Detect the card:
<img src="docs/readme/detect_1_1.png" width="400">

Detect the patterns:
<img src="docs/readme/detect_2.png" width="400">

Extract four patterns:
<img src="docs/readme/detect_3.png" width="400">

**Step4. Data Augmentation**

Using existant photos to make ddata augmentation.

Using albumentations to implement:
```
augmentations = [
    ("brightness", A.RandomBrightnessContrast(p=1.0)),
    ("hue", A.HueSaturationValue(p=1.0)),
    ("gamma", A.RandomGamma(p=1.0)),
    ("motion_blur", A.MotionBlur(blur_limit=5, p=1.0)),
    ("gaussian_blur", A.GaussianBlur(blur_limit=5, p=1.0)),
    ("clahe", A.CLAHE(clip_limit=4.0, p=1.0)),
    ("noise", A.ISONoise(p=1.0)),
    ("rotate", A.Rotate(limit=10, p=1.0, border_mode=cv2.BORDER_REFLECT)),
]
```

After augmentation, the augmentation method is marked in the file name:

<img src="docs/readme/aug_data.png" width="400">

The original 250 photos are increased to 2000+ photos.

**Step5. Extract Patterns**

Extract patterns and finish preparation for feature extractions.

<img src="docs/readme/extract_patterns.png" width="400">

After processing completed: Total number of images 2295, number of generated files 9263, number of failed images 21.

So far, the ETL process is finished. The next step is ETV.

## Phase 2: ETV Processing

1. **Feature Extraction**
   - Extract Red, Green, Blue, and Contrast values (`Rp, Gp, Bp, Cp`).
   - Label each detected pattern with its corresponding reference color.
   - Store extracted features in structured datasets for further processing.

After running extract_feature.py, the features of 9000 photos are stored in a .csv file:

<img src="docs/readme/feature_extraction.png" width="400">

The feature extraction logic now is calculate the center region of the photos, and calculate the mean RGB value. The features will be stored so there will be more feature extraction logics.

2. **Data Storage and Versioning**
   - Store processed data in a version-controlled database using **DVC**.
   - Maintain different versions for traceability and reproducibility.




## Phase 3: MTL Processing

In phase 3, the system will focus on **Machine Learning Training & Learning**:

1. **Model Training & Validation**
   - Train machine learning models on the extracted feature dataset.
   - Evaluate different models (e.g., SVM, Random Forest, Neural Networks) for best accuracy.
   - Apply hyperparameter tuning to optimize model performance.

2. **MLflow Experiment Tracking**
   - Log model training runs, parameters, and metrics using **MLflow**.
   - Enable easy model comparison and reproducibility.
   - Store best-performing models in the **Model Registry**.

3. **Inference Artifact Generation**
   - Convert trained models into deployable inference artifacts.
   - Optimize models for real-time inference efficiency.
   - Prepare models for integration into production.

## Phase 4: MRD Processing

In phase 4, the system will be **deployed and monitored**:

1. **Model Deployment**
   - Deploy the trained model as a **FastAPI REST API** for real-time inference.
   - Containerize the deployment using **Docker & Kubernetes**.
   - Implement auto-scaling for handling different workloads.

2. **Model Registry & Version Control**
   - Store all trained models in the **Model Registry** with versioning.
   - Enable easy rollback in case of performance degradation.
   - Maintain metadata logs for auditability and traceability.

3. **Monitoring & Continuous Improvement**
   - Implement monitoring tools (**Prometheus, Grafana**) to track model performance.
   - Detect concept drift and trigger re-training when necessary.
   - Set up an alerting system for anomaly detection and failures.



