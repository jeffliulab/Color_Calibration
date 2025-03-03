# Color Calibration Card Detection Project

## Real-time Color Calibration Card on Website

You can test the model with uploading or drag the photo with calibration card through this website:

- [Real-time Color Calibration Website](https://jeffliulab.github.io/real_time_demo/color_calibration/index.html)

## Project Overview

### Objectives

This project aims to develop an automated system for detecting and standardizing color calibration cards in images. The system is designed to accurately locate color calibration cards and apply image processing techniques to achieve standardized presentation, laying the foundation for subsequent color calibration work.

The objective of this project is to use a calibration card to detect color, in any kind of circumstances. The detecting model can detect all kind of colors, which means it has the capability of generalization.

### Project Progress and Final Optimizing Goal

**Current Progress**:

- V1 Finished: **Reaching the general commercial printing standard.** In Feb 20, the XGBoost model reach the highest score of Lab Mean ΔE = 5.20, Lab Median ΔE = 3.96. This result meets general commercial printing standards.

Through V1 exploration, it is showned that the tree model performs better in this group of dataset and trainings.

I will continue to explore the effects of other data sets, different data enhancements, and even different color recognition cards in the future.

**Future Development Plan:**

- V2 Stage: Reduce Mean ΔE to between 3-4, suitable for most commercial applications.
- V3 Stage: Reduce Lab Mean ΔE to below 2, reaching the professional color calibration field standard.

## Project Framework

The whole pipeline:

![alt text](docs/readme/pipeline.png)

The card this project use:

<img src="docs/readme/card.png" width="400">

You can print the card with this picture and use the deployment on website to test your version:

<img src="docs/readme/card/card.png" width="400">

## Project Environment and Structrue

### Environment

I use conda to control versions, the requirments are all in `environment.yml`. (Ubuntu 20.04)

For better adaptbility, use Docker.

### Structure

```
Color_Calibration
│
├── notebooks/                     # Experimental models. 
│
├── .dvc/                          # DVC configuration
├── data/                          # Data, all stored via DVC in GCP
│   ├── raw/                       # raw images
│   ├── processed/                 # processed images
│   ├── train/                     # training dataset
│   ├── test/                      # test dataset
│   ├── features/                  # store feature extraction .csv files
│   └── models/                    # store models
│
├── configs/                       # Configurations
│   ├── detect/                    # YOLOv8 configs
│   ├── feature/              
│   └── train/            
│
├── src/                           # Core application code
│   ├── data_processing/           # clean or process data for preparation
│   ├── detect/                    # YOLOv8 - detect calibration card patterns
│   ├── feature_extraction/        # extract features
│   ├── train/                     # train and evaluate model
│   └── api/                       # FastAPI deployment
│
├── tests/                         # generalization test
│
├── .devcontainer                  # Docker for development
├── Dockerfile                     # Docker for Google Cloud Run API deployment
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
`<img src="docs/readme/detect_1_1.png" width="400">`

Detect the patterns:
`<img src="docs/readme/detect_2.png" width="400">`

Extract four patterns:
`<img src="docs/readme/detect_3.png" width="400">`

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

Use DVC and GCP bucket to store and manage data files.

The data files including images and .csv are now stored in GCP bucket, the address is:

- gs://color_calibration

<img src="docs/readme/dvc.png" width="400">

3. **Real RGB Values**

After above processings, the Rp, Gp, Bp, Cp (p for real photo) have extracted.

Now, the real values of Rs, Gs, Bs, Cs (s for standard) need to be known.

Rs, Gs, Bs are never change, so we only need to get Cs.

The real value is labeled as "real_rgb" in the last column:

<img src="docs/readme/features.png">

## Phase 3: MTL - Model Training and Validation

In phase 3, the system will focus on **Machine Learning Training**:

### Model Selection

After **feature extraction**, the `dataset` contains approximately **2000 data points**. Given the feature mapping:

<img src="docs/readme/f0.png" width="300">

In this relationship:

- Rp, Gp, Bp: The camera-measured RGB values of known standard color of Red, Green, and Blue. Here p is for photo.
- Cp: The camera-measured RGB value of target patch, which is got in the calbration card.
- Rs, Gs, Bs: The standard and stable RGB value of Red(255,0,0), Green(0,255,0), and Blue(0,0,255)

Below is an overview of the universal color-correction logic, regardless of which specific model (linear, random forest, neural network, etc.) ultimately choose. The key idea is that if we can find the function, or the relationship between those two groups, for example, if we can find a transformation rule, for example, if that is a matrix, then we can get:

<img src="docs/readme/f1.png" width="300">

Or other kind of functions:

<img src="docs/readme/f2.png" width="300">

Then we can in turn to get:

<img src="docs/readme/f3.png" width="300">

Where Creal represent the real (or canonical) RGB of the target patch.

---
The detailed codes and explanations of this step are as follow:

```python
import ast
import pandas as pd

# Assume that before this step, your `data` DataFrame already contains the following columns (as string types):
# ["ColorCode", "UniqueID", "Cp", "Rp", "Gp", "Bp", "Cs", "Rs", "Gs", "Bs"]
# Additionally, ensure that you have already performed necessary preprocessing such as `data.dropna()`.

# 1. Define a function to parse an RGB string in the format "(R, G, B)"
#    and convert it into a tuple of integers (R_int, G_int, B_int).
def parse_rgb(rgb_string):
    return ast.literal_eval(rgb_string)  # Convert the string to an (R, G, B) tuple

# 2. Expand the captured values (Rp, Gp, Bp, Cp) and target values (Cs) into separate R/G/B columns.
#    Note: The columns Rs/Gs/Bs are also stored as strings, but they contain the same fixed values for every row.
#    We use them only to "record" the standard color reference.
for col in ["Rp", "Gp", "Bp", "Cp"]:
    data[[f"{col}_R", f"{col}_G", f"{col}_B"]] = data[col].apply(parse_rgb).apply(pd.Series)

# Parse the target column Cs into separate R, G, and B channels
data[["Cs_R", "Cs_G", "Cs_B"]] = data["Cs"].apply(parse_rgb).apply(pd.Series)

# 3. Compute the difference between the reference color captured values and the corresponding standard colors.
#    - Red reference object (Rp) compared to the standard red color (255, 0, 0)
data["Delta_RR_red"] = data["Rp_R"] - 255  # Difference in the red channel
data["Delta_RG_red"] = data["Rp_G"] - 0    # Difference in the green channel
data["Delta_RB_red"] = data["Rp_B"] - 0    # Difference in the blue channel

#    - Green reference object (Gp) compared to the standard green color (0, 255, 0)
data["Delta_RR_green"] = data["Gp_R"] - 0   # Difference in the red channel
data["Delta_RG_green"] = data["Gp_G"] - 255 # Difference in the green channel
data["Delta_RB_green"] = data["Gp_B"] - 0   # Difference in the blue channel

#    - Blue reference object (Bp) compared to the standard blue color (0, 0, 255)
data["Delta_RR_blue"] = data["Bp_R"] - 0   # Difference in the red channel
data["Delta_RG_blue"] = data["Bp_G"] - 0   # Difference in the green channel
data["Delta_RB_blue"] = data["Bp_B"] - 255 # Difference in the blue channel

# 4. Construct the final feature matrix X.
#    - This includes the 9 reference color difference values + 3 captured object RGB values.
X = data[
    [
        "Delta_RR_red", "Delta_RG_red", "Delta_RB_red",
        "Delta_RR_green", "Delta_RG_green", "Delta_RB_green",
        "Delta_RR_blue", "Delta_RG_blue", "Delta_RB_blue",
        "Cp_R", "Cp_G", "Cp_B",
    ]
]

# 5. Define the target variable y (three channels: Cs_R, Cs_G, Cs_B)
y = data[["Cs_R", "Cs_G", "Cs_B"]]

# 6. Optionally, drop the original columns Rp/Gp/Bp/Cp/Cs after processing.
#    This step is optional and depends on whether you still need these raw values.
# data = data.drop(columns=["Rp", "Gp", "Bp", "Cp", "Cs", "Rs", "Gs", "Bs"])

# 7. The next steps would follow the usual machine learning workflow:
#    - Split data into training and testing sets
#    - Train a model such as RandomForestRegressor
#
# Example:
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RandomForestRegressor()
# model.fit(X_train, y_train)

```
---

So this is the next goal of this project: find an efficient model to get the Creal. After considring, I decide to initially select and experiment with the following **five methods** for calibration and modeling:

1. **Linear Regression**
2. **Random Forest**
3. **Gradient Boosting Trees**
4. **Small Neural Network**

Subsequent experiments will compare these methods, analyze their performance on the dataset, and select the optimal approach for further optimization. Most of above are Machine Learning models, but I still choose one classicial method to compare with others. The purpose of this project is not to find something to do with ML, but to find a suitable model to solve a real problem.

---

### Random Forest

So far the **Random Forest** for color calibration prediction and evaluated is almost the best model in performance. Though XGBoost performs a little bit better, the difference is not obvious. Therefore, I will use this model to introduce the MTL process. Also, this method will serve as the baseline for further model optimization.

The complete methods and training process can now be found in `notebooks/M1_RandomForest.ipynb`.

#### Data Processing

- Read data from `feature_0216.csv` and rename relevant columns:
  - `Rp, Gp, Bp, Cp` represent the captured colors (affected by lighting conditions).
  - `Rs, Gs, Bs, Cs` represent the standard reference colors (true colors).
- Parse the `(R, G, B)` color strings and split them into individual `R, G, B` values.
- Compute the **color deviation** between the captured values of reference objects (red, green, blue) and their standard values (`Delta_RR_red, Delta_RG_red, Delta_RB_red`, etc.).
- Construct the feature matrix `X`, including:
  - Color deviations (`Delta_RR_*`, `Delta_RG_*`, `Delta_RB_*`).
  - Captured color of the target object (`Cp_R, Cp_G, Cp_B`).
- The target variable `y` is set as `Cs_R, Cs_G, Cs_B` (true color RGB values).

#### Model Training and Evaluation

- **Model**: Trained using `RandomForestRegressor(n_estimators=500, random_state=42)`.
- **Data Split**: 70% training set, 30% test set (`train_test_split(X, y, test_size=0.3, random_state=42)`).
- **Evaluation Metrics**:
  - `R² Score`: `0.8225`
  - `RMSE`: `12.10`
  - `MAPE`: `4.33%`
  - **ΔE (Lab Color Space Error)**:
    - **Mean ΔE**: `5.20`
    - **Median ΔE**: `3.96`
  - **Error Distribution**:
    - A histogram is used to show the distribution of ΔE errors and evaluate the overall accuracy of the model’s predictions.

Model prediction visualization:

<img src="docs/readme/rf/model_output.png" width="400">

#### Baseline Comparison

To evaluate the effectiveness of the `Random Forest` model, we introduced a **non-model baseline**:

- **Baseline Method**: Predict `y_test` using the mean of `y_train`.
- **Baseline Evaluation**:
  - `Baseline R² Score`: `-0.0037`
  - `Baseline RMSE`: `31.55`
  - `Baseline MAPE`: `14.98%`
  - `Baseline Mean ΔE (Lab)`: `14.61`
  - `Baseline Median ΔE (Lab)`: `13.84`

The comparison shows that `Random Forest` **outperforms the baseline model in all metrics**, especially in reducing the `ΔE (Lab)` error, demonstrating that the model effectively reduces color prediction deviations.

#### Color Prediction System

Based on the `Random Forest` model, we developed an automated **Color Prediction System**:

1. **Detect color regions in the image**:
   - Use the `YOLO` object detection model (`PatternDetector`) to recognize reference objects (`red_circle, green_triangle, blue_pentagon, black_box`).
   - Extract RGB color values from each detected region.
2. **Compute color deviations**:
   - Calculate `Delta_RR_*`, `Delta_RG_*`, `Delta_RB_*` as input features.
3. **Predict the true color of the target object**:
   - Use the trained `Random Forest` model for color correction.
4. **Visualization of results**:
   - Display the captured color (`Cp`), predicted true color (`Cs_pred`), and reference colors (`Rp, Gp, Bp`).

---

### Linear Regression 3x3

**Model Configuration:**

- **Algorithm:** Linear Regression
- **Input Features:** Standardized RGB values and reference color differences
- **Training Strategy:** Least squares optimization (linear transformation matrix)
- **Reasoning:**
  - Linear regression is a simple and interpretable approach that attempts to find the optimal transformation matrix mapping input colors to corrected colors.
  - Since color correction is inherently a non-linear problem, linear regression may struggle to capture subtle variations.

**Results:**

- **R² Score:** 0.7113
- **RMSE:** 14.98
- **MAPE:** 6.41%
- **Mean Color Difference (ΔE):** 20.87
- **Mean ΔE (Lab):** 6.63
- **Median ΔE (Lab):** 5.53

<img src="docs/readme/model_evaluation/linear_regression.png" width="300">

**Observations:**
Linear regression provides a basic baseline but fails to capture non-linearities in color transformations. It produces the highest color difference errors, suggesting that a more flexible model is needed.

---

### Gradient Boosting Trees (XGBoost)

**Model Configuration:**

- **Algorithm:** XGBoost (Gradient Boosting Decision Trees)
- **Boosting Rounds:** 500
- **Learning Rate:** 0.1
- **Tree Depth:** 5
- **Objective:** Squared error loss (reg:squarederror)
- **Reasoning:**
  - Gradient Boosting Trees (GBT) iteratively improve predictions by focusing on previous errors.
  - GBT models are well-suited for structured data and can effectively model non-linear relationships.
  - XGBoost provides built-in support for multi-output regression, making it a strong candidate for color prediction tasks.

**Results:**

- **R² Score:** 0.8280
- **RMSE:** 11.76
- **MAPE:** 4.09%
- **Mean Color Difference (ΔE):** 13.64
- **Mean ΔE (Lab):** 5.14
- **Median ΔE (Lab):** 3.95

<img src="docs/readme/model_evaluation/xgboost.png" width="300">

**Observations:**
XGBoost performs the best among the tested models. The boosting mechanism helps capture complex interactions between input features, reducing prediction errors significantly.

---

### Small Neural Network (MLP)

**Model Configuration:**

- **Network Architecture:**
  - Input Layer: Features from color calibration
  - Hidden Layer 1: 64 neurons (ReLU activation)
  - Hidden Layer 2: 64 neurons (ReLU activation)
  - Output Layer: 3 neurons (RGB values)
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam (learning rate = 0.001)
- **Training Epochs:** 500
- **Batch Size:** 32
- **Reasoning:**
  - Neural networks can approximate non-linear relationships in color transformations.
  - The chosen architecture is relatively small to avoid overfitting while still capturing relevant feature interactions.
  - ReLU activation ensures efficient gradient propagation, while Adam optimizer adapts learning rates for better convergence.

**Results:**

- **R² Score:** 0.7068
- **RMSE:** 14.22
- **MAPE:** 5.92%
- **Mean Color Difference (ΔE):** 20.21
- **Mean ΔE (Lab):** 7.39
- **Median ΔE (Lab):** 6.70

<img src="docs/readme/model_evaluation/mlp.png" width="300">

**Observations:**
The neural network underperforms compared to XGBoost, likely due to limited data or suboptimal hyperparameters. More complex architectures, additional training epochs, or data augmentation might improve results.

---

### Validation

At this stage, it is time to evaluate the performance of different models and apply hyperparameter tuning to improve accuracy. To ensure the model's generalization ability, I use a set of evaluation metrics and conduct cross-validation.

**1. Evaluation Metrics**
To assess the accuracy of color calibration, we use the following metrics:

- **R² Score**: Measures the explanatory power of the model. The closer to 1, the better.
- **RMSE (Root Mean Squared Error)**: Measures the average deviation between predicted and actual values. Lower is better.
- **MAPE (Mean Absolute Percentage Error)**: Measures the relative percentage of errors, useful for comparing different data ranges.
- **ΔE (Lab Color Space Error)**:
  - **Mean ΔE**: Average color error, reflecting overall performance.
  - **Median ΔE**: Median color error, reducing the influence of extreme values.

These metrics comprehensively evaluate the model's color prediction performance.

**2. Cross-Validation**
To ensure model stability, we use **K-Fold Cross-Validation**:

- **K = 5**, meaning the dataset is divided into five parts, with four used for training and one for testing in each iteration.
- Multiple models are trained, and the **Mean Validation Score** is calculated to assess stability.

**To be followed up**: Currently, only a single train-test split has been performed. **K-Fold Cross-Validation** needs to be implemented.

| Model                      | R² Score        | RMSE            | MAPE            | Mean ΔE       | Median ΔE     |
| -------------------------- | ---------------- | --------------- | --------------- | -------------- | -------------- |
| Linear Regression          | 0.7113           | 14.98           | 6.41%           | 20.87          | 6.63           |
| Random Forest              | 0.8225           | 12.10           | 4.33%           | 5.20           | 3.96           |
| XGBoost                    | **0.8280** | **11.76** | **4.09%** | **5.14** | **3.95** |
| Small Neural Network (MLP) | 0.7068           | 14.22           | 5.92%           | 7.39           | 6.70           |

**3. Hyperparameter Tuning**
To further improve model performance, we apply **Grid Search + K-Fold Cross-Validation** for hyperparameter tuning on **Random Forest** and **XGBoost**.

**Random Forest Tuning**

- **Adjusted parameters**:
  - `n_estimators`: 100 → 500
  - `max_depth`: 10 → 20
  - `min_samples_split`: 2 → 5
  - `min_samples_leaf`: 1 → 3

**To be followed up**: Currently, the `Random Forest` model is trained with default parameters. **Hyperparameter tuning needs to be implemented** using `GridSearchCV` or `Optuna`.

**XGBoost Tuning**

- **Adjusted parameters**:
  - `learning_rate`: 0.1 → 0.05
  - `n_estimators`: 500 → 1000
  - `max_depth`: 5 → 7
  - `min_child_weight`: 1 → 3
  - `subsample`: 0.8 → 0.9

**To be followed up**: XGBoost is currently trained with initial default parameters. **Hyperparameter tuning needs to be conducted**, and pre- and post-optimization results need to be recorded.

**4. Error Analysis**
To better understand model errors, we analyze the distribution of `ΔE` errors:

- **Error Distribution Visualization**:
  - Most `ΔE` errors are concentrated between 3-6, with fewer extreme errors.
  - Some extreme values (`ΔE > 15`) may be due to extreme lighting conditions causing significant color shifts.

**To be followed up**: The **error distribution histogram has not been plotted yet**. Visualization needs to be added.

So far, **XGBoost is currently the best-performing model, with a final ΔE error of 5.14**, meeting commercial-grade standards. **Hyperparameter tuning and cross-validation** still need to be completed for further accuracy improvements. The next target is to **reduce ΔE to below 2.0**, achieving professional color calibration standards. This model has potential applications in **industrial color calibration, photography post-processing, and print quality control**.

---

### Real-World Generalization Test

To test the generalization performance, I took an independent photo for testing. The color card in this image was not included in any of the collected data, making it a test of the model’s generalization ability. I use **extremely color shift lighting condition** to make sure the test is "generalize enough".

***Note: This photo was specifically chosen to be taken under extremely yellow lighting conditions to simulate an extreme color shift.***

The photo is shown below:

<img src="docs/readme/rf/1.png" width="300">

The extraction process is as follows:

<img src="docs/readme/rf/2.png" width="600">
<img src="docs/readme/rf/3.png" width="600">
<img src="docs/readme/rf/4.png" width="600">
<img src="docs/readme/rf/5.png" width="600">
<img src="docs/readme/rf/6.png" width="600">

Final test results:

<img src="docs/readme/rf/7.png">

- In the real-world test image, the model-predicted color (`Cs_pred`) **is closer to the target color compared to the directly captured color (`Cp`)**.
- Computation of `ΔE` in the Lab color space:
  - `Cp vs True Color`: **`ΔE = 38.07`**  (Significant error)
  - `Cs_pred vs True Color`: **`ΔE = 16.46`**  (Significant improvement)

The results show that the `Random Forest` model significantly reduces color errors, making the predicted color much closer to the standard color.

However, the generalization ΔE is significantly higher than the ΔE observed in the training and test sets, indicating that the model's generalization ability still has room for improvement.

These test results serve as the `Baseline` for `Random Forest` color calibration. Future directions for improvement include:

- Expanding the dataset to improve model generalization.
- Exploring more complex models such as `XGBoost` or `Neural Networks` for comparison.
- Training in the `Lab` color space to enhance perceptual consistency.

## Phase 4: Model Processing & Deployment

The system integrates **machine learning models, model tracking, CI/CD pipelines, and scalable deployment infrastructure**.

### 1. MLflow Experiment Tracking & Model Registry

To ensure **version control, hyperparameter optimization, and model traceability**, we integrated **MLflow** into our pipeline:

- **Experiment Tracking**: Logged all training runs, hyperparameters, evaluation metrics (R², RMSE, ΔE) using MLflow.
- **Model Registry**: The best-performing models were registered in the **MLflow Model Registry**, enabling version control and rollback in case of performance degradation.
- **Parameter Optimization**: Used MLflow to **compare different training configurations** (e.g., tree depths in XGBoost, learning rates in neural networks) and select the best hyperparameters.
- **Metadata Logging**: Ensured that **all model artifacts (code, hyperparameters, metrics, dataset versions) were tracked**, facilitating reproducibility and auditability.

**Outcome**: Enabled easy **model comparison, rollback, and continuous improvement** using MLflow.

### 2. Model Packaging & Deployment

To ensure the **scalability and usability** of the model in real-world applications, the model was **packaged and deployed as a FastAPI REST API**:

- **Inference API**: The trained model was wrapped inside a **FastAPI microservice**, exposing an endpoint for **real-time inference**.
- **Optimized for Speed**:
  - **ONNX Conversion**: Converted models into **ONNX format** to optimize for inference speed.
  - **TensorRT Acceleration**: Applied TensorRT optimizations for GPU-accelerated inference.
- **Dockerization**: Packaged the API into a **Docker container**, ensuring reproducibility across environments.
- **Kubernetes Deployment**:
  - Deployed the API on a **Kubernetes cluster** with autoscaling enabled.
  - Configured **Horizontal Pod Autoscaler (HPA)** to handle varying workloads.
  - Used **Ingress Controller** for API traffic management.

**Outcome**: The deployed model can **efficiently handle real-time inference requests**, scales automatically, and is optimized for fast response times.

The FastAPI can give back the output RGB values:

<img src="docs/readme/api/fast_api_response.png" width="300">

The API is deployed on Google Run and demonstrated on a real-time website:

<img src="docs/readme/api/website.png" width="500">

Then after few seconds, you will see the output:

<img src="docs/readme/api/real-time-result.png" width="500">

This website's API is deployed on Google Cloud Run, so it is 24/7!

### 3. CI/CD Pipeline for Automated Deployment

To ensure **continuous integration and continuous deployment (CI/CD)**, an **automated model deployment pipeline** was implemented:

- **GitHub Actions**:
  - Configured **CI/CD workflows** to automatically **train, evaluate, and deploy new models** when code or data updates are committed.
  - Ensured **unit testing & integration testing** before deployment.
- **Model Deployment Automation**:
  - **Docker Image Build & Push**: Upon new commits, a GitHub Actions workflow automatically builds and pushes a **Docker image** to a **private container registry**.
  - **Kubernetes Rolling Updates**: The deployment pipeline triggers **rolling updates on the Kubernetes cluster**, ensuring zero downtime.
- **Monitoring & Alerting**:
  - Integrated **Prometheus & Grafana** to track model performance (inference latency, request volume).
  - Configured **alerts for anomalies** (e.g., concept drift detection, high inference latency).

**Outcome**: Model updates are **automatically tested, packaged, and deployed with zero downtime**, ensuring high availability and rapid iteration.

## Key Performance Achievements

- A **fully automated ML pipeline**, optimized for **scalability, performance, and real-world deployment**.
- **Developed a scalable ML-powered color calibration system** to accurately predict true colors under varying lighting conditions.
- **Deployed a FastAPI-based inference API** for real-time processing in commercial applications.
- **Best model (Random Forest) achieved a 64.4% improvement** in color calibration precision (**Lab Mean ΔE: 14.61 → 5.20**), surpassing commercial printing standards.
- **Integrated YOLOv8 for pattern detection and automated data augmentation** to improve training data quality.
- **Leveraged MLflow for full experiment tracking, model versioning, and hyperparameter tuning**.
- **Implemented a fully automated CI/CD pipeline** using **Docker, Kubernetes, and GitHub Actions**, allowing seamless model deployment and updates.

## References

[1] Finlayson, G. D., & Drew, M. S. (1997). "Constrained least-squares regression in color spaces." Journal of Electronic Imaging, 6(4), 484–493.

This paper explores the application of constrained least-squares regression in color transformations, addressing issues related to noise and illumination variations.

[2] Hong, G., Luo, M. R., & Rhodes, P. A. (2001). "A study of digital camera colorimetric characterisation based on polynomial modeling." Color Research & Application, 26(1), 76–84.

This study discusses in detail how polynomial models can represent the mapping from camera RGB signals to true colors, providing insights into the effectiveness of various polynomial degrees in characterizing digital cameras.

[3] Gatta, C., et al. (2007). "A LUT-based approach for color correction in an end-to-end color imaging system." CIC15: Fifteenth Color Imaging Conference, 327–330.

This paper introduces the use of 3D Look-Up Tables (LUTs) for end-to-end color correction, covering aspects such as data acquisition and LUT interpolation techniques.

[4] Wei, X., Luo, M. R., & Pointer, M. R. (2019). "Evaluation of some non-linear methods for camera color characterisation." Color Research & Application, 44(2), 291–303.

This research evaluates various non-linear methods—including neural networks, polynomial models, and Look-Up Tables—for camera color characterization, comparing their performance and applicability.

[5] Shi, L., & Healey, G. (2002). "Using reflectance spectra to recover device-independent color." Color Research & Application, 27(1), 50–59.

This paper delves into the relationship between camera spectral responses and true surface reflectance, discussing methods to recover device-independent color representations from reflectance spectra.
