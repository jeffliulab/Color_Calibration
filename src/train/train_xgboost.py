"""
TRAINING - XGBOOST (3x3 MATRIX METHOD)
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from pathlib import Path
from skimage import color
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.train.pre_train import load_preprocessed_data

# ================================
# 1. Define Paths & Configurations
# ================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_CONFIG = {
    "model_type": "xgboost",
    "version": "v1",
    "extension": "pkl",
}

MODEL_PATH = ROOT_DIR / "data/models" / MODEL_CONFIG["model_type"] / f"model_{MODEL_CONFIG['version']}.{MODEL_CONFIG['extension']}"

# ================================
# 2. Train XGBoost Model
# ================================
def train_xgboost(save_model=True):
    """
    Train an XGBoost model for multi-output regression (predicting RGB values).

    Args:
        save_model (bool): Whether to save the trained model.

    Returns:
        tuple: (xgb_model, X_test, y_test, y_pred)
    """
    # Load data
    X, y = load_preprocessed_data()

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train XGBoost model (supports multi-output regression natively)
    xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, objective='reg:squarederror', random_state=42)
    xgb_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = xgb_model.predict(X_test)

    # Save model
    if save_model:
        joblib.dump(xgb_model, MODEL_PATH)
        print(f"Model saved at {MODEL_PATH}")

    return xgb_model, X_test, y_test, y_pred

# ================================
# 3. Model Evaluation
# ================================
def evaluate_xgboost(y_test, y_pred):
    """
    Evaluate the trained XGBoost model.
    """
    # Scatter Plot for Real vs Predicted RGB
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    colors = ['red', 'green', 'blue']
    labels = ['R', 'G', 'B']

    for i in range(3):
        axs[i].scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.5, color=colors[i])
        axs[i].plot([min(y_test.iloc[:, i]), max(y_test.iloc[:, i])],
                    [min(y_test.iloc[:, i]), max(y_test.iloc[:, i])], linestyle='--', color='black')
        axs[i].set_title(f"Real vs Predicted ({labels[i]} channel)")
        axs[i].set_xlabel("Real")
        axs[i].set_ylabel("Predicted")

    plt.tight_layout()
    plt.show()

    # Performance Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    delta_e = np.sqrt(((y_test - y_pred) ** 2).sum(axis=1)).mean()

    print()
    print("XGBoost Evaluation:")
    print()
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Mean Color Difference (ΔE): {delta_e:.2f}")

    # Error Distribution
    error = np.sqrt(((y_test - y_pred) ** 2).sum(axis=1))
    sns.histplot(error, bins=30, kde=True)
    plt.xlabel("Color Difference (ΔE)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.show()

    # Visualize Sample Colors
    fig, ax = plt.subplots(10, 2, figsize=(5, 25))  # Display 10 sets of contrasting colors
    for i in range(10):
        true_color = tuple(y_test.iloc[i] / 255)  # Normalized to 0-1
        pred_color = tuple(y_pred[i] / 255)

        ax[i, 0].add_patch(patches.Rectangle((0, 0), 1, 1, color=true_color))
        ax[i, 0].set_title("Real")
        ax[i, 0].axis("off")

        ax[i, 1].add_patch(patches.Rectangle((0, 0), 1, 1, color=pred_color))
        ax[i, 1].set_title("Predicted")
        ax[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

    # Compute Lab Color Difference
    y_test_norm = y_test.to_numpy() / 255.0
    y_pred_norm = y_pred / 255.0

    y_test_lab = color.rgb2lab(y_test_norm.reshape(-1, 1, 3)).reshape(-1, 3)
    y_pred_lab = color.rgb2lab(y_pred_norm.reshape(-1, 1, 3)).reshape(-1, 3)

    delta_e_lab = np.sqrt(np.sum((y_test_lab - y_pred_lab) ** 2, axis=1))
    mean_delta_e_lab = np.mean(delta_e_lab)
    median_delta_e_lab = np.median(delta_e_lab)

    print()
    print(f"Mean ΔE (Lab): {mean_delta_e_lab:.2f}")
    print(f"Median ΔE (Lab): {median_delta_e_lab:.2f}")
    print()

# ================================
# 4. Main Execution
# ================================
if __name__ == "__main__":
    # Train model and get test results
    model, X_test, y_test, y_pred = train_xgboost()
    
    # Evaluate the model
    evaluate_xgboost(y_test, y_pred)
