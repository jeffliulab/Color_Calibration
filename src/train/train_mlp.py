"""
TRAINING - NEURAL NETWORK (MLP)
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from pathlib import Path
from skimage import color
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from src.train.pre_train import load_preprocessed_data

# ================================
# 1. Define Paths & Configurations
# ================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_CONFIG = {
    "model_type": "mlp",
    "version": "v1",
    "extension": "pth",
}

MODEL_PATH = ROOT_DIR / "data/models" / MODEL_CONFIG["model_type"] / f"model_{MODEL_CONFIG['version']}.{MODEL_CONFIG['extension']}"

# ================================
# 2. Define Neural Network
# ================================
class ColorPredictor(nn.Module):
    def __init__(self, input_dim):
        super(ColorPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Output 3 values (R, G, B)
        )

    def forward(self, x):
        return self.model(x)

# ================================
# 3. Train Neural Network
# ================================
def train_neural_network(save_model=True, epochs=500, batch_size=32, lr=0.001):
    """
    Train a simple neural network for color prediction.

    Args:
        save_model (bool): Whether to save the trained model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.

    Returns:
        tuple: (model, X_test, y_test, y_pred)
    """
    # Load data
    X, y = load_preprocessed_data()
    
    # Convert to PyTorch tensors
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Initialize model, loss function, optimizer
    model = ColorPredictor(input_dim=X.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred_train = model(X_train_tensor)
        loss = loss_fn(y_pred_train, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Print progress
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")

    # Save model
    if save_model:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved at {MODEL_PATH}")

    # Predict on test set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()

    return model, X_test, y_test, y_pred

# ================================
# 4. Model Evaluation
# ================================
def evaluate_neural_network(y_test, y_pred):
    """
    Evaluate the trained neural network model.
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
    print("Neural Network Evaluation:")
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
# 5. Main Execution
# ================================
if __name__ == "__main__":
    # Train model and get test results
    model, X_test, y_test, y_pred = train_neural_network()
    
    # Evaluate the model
    evaluate_neural_network(y_test, y_pred)
