"""
THIS IS THE PRE-TRAINING DATA PROCESSING SCRIPT FOR FEATURE EXTRACTION.
(COMMONLY KNOWN AS DATA PROCESSING)
"""

import pandas as pd
import ast
from pathlib import Path

# Define the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FEATURE_DATA_PATH = ROOT_DIR / "data/features/feature_0216.csv"

def parse_rgb(rgb_string):
    """
    Convert a string in the format "(R, G, B)" into a tuple of integers (R, G, B).
    
    Args:
        rgb_string (str): RGB color in string format, e.g., "(255, 0, 0)".

    Returns:
        tuple: A tuple of integers representing the RGB values.
    """
    return ast.literal_eval(rgb_string)

def load_preprocessed_data(data_path=FEATURE_DATA_PATH):
    """
    Load and preprocess data, returning feature matrix X and target variable y.

    Args:
        data_path (str or Path): Path to the CSV file containing feature data.

    Returns:
        X (DataFrame): Feature matrix containing computed color differences and captured color values.
        y (DataFrame): Target variable (true color values in RGB).
    """

    # Read the dataset
    data = pd.read_csv(data_path)

    # Rename columns for clarity
    data = data.rename(columns={
        "color_code": "ColorCode",
        "unique_id": "UniqueID",
        "black_box": "Cp",
        "red_circle": "Rp",
        "green_triangle": "Gp",
        "blue_pentagon": "Bp",
        "real_rgb": "Cs"
    })

    # Define standard reference colors
    data["Rs"] = "(255, 0, 0)"
    data["Gs"] = "(0, 255, 0)"
    data["Bs"] = "(0, 0, 255)"

    # Rp, Gp, Bp, Cp: Colors in the captured image, influenced by lighting conditions.
    # Rs, Gs, Bs, Cs: Standard reference colors (true values).

    # Drop rows with missing values
    data = data.dropna()

    # Convert RGB strings into separate numerical R, G, B columns
    for col in ["Rp", "Gp", "Bp", "Cp"]:
        data[[f"{col}_R", f"{col}_G", f"{col}_B"]] = data[col].apply(parse_rgb).apply(pd.Series)

    # Convert target variable Cs into separate R, G, B columns
    data[["Cs_R", "Cs_G", "Cs_B"]] = data["Cs"].apply(parse_rgb).apply(pd.Series)

    # Compute color differences between reference colors and their expected values
    # - Red reference object (Rp) compared to standard red (255, 0, 0)
    data["Delta_RR_red"] = data["Rp_R"] - 255
    data["Delta_RG_red"] = data["Rp_G"] - 0
    data["Delta_RB_red"] = data["Rp_B"] - 0

    # - Green reference object (Gp) compared to standard green (0, 255, 0)
    data["Delta_RR_green"] = data["Gp_R"] - 0
    data["Delta_RG_green"] = data["Gp_G"] - 255
    data["Delta_RB_green"] = data["Gp_B"] - 0

    # - Blue reference object (Bp) compared to standard blue (0, 0, 255)
    data["Delta_RR_blue"] = data["Bp_R"] - 0
    data["Delta_RG_blue"] = data["Bp_G"] - 0
    data["Delta_RB_blue"] = data["Bp_B"] - 255

    # Construct the feature matrix X
    # - It includes: color difference values (9 columns) + captured color values (3 columns)
    X = data[
        [
            "Delta_RR_red", "Delta_RG_red", "Delta_RB_red",
            "Delta_RR_green", "Delta_RG_green", "Delta_RB_green",
            "Delta_RR_blue", "Delta_RG_blue", "Delta_RB_blue",
            "Cp_R", "Cp_G", "Cp_B",
        ]
    ]

    # Define the target variable y (true color values: R, G, B)
    y = data[["Cs_R", "Cs_G", "Cs_B"]]

    # Optionally, drop original columns if they are no longer needed
    # data = data.drop(columns=["Rp", "Gp", "Bp", "Cp", "Cs", "Rs", "Gs", "Bs"])
    # (Modify according to your requirements)

    return X, y 

# Test the function if the script is executed directly
if __name__ == "__main__":
    X, y = load_preprocessed_data()
    print(f"✅ Data successfully loaded! X shape: {X.shape}, y shape: {y.shape}")
