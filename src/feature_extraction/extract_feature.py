import csv
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os
import datetime
from tqdm import tqdm  # Progress bar library

# Define ROOT_DIR
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = ROOT_DIR / "data/processed/features/0215/"
OUTPUT_DIR = ROOT_DIR / "outputs/feature/0215"

# Ensure the output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_filename(filename):
    """
    Extracts color code, pattern, and unique identifier from the filename.
    
    Args:
        filename (str): Format should be like "1002_4B_noise_34_blue_pentagon.jpg"
        
    Returns:
        tuple: (color_code, unique_id, pattern) or (None, None, None) if parsing fails
        
    Example:
        For "1002_4B_noise_34_blue_pentagon.jpg":
        - color_code = "1002_4B"
        - unique_id = "noise_34"
        - pattern = "pentagon"
    """
    # Remove file extension first
    name_without_ext = filename.split('.')[0]
    
    # Count total underscores to validate format
    if name_without_ext.count('_') < 4:
        print(f"Warning: Filename does not have enough underscores: {filename}")
        return None, None, None
    
    try:
        # Find the second underscore from left (for color_code)
        second_underscore_left = -1
        first_found = False
        for i, char in enumerate(name_without_ext):
            if char == '_':
                if first_found:
                    second_underscore_left = i
                    break
                first_found = True
        
        # Find the second underscore from right (for pattern)
        reversed_name = name_without_ext[::-1]
        second_underscore_right = -1
        first_found = False
        for i, char in enumerate(reversed_name):
            if char == '_':
                if first_found:
                    second_underscore_right = len(name_without_ext) - i - 1
                    break
                first_found = True
        
        if second_underscore_left == -1 or second_underscore_right == -1:
            print(f"Warning: Could not find required underscores in filename: {filename}")
            return None, None, None
        
        # Extract the three parts
        color_code = name_without_ext[:second_underscore_left]
        unique_id = name_without_ext[second_underscore_left+1:second_underscore_right]
        pattern = name_without_ext[second_underscore_right+1:]
        
        return color_code, unique_id, pattern
        
    except Exception as e:
        print(f"Error parsing filename {filename}: {str(e)}")
        return None, None, None

def feature(image_path):
    """
    Extracts the average RGB color of the center region of a 3x3 grid.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB color space
    h, w, _ = img.shape  # Get image dimensions

    # Calculate the center region boundaries of the 3x3 grid
    grid_size = 3
    row_start = h // grid_size
    row_end = 2 * (h // grid_size)
    col_start = w // grid_size
    col_end = 2 * (w // grid_size)

    center_region = img[row_start:row_end, col_start:col_end]  # Extract center region
    avg_color = np.mean(center_region, axis=(0, 1))  # Compute mean RGB values

    return tuple(avg_color.astype(int))  # Return integer RGB values

def output(df, output_dir):
    """
    Writes extracted features to a CSV file with a timestamp.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"features_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Feature extraction completed. Results saved to {output_file}")

if __name__ == "__main__":
    # Get all image files
    image_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith(".jpg")]

    # Create DataFrame structure
    df = pd.DataFrame(columns=["color_code", "unique_id", "black_box", "red_circle", "green_triangle", "blue_pentagon"])

    # Iterate through input images with a progress bar
    for filename in tqdm(image_files, desc="Processing Images", unit="image"):
        image_path = FEATURES_DIR / filename
        color_code, unique_id, pattern = parse_filename(filename)
        feature_value = feature(image_path)

        if feature_value is None:
            continue  # Skip images that cannot be processed

        # Check if the same color_code + unique_id already exists
        existing_row = df[(df["color_code"] == color_code) & (df["unique_id"] == unique_id)]

        if existing_row.empty:
            # If not exist, create a new row
            new_row = {"color_code": color_code, "unique_id": unique_id,
                       "black_box": None, "red_circle": None, "green_triangle": None, "blue_pentagon": None}
            new_row[pattern] = feature_value  # Only update the extracted pattern's RGB value
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # If exists, update the corresponding pattern column
            row_index = existing_row.index[0]
            df.at[row_index, pattern] = feature_value

    # Save the CSV file (with a timestamp)
    output(df, OUTPUT_DIR)
