"""
YOLO-based Pattern Detection Module
"""

import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO
import numpy as np


# ================================
# 1. Load Configuration
# ================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  
CONFIG = {
    "yolo_model_1": ROOT_DIR / "data/models/detect/yolo_0214_first.pt",
    "yolo_model_2": ROOT_DIR / "data/models/detect/yolo_0214_second.pt",
}

# ================================
# 2. YOLO-based Pattern Detection

# ================================
class PatternDetector:
    """
    A class for detecting color calibration patterns using YOLO models.
    
    Attributes:
        model1 (YOLO): The first YOLO model for detecting the main card.
        model2 (YOLO): The second YOLO model for detecting patterns within the cropped card.
    """

    def __init__(self):
        """Initialize YOLO models for detection."""
        self.model1 = YOLO(CONFIG["yolo_model_1"])
        self.model2 = YOLO(CONFIG["yolo_model_2"])

    def getPattern(self, image: np.ndarray, show_results: bool = True) -> List[Tuple[Optional[np.ndarray], Optional[str]]]:
        """
        Extract patterns from an image using YOLO models and return detected patterns with class labels.

        Args:
            image (np.ndarray): Input image (BGR format).
            show_results (bool): Whether to display detection results.

        Returns:
            List[Tuple[np.ndarray, str]]: A list of (cropped_pattern_image, class_name), padded to length 4.
        """
        # First model detection - detect the main card
        results1 = self.model1(image)
        boxes = results1[0].boxes if results1 else []

        if len(boxes) == 0:
            print("Card Not Recognized")
            return [(None, None)] * 4  # Ensure return length is always 4

        # Extract bounding box of the detected card
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w, _ = image.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        image_crop = image[y1:y2, x1:x2]

        # Second model detection - detect patterns within the cropped card
        results2 = self.model2(image_crop)
        boxes = results2[0].boxes if results2 else []
        patterns = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            pattern_crop = image_crop[y1:y2, x1:x2]

            # Get class name from model prediction
            class_id = int(box.cls[0])
            class_name = results2[0].names[class_id]

            patterns.append((pattern_crop, class_name))

        # Ensure the list is always of length 4 by padding with (None, None)
        while len(patterns) < 4:
            patterns.append((None, None))
        patterns = patterns[:4]  # Truncate if more than 4

        # Visualization (optional)
        if show_results:
            self._visualize_patterns(image, image_crop, patterns)

        return patterns

    def _visualize_patterns(self, original_image: np.ndarray, cropped_image: np.ndarray, patterns: List[Tuple[Optional[np.ndarray], Optional[str]]]):
        """
        Visualize the detected card and patterns.

        Args:
            original_image (np.ndarray): The original input image.
            cropped_image (np.ndarray): The cropped card image.
            patterns (List[Tuple[np.ndarray, str]]): List of detected patterns.
        """
        import matplotlib.pyplot as plt
        import os

        plt.figure(figsize=(15, 5))

        # Show original image
        plt.subplot(1, 6, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        # Show cropped card
        plt.subplot(1, 6, 2)
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title("Cropped Card")
        plt.axis("off")

        # Show detected patterns
        for i, (pattern, class_name) in enumerate(patterns, start=1):
            plt.subplot(1, 6, i + 2)
            if pattern is not None:
                plt.imshow(cv2.cvtColor(pattern, cv2.COLOR_BGR2RGB))
                plt.title(f"{class_name}")
            else:
                plt.text(0.5, 0.5, "No Pattern", ha="center", va="center")
                plt.title("None")
            plt.axis("off")

        plt.tight_layout()
        
        # show patterns
        plt.show()

        # save patterns
        save_dir = ROOT_DIR / "outputs/yolo_detect"
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / "yolo_detection_result.png"
        plt.savefig(str(save_path))
        plt.close()
        print(f"saved at: {save_path}")

# ================================
# 3. Standalone Test (if run directly)
# ================================
if __name__ == "__main__":
    test_image_path = ROOT_DIR / "notebooks/demo_image/test_image.png"
    
    # Load test image
    test_image = cv2.imread(str(test_image_path))

    # Initialize detector and run pattern detection
    detector = PatternDetector()
    detected_patterns = detector.getPattern(test_image, show_results=True)

    print("\nDetected Patterns:")
    for i, (pattern, class_name) in enumerate(detected_patterns, start=1):
        if class_name:
            print(f"{i}. Class: {class_name}")
        else:
            print(f"{i}. No Pattern Detected")
