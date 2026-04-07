"""
Model loading and inference utilities for the Color Calibration Space.

Models are downloaded from HuggingFace Hub on first use and cached locally.
"""

import os
import logging
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_MODEL_REPO = "jeffliulab/card-calibration-v1"
CACHE_DIR = Path(__file__).parent / "checkpoints"
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_model_cache = {}


def _download_model(filename: str) -> str:
    """Download a model file from HF Hub if not already cached locally."""
    local_path = CACHE_DIR / filename
    if local_path.exists():
        logger.info(f"Using cached model: {local_path}")
        return str(local_path)

    logger.info(f"Downloading {filename} from {HF_MODEL_REPO} ...")
    path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=filename,
        repo_type="model",
        local_dir=str(CACHE_DIR),
        token=os.environ.get("HF_TOKEN"),
    )
    logger.info(f"Downloaded to {path}")
    return path


def load_yolo_model(filename: str) -> YOLO:
    """Load a YOLO model, downloading from HF Hub if needed."""
    if filename in _model_cache:
        return _model_cache[filename]
    path = _download_model(filename)
    model = YOLO(path)
    _model_cache[filename] = model
    return model


def load_calibration_model(filename: str):
    """Load a scikit-learn / xgboost model saved with joblib."""
    if filename in _model_cache:
        return _model_cache[filename]
    path = _download_model(filename)
    model = joblib.load(path)
    _model_cache[filename] = model
    return model


# ---------------------------------------------------------------------------
# Detection pipeline
# ---------------------------------------------------------------------------

# Standard reference colors
REFERENCE_COLORS = {
    "Rs": (255, 0, 0),
    "Gs": (0, 255, 0),
    "Bs": (0, 0, 255),
}

FEATURE_NAMES = [
    "Delta_RR_red", "Delta_RG_red", "Delta_RB_red",
    "Delta_RR_green", "Delta_RG_green", "Delta_RB_green",
    "Delta_RR_blue", "Delta_RG_blue", "Delta_RB_blue",
    "Cp_R", "Cp_G", "Cp_B",
]


def detect_card(image: np.ndarray, yolo1: YOLO) -> np.ndarray | None:
    """Detect the calibration card in the image and return the cropped card."""
    results = yolo1(image)
    boxes = results[0].boxes if results else []
    if len(boxes) == 0:
        return None
    box = boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


def detect_patterns(card_image: np.ndarray, yolo2: YOLO) -> dict:
    """Detect color patterns within the cropped card image.

    Returns a dict mapping class_name -> (crop, avg_rgb).
    """
    results = yolo2(card_image)
    boxes = results[0].boxes if results else []
    patterns = {}
    h, w = card_image.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = card_image[y1:y2, x1:x2]
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]

        # Extract average RGB from center 1/3 region
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ch, cw = crop_rgb.shape[:2]
        r0, r1 = ch // 3, 2 * (ch // 3)
        c0, c1 = cw // 3, 2 * (cw // 3)
        center = crop_rgb[r0:r1, c0:c1]
        avg_rgb = tuple(np.mean(center, axis=(0, 1)).astype(int))

        patterns[class_name] = (crop, avg_rgb)

    return patterns


def build_features(patterns: dict) -> np.ndarray:
    """Build the 12-dimensional feature vector from detected patterns.

    Returns a (1, 12) DataFrame ready for model.predict().
    """
    Rp = patterns["red_circle"][1]
    Gp = patterns["green_triangle"][1]
    Bp = patterns["blue_pentagon"][1]
    Cp = patterns["black_box"][1]

    features = [
        Rp[0] - 255, Rp[1] - 0, Rp[2] - 0,       # Delta red reference
        Gp[0] - 0,   Gp[1] - 255, Gp[2] - 0,       # Delta green reference
        Bp[0] - 0,   Bp[1] - 0,   Bp[2] - 255,      # Delta blue reference
        Cp[0], Cp[1], Cp[2],                          # Captured target color
    ]
    return pd.DataFrame([features], columns=FEATURE_NAMES)


def predict_color(image: np.ndarray, model_name: str = "xgboost") -> dict:
    """Run the full pipeline: detect card -> detect patterns -> predict true color.

    Args:
        image: Input image in BGR format (as read by cv2).
        model_name: Which calibration model to use ("xgboost" or "random_forest").

    Returns:
        dict with keys: captured_rgb, predicted_rgb, patterns, card_crop, delta_e
    """
    # Load models
    yolo1 = load_yolo_model("yolo_first.pt")
    yolo2 = load_yolo_model("yolo_second.pt")

    model_files = {
        "xgboost": "xgboost_v1.pkl",
        "random_forest": "random_forest_v1.pkl",
    }
    cal_model = load_calibration_model(model_files[model_name])

    # Step 1: Detect card
    card_crop = detect_card(image, yolo1)
    if card_crop is None:
        raise ValueError("Could not detect a calibration card in the image. "
                         "Make sure the card is clearly visible.")

    # Step 2: Detect patterns
    patterns = detect_patterns(card_crop, yolo2)
    required = ["red_circle", "green_triangle", "blue_pentagon", "black_box"]
    missing = [p for p in required if p not in patterns]
    if missing:
        raise ValueError(f"Missing patterns: {missing}. "
                         f"Detected: {list(patterns.keys())}")

    # Step 3: Build features and predict
    features_df = build_features(patterns)
    predicted_rgb = tuple(cal_model.predict(features_df)[0].astype(int))
    predicted_rgb = tuple(max(0, min(255, v)) for v in predicted_rgb)

    captured_rgb = patterns["black_box"][1]

    # Compute delta E (simple Euclidean in RGB space)
    delta_e = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(captured_rgb, predicted_rgb))))

    return {
        "captured_rgb": captured_rgb,
        "predicted_rgb": predicted_rgb,
        "patterns": patterns,
        "card_crop": card_crop,
        "delta_e": delta_e,
    }
