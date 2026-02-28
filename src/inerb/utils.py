"""
Utility functions for INERB drunk/sober detection.
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple, Dict


def load_image(image_source) -> Optional[np.ndarray]:
    """Load an image from file path, bytes, or PIL Image."""
    try:
        if isinstance(image_source, str):
            img = cv2.imread(image_source)
            return img
        elif isinstance(image_source, bytes):
            nparr = np.frombuffer(image_source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        else:
            return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def validate_image(img: np.ndarray) -> Tuple[bool, str]:
    """Validate that the image is suitable for processing."""
    if img is None:
        return False, "Image is None"
    
    if len(img.shape) != 3:
        return False, "Image must be a color image (3 channels)"
    
    if img.shape[2] != 3:
        return False, "Image must have 3 color channels"
    
    h, w = img.shape[:2]
    if h < 100 or w < 100:
        return False, "Image too small. Minimum size is 100x100 pixels"
    
    return True, ""


def format_prediction_result(prediction: str, confidence: float, probabilities: dict) -> dict:
    """Format prediction result for display."""
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "label": "Drunk" if prediction == "drunk" else "Sober",
        "color": "#e74c3c" if prediction == "drunk" else "#2ecc71",
        "emoji": "🍺" if prediction == "drunk" else "💚"
    }


def get_model_path() -> str:
    """Get the path to the trained model."""
    return "models/detection_model.pkl"


def ensure_directory(path: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)
