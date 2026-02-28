"""
Utility functions for INERB drunk/sober detection.
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple, Dict


def load_image(image_source: any) -> Optional[np.ndarray]:
    """
    Load an image from file path, bytes, or PIL Image.
    
    Args:
        image_source: Either a file path (str), bytes, or PIL Image.
        
    Returns:
        Image as numpy array in BGR format, or None if failed.
    """
    try:
        if isinstance(image_source, str):
            # File path
            img = cv2.imread(image_source)
            return img
        elif isinstance(image_source, bytes):
            # Bytes
            nparr = np.frombuffer(image_source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        else:
            return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def validate_image(img: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that the image is suitable for processing.
    
    Args:
        img: Input image as numpy array.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
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
    """
    Format prediction result for display.
    
    Args:
        prediction: "drunk" or "sober"
        confidence: Confidence score (0-1)
        probabilities: Dictionary with "drunk" and "sober" probabilities
        
    Returns:
        Formatted result dictionary.
    """
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "label": "Drunk" if prediction == "drunk" else "Sober",
        "color": "#e74c3c" if prediction == "drunk" else "#2ecc71",
        "emoji": "🍺" if prediction == "drunk" else "💚"
    }


def get_model_path() -> str:
    """
    Get the path to the trained model.
    
    Returns:
        Path to model file.
    """
    return "models/detection_model.pkl"


def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure exists.
    """
    os.makedirs(path, exist_ok=True)
