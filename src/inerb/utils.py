"""
Utility functions for INERB drunk/sober detection.

This module provides utility functions for image loading, face detection
validation, and result formatting.
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List


def load_image(path: str) -> Optional[np.ndarray]:
    """
    Load an image from a file path.
    
    Args:
        path: Path to the image file.
        
    Returns:
        Image as numpy array in BGR format, or None if loading failed.
    """
    if not os.path.exists(path):
        return None
    
    img = cv2.imread(path)
    return img


def load_image_from_bytes(bytes_data: bytes) -> Optional[np.ndarray]:
    """
    Load an image from byte data.
    
    Args:
        bytes_data: Image data as bytes.
        
    Returns:
        Image as numpy array in BGR format, or None if loading failed.
    """
    try:
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def validate_image_file(path: str) -> Tuple[bool, str]:
    """
    Validate that a file is a valid image.
    
    Args:
        path: Path to the image file.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    # Check file exists
    if not os.path.exists(path):
        return False, f"File not found: {path}"
    
    # Check file extension
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    ext = os.path.splitext(path)[1].lower()
    if ext not in valid_extensions:
        return False, f"Invalid file extension: {ext}. Supported: {valid_extensions}"
    
    # Try to load the image
    img = cv2.imread(path)
    if img is None:
        return False, "Could not read image file"
    
    # Check image dimensions
    if img.shape[0] < 50 or img.shape[1] < 50:
        return False, "Image too small (minimum 50x50 pixels)"
    
    return True, ""


def validate_face_detection(img: np.ndarray, min_size: int = 100) -> Tuple[bool, str]:
    """
    Validate that an image contains a detectable face of sufficient size.
    
    Args:
        img: Input image.
        min_size: Minimum face size in pixels.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if img is None:
        return False, "No image provided"
    
    height, width = img.shape[:2]
    
    if height < min_size or width < min_size:
        return False, f"Image too small: {width}x{height} (minimum {min_size}x{min_size})"
    
    return True, ""


def resize_image(
    img: np.ndarray,
    max_width: int = 1920,
    max_height: int = 1080
) -> np.ndarray:
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        img: Input image.
        max_width: Maximum width.
        max_height: Maximum height.
        
    Returns:
        Resized image.
    """
    height, width = img.shape[:2]
    
    # Check if resize needed
    if width <= max_width and height <= max_height:
        return img
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


def format_prediction_result(
    prediction: str,
    confidence: float,
    probabilities: Dict[str, float]
) -> Dict[str, Any]:
    """
    Format a prediction result for display.
    
    Args:
        prediction: Prediction string ("drunk" or "sober").
        confidence: Confidence score (0-1).
        probabilities: Dictionary with probabilities for each class.
        
    Returns:
        Formatted result dictionary.
    """
    # Determine color based on prediction
    if prediction == "drunk":
        color = "#e74c3c"  # Red
        emoji = "🍺"
    else:
        color = "#2ecc71"  # Green
        emoji = "💚"
    
    # Format percentage strings
    confidence_pct = f"{confidence * 100:.1f}%"
    sober_pct = f"{probabilities.get('sober', 0) * 100:.1f}%"
    drunk_pct = f"{probabilities.get('drunk', 0) * 100:.1f}%"
    
    return {
        "prediction": prediction,
        "prediction_display": f"{emoji} {prediction.upper()}",
        "confidence": confidence,
        "confidence_display": confidence_pct,
        "probabilities": {
            "sober": probabilities.get("sober", 0),
            "drunk": probabilities.get("drunk", 0),
        },
        "probabilities_display": {
            "sober": sober_pct,
            "drunk": drunk_pct,
        },
        "color": color,
    }


def get_feature_importance_display(
    feature_importances: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Format feature importances for display.
    
    Args:
        feature_importances: Dictionary mapping feature names to importance values.
        
    Returns:
        List of dictionaries with feature name, importance, and formatted percentage.
    """
    # Sort by importance
    sorted_features = sorted(
        feature_importances.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [
        {
            "name": name,
            "importance": importance,
            "percentage": f"{importance * 100:.1f}%",
        }
        for name, importance in sorted_features
    ]


def create_result_summary(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create a summary of multiple prediction results.
    
    Args:
        results: List of prediction result dictionaries.
        
    Returns:
        Summary dictionary.
    """
    if not results:
        return {
            "count": 0,
            "drunk_count": 0,
            "sober_count": 0,
            "average_confidence": 0,
        }
    
    drunk_count = sum(1 for r in results if r.get("prediction") == "drunk")
    sober_count = sum(1 for r in results if r.get("prediction") == "sober")
    avg_confidence = sum(r.get("confidence", 0) for r in results) / len(results)
    
    return {
        "count": len(results),
        "drunk_count": drunk_count,
        "sober_count": sober_count,
        "average_confidence": avg_confidence,
        "average_confidence_display": f"{avg_confidence * 100:.1f}%",
    }


def check_model_exists(model_path: str = "models/detection_model.pkl") -> bool:
    """
    Check if the trained model file exists.
    
    Args:
        model_path: Path to the model file.
        
    Returns:
        True if model exists, False otherwise.
    """
    return os.path.exists(model_path)


def get_dataset_paths(
    base_dir: str = "."
) -> Dict[str, str]:
    """
    Get paths to dataset directories.
    
    Args:
        base_dir: Base directory for the project.
        
    Returns:
        Dictionary with paths.
    """
    return {
        "data_dir": os.path.join(base_dir, "data"),
        "drunk_dir": os.path.join(base_dir, "data", "drunk"),
        "sober_dir": os.path.join(base_dir, "data", "sober"),
        "models_dir": os.path.join(base_dir, "models"),
        "model_path": os.path.join(base_dir, "models", "detection_model.pkl"),
        "config_path": os.path.join(base_dir, "config", "config.yaml"),
    }
