"""
Feature extraction module for INERB drunk/sober detection.

This module uses Mediapipe face mesh to extract facial landmarks and compute
various features used for drunk/sober classification.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional, Any


# Initialize Mediapipe FaceMesh (lazy loading)
_face_mesh = None


def _get_face_mesh():
    """Get or initialize the Mediapipe FaceMesh instance."""
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    return _face_mesh


def get_landmarks(img: np.ndarray) -> Optional[List[Any]]:
    """
    Extract facial landmarks from an image using Mediapipe FaceMesh.
    
    Args:
        img: Input image in BGR format (OpenCV format).
        
    Returns:
        List of landmark points if face is detected, None otherwise.
    """
    if img is None:
        return None
    
    # Convert to RGB for Mediapipe
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = _get_face_mesh().process(rgb_img)
    
    # Return landmarks if face detected
    if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
        return results.multi_face_landmarks[0].landmark
    
    return None


def _distance(point1: Any, point2: Any, img_height: int, img_width: int) -> float:
    """Calculate Euclidean distance between two normalized landmark points."""
    return np.linalg.norm([
        (point1.x - point2.x) * img_width,
        (point1.y - point2.y) * img_height
    ])


def eye_aspect_ratio(landmarks: List[Any], img: np.ndarray) -> float:
    """Calculate the Eye Aspect Ratio (EAR)."""
    ih, iw = img.shape[:2]
    
    vertical_opening = (
        _distance(landmarks[159], landmarks[153], ih, iw) +
        _distance(landmarks[158], landmarks[144], ih, iw)
    ) / 2
    
    horizontal_opening = _distance(landmarks[33], landmarks[133], ih, iw)
    
    if horizontal_opening == 0:
        return 0.0
    
    return vertical_opening / horizontal_opening


def cheek_redness(img: np.ndarray) -> float:
    """Calculate cheek redness by analyzing hue in HSV color space."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask for red hues (0-10 degrees)
    mask = cv2.inRange(hsv[:, :, 0], 0, 10)
    
    return mask.sum() / 255 / img.size


def mouth_opening(landmarks: List[Any], img: np.ndarray) -> float:
    """Calculate mouth opening distance."""
    if landmarks is None:
        return 0.0
    
    ih = img.shape[0]
    return abs(landmarks[13].y - landmarks[14].y) * ih


def eye_circularity(landmarks: List[Any], img: np.ndarray) -> float:
    """Calculate eye circularity."""
    ih, iw = img.shape[:2]
    
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    left_eye_left = landmarks[33]
    left_eye_right = landmarks[133]
    
    vertical = abs(left_eye_top.y - left_eye_bottom.y) * ih
    horizontal = abs(left_eye_right.x - left_eye_left.x) * iw
    
    if horizontal == 0:
        return 0.0
    
    return vertical / horizontal


def lip_line_curvature(landmarks: List[Any], img: np.ndarray) -> float:
    """Calculate lip line curvature."""
    ih, iw = img.shape[:2]
    
    left_corner = landmarks[61]
    right_corner = landmarks[291]
    center = landmarks[13]
    
    left_y = left_corner.y * ih
    right_y = right_corner.y * ih
    center_y = center.y * ih
    
    return abs((left_y + right_y) / 2 - center_y)


def extract_features(img: np.ndarray) -> Optional[List[float]]:
    """Extract all 5 features from a face image."""
    landmarks = get_landmarks(img)
    
    if landmarks is None:
        return None
    
    features = [
        eye_aspect_ratio(landmarks, img),
        cheek_redness(img),
        mouth_opening(landmarks, img),
        eye_circularity(landmarks, img),
        lip_line_curvature(landmarks, img),
    ]
    
    return features


def get_feature_names() -> List[str]:
    """Get the names of all extracted features."""
    return [
        "eye_aspect_ratio",
        "cheek_redness",
        "mouth_opening",
        "eye_circularity",
        "lip_line_curvature",
    ]


def extract_features_with_details(img: np.ndarray) -> Optional[dict]:
    """Extract features and return them with their names and values."""
    features = extract_features(img)
    
    if features is None:
        return None
    
    feature_names = get_feature_names()
    return dict(zip(feature_names, features))
