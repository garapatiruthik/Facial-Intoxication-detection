"""
Feature extraction module for INERB drunk/sober detection.

This module uses Mediapipe face mesh to extract facial landmarks and compute
various features used for drunk/sober classification.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional, Tuple, Any


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
    """
    Calculate Euclidean distance between two normalized landmark points.
    
    Args:
        point1: First landmark point (normalized coordinates).
        point2: Second landmark point (normalized coordinates).
        img_height: Image height in pixels.
        img_width: Image width in pixels.
        
    Returns:
        Euclidean distance in pixels.
    """
    return np.linalg.norm([
        (point1.x - point2.x) * img_width,
        (point1.y - point2.y) * img_height
    ])


def eye_aspect_ratio(landmarks: List[Any], img: np.ndarray) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) - ratio of eye height to width.
    
    The EAR is used to detect eye openness, which can indicate drowsiness
    or intoxication.
    
    Args:
        landmarks: List of facial landmarks from Mediapipe.
        img: Input image for dimensions.
        
    Returns:
        Eye Aspect Ratio value.
    """
    ih, iw = img.shape[:2]
    
    # Eye landmarks (using left eye indices from Mediapipe)
    # 159: upper left eyelid, 153: lower left eyelid
    # 158: upper right eyelid, 144: lower right eyelid
    # 33: left eye corner, 133: right eye corner
    
    vertical_opening = (
        _distance(landmarks[159], landmarks[153], ih, iw) +
        _distance(landmarks[158], landmarks[144], ih, iw)
    ) / 2
    
    horizontal_opening = _distance(landmarks[33], landmarks[133], ih, iw)
    
    if horizontal_opening == 0:
        return 0.0
    
    return vertical_opening / horizontal_opening


def cheek_redness(img: np.ndarray) -> float:
    """
    Calculate cheek redness by analyzing the hue channel in HSV color space.
    
    Reddish cheeks can be an indicator of alcohol consumption due to
    vasodilation.
    
    Args:
        img: Input image in BGR format.
        
    Returns:
        Normalized redness score (0 to 1).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask for red hues (hue values 0-10 in OpenCV)
    mask = cv2.inRange(hsv[:, :, 0], 0, 10)
    
    # Return ratio of red pixels to total pixels
    return mask.sum() / 255 / img.size


def mouth_opening(landmarks: List[Any], img: np.ndarray) -> float:
    """
    Calculate mouth opening distance.
    
    Mouth opening can increase due to relaxation from alcohol consumption.
    
    Args:
        landmarks: List of facial landmarks from Mediapipe.
        img: Input image for dimensions.
        
    Returns:
        Mouth opening distance in pixels.
    """
    if landmarks is None:
        return 0.0
    
    ih = img.shape[0]
    
    # Upper lip: landmark 13, Lower lip: landmark 14
    return abs(landmarks[13].y - landmarks[14].y) * ih


def eye_circularity(landmarks: List[Any], img: np.ndarray) -> float:
    """
    Calculate eye circularity - ratio of vertical to horizontal eye dimensions.
    
    This can indicate eye shape changes that may occur with intoxication.
    
    Args:
        landmarks: List of facial landmarks from Mediapipe.
        img: Input image for dimensions.
        
    Returns:
        Eye circularity ratio.
    """
    ih, iw = img.shape[:2]
    
    # Left eye region landmarks
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
    """
    Calculate lip line curvature.
    
    The curvature of the lips can change due to facial muscle relaxation
    from alcohol consumption.
    
    Args:
        landmarks: List of facial landmarks from Mediapipe.
        img: Input image for dimensions.
        
    Returns:
        Lip curvature value.
    """
    ih, iw = img.shape[:2]
    
    # Lip landmarks
    left_corner = landmarks[61]   # Left corner of mouth
    right_corner = landmarks[291]  # Right corner of mouth
    center = landmarks[13]       # Upper lip center
    
    # Calculate the vertical deviation from the center line
    left_y = left_corner.y * ih
    right_y = right_corner.y * ih
    center_y = center.y * ih
    
    # Curvature as deviation from straight line
    curvature = abs((left_y + right_y) / 2 - center_y)
    
    return curvature


def extract_features(img: np.ndarray) -> Optional[List[float]]:
    """
    Extract all 5 features from a face image.
    
    This is the main function that extracts all features used for
    drunk/sober classification:
    1. eye_aspect_ratio - Eye openness measure
    2. cheek_redness - Facial redness indicator
    3. mouth_opening - Mouth openness
    4. eye_circularity - Eye shape measure
    5. lip_line_curvature - Lip curvature measure
    
    Args:
        img: Input image in BGR format (OpenCV format).
        
    Returns:
        List of 5 feature values, or None if face not detected.
    """
    # Get facial landmarks
    landmarks = get_landmarks(img)
    
    if landmarks is None:
        return None
    
    # Extract all features
    features = [
        eye_aspect_ratio(landmarks, img),
        cheek_redness(img),
        mouth_opening(landmarks, img),
        eye_circularity(landmarks, img),
        lip_line_curvature(landmarks, img),
    ]
    
    return features


def get_feature_names() -> List[str]:
    """
    Get the names of all extracted features.
    
    Returns:
        List of feature names.
    """
    return [
        "eye_aspect_ratio",
        "cheek_redness",
        "mouth_opening",
        "eye_circularity",
        "lip_line_curvature",
    ]


def extract_features_with_details(img: np.ndarray) -> Optional[dict]:
    """
    Extract features and return them with their names and values.
    
    Args:
        img: Input image in BGR format.
        
    Returns:
        Dictionary with feature names as keys and values, or None if face not detected.
    """
    features = extract_features(img)
    
    if features is None:
        return None
    
    feature_names = get_feature_names()
    return dict(zip(feature_names, features))
