"""
Dataset handling module for INERB drunk/sober detection.

This module handles loading, preprocessing, and exporting of training data.
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split

from .features import extract_features, get_feature_names


def load_dataset(
    data_dir: str = "data",
    drunk_dir: str = "drunk",
    sober_dir: str = "sober"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images and extract features from both drunk and sober directories.
    
    Args:
        data_dir: Root data directory.
        drunk_dir: Name of drunk images subdirectory.
        sober_dir: Name of sober images subdirectory.
        
    Returns:
        Tuple of (features, labels) arrays.
        features: Array of shape (n_samples, 5).
        labels: Array of shape (n_samples,), where 1=drunk, 0=sober.
    """
    X, y = [], []
    
    # Load drunk images (label = 1)
    drunk_path = os.path.join(data_dir, drunk_dir)
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for path in glob.glob(os.path.join(drunk_path, ext)):
            img = cv2.imread(path)
            if img is not None:
                features = extract_features(img)
                if features is not None:
                    X.append(features)
                    y.append(1)
    
    # Load sober images (label = 0)
    sober_path = os.path.join(data_dir, sober_dir)
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for path in glob.glob(os.path.join(sober_path, ext)):
            img = cv2.imread(path)
            if img is not None:
                features = extract_features(img)
                if features is not None:
                    X.append(features)
                    y.append(0)
    
    if len(X) == 0:
        raise ValueError(
            f"No valid images found. Please add images to '{drunk_path}' and '{sober_path}'."
        )
    
    return np.array(X), np.array(y)


def load_images_from_directory(directory: str) -> List[Tuple[str, np.ndarray]]:
    """
    Load all images from a directory.
    
    Args:
        directory: Path to the directory containing images.
        
    Returns:
        List of tuples (image_path, image_array).
    """
    images = []
    
    if not os.path.exists(directory):
        return images
    
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for path in glob.glob(os.path.join(directory, ext)):
            img = cv2.imread(path)
            if img is not None:
                images.append((path, img))
    
    return images


def extract_all_features(
    data_dir: str = "data",
    drunk_dir: str = "drunk",
    sober_dir: str = "sober"
) -> Tuple[List[List[float]], List[int], List[str]]:
    """
    Extract features from all images and return with labels and paths.
    
    Args:
        data_dir: Root data directory.
        drunk_dir: Name of drunk images subdirectory.
        sober_dir: Name of sober images subdirectory.
        
    Returns:
        Tuple of (features_list, labels_list, image_paths).
    """
    features_list = []
    labels_list = []
    image_paths = []
    
    # Process drunk images
    drunk_path = os.path.join(data_dir, drunk_dir)
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for path in glob.glob(os.path.join(drunk_path, ext)):
            img = cv2.imread(path)
            if img is not None:
                features = extract_features(img)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(1)  # drunk
                    image_paths.append(path)
    
    # Process sober images
    sober_path = os.path.join(data_dir, sober_dir)
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for path in glob.glob(os.path.join(sober_path, ext)):
            img = cv2.imread(path)
            if img is not None:
                features = extract_features(img)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(0)  # sober
                    image_paths.append(path)
    
    return features_list, labels_list, image_paths


def export_features_to_csv(
    output_path: str = "features.csv",
    data_dir: str = "data",
    drunk_dir: str = "drunk",
    sober_dir: str = "sober"
) -> pd.DataFrame:
    """
    Extract features from all images and export to CSV.
    
    Args:
        output_path: Path to save the CSV file.
        data_dir: Root data directory.
        drunk_dir: Name of drunk images subdirectory.
        sober_dir: Name of sober images subdirectory.
        
    Returns:
        DataFrame with features and labels.
    """
    features_list, labels_list, image_paths = extract_all_features(
        data_dir, drunk_dir, sober_dir
    )
    
    feature_names = get_feature_names()
    
    df = pd.DataFrame(features_list, columns=feature_names)
    df["label"] = labels_list
    df["label_name"] = df["label"].map({0: "sober", 1: "drunk"})
    df["image_path"] = image_paths
    
    df.to_csv(output_path, index=False)
    
    return df


def get_dataset_statistics(
    data_dir: str = "data",
    drunk_dir: str = "drunk",
    sober_dir: str = "sober"
) -> Dict[str, int]:
    """
    Get statistics about the dataset.
    
    Args:
        data_dir: Root data directory.
        drunk_dir: Name of drunk images subdirectory.
        sober_dir: Name of sober images subdirectory.
        
    Returns:
        Dictionary with dataset statistics.
    """
    features_list, labels_list, _ = extract_all_features(
        data_dir, drunk_dir, sober_dir
    )
    
    return {
        "total_samples": len(features_list),
        "drunk_samples": sum(labels_list),
        "sober_samples": len(labels_list) - sum(labels_list),
        "feature_dim": len(features_list[0]) if features_list else 0,
    }


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and testing sets.
    
    Args:
        X: Feature matrix.
        y: Labels.
        test_size: Fraction of data for testing.
        random_state: Random state for reproducibility.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def validate_dataset(data_dir: str) -> Tuple[bool, List[str]]:
    """
    Validate that the dataset has sufficient samples for training.
    
    Args:
        data_dir: Root data directory.
        
    Returns:
        Tuple of (is_valid, error_messages).
    """
    errors = []
    
    drunk_dir = os.path.join(data_dir, "drunk")
    sober_dir = os.path.join(data_dir, "sober")
    
    # Check directories exist
    if not os.path.exists(drunk_dir):
        errors.append(f"Drunk directory not found: {drunk_dir}")
    if not os.path.exists(sober_dir):
        errors.append(f"Sober directory not found: {sober_dir}")
    
    if errors:
        return False, errors
    
    # Count images (support all formats)
    n_drunk = (
        len(glob.glob(os.path.join(drunk_dir, "*.jpg"))) +
        len(glob.glob(os.path.join(drunk_dir, "*.jpeg"))) +
        len(glob.glob(os.path.join(drunk_dir, "*.png")))
    )
    n_sober = (
        len(glob.glob(os.path.join(sober_dir, "*.jpg"))) +
        len(glob.glob(os.path.join(sober_dir, "*.jpeg"))) +
        len(glob.glob(os.path.join(sober_dir, "*.png")))
    )
    
    # Check minimum requirements
    if n_drunk < 1:
        errors.append(f"Need at least 1 drunk image, found {n_drunk}")
    if n_sober < 1:
        errors.append(f"Need at least 1 sober image, found {n_sober}")
    
    if n_drunk + n_sober < 2:
        errors.append("Need at least 2 total images for training")
    
    return len(errors) == 0, errors
