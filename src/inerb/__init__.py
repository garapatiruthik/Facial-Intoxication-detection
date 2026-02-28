"""
INERB - Drunk/Sober Detection System

A machine learning application that detects whether a person is drunk or sober
based on facial features extracted using Mediapipe.
"""

__version__ = "1.0.0"
__author__ = "INERB Team"

from .features import extract_features, get_landmarks
from .model import train_model, predict, load_model, save_model
from .dataset import load_dataset, export_features_to_csv

__all__ = [
    "extract_features",
    "get_landmarks",
    "train_model",
    "predict",
    "load_model",
    "save_model",
    "load_dataset",
    "export_features_to_csv",
]
