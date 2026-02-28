"""
INERB - Drunk/Sober Detection System
"""

__version__ = "1.0.0"
__author__ = "INERB Team"

# Use dot notation for relative imports
from . import features
from . import model
from . import dataset
from . import utils

# Export main functions
extract_features = features.extract_features
get_landmarks = features.get_landmarks
train_model = model.train_model
predict = model.predict
load_model = model.load_model
save_model = model.save_model
load_dataset = dataset.load_dataset
export_features_to_csv = dataset.export_features_to_csv

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
