"""
Model module for INERB drunk/sober detection.
"""

import os
import glob
import cv2
import numpy as np
import joblib
from typing import Tuple, List, Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from . import features

DEFAULT_MODEL_PATH = "models/detection_model.pkl"

def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_image_as_array(path: str) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(path)
        return img
    except Exception:
        return None

def save_model(model_obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model_obj, path)

class DetectionModel:
    def __init__(self, model: Optional[RandomForestClassifier] = None):
        self.model = model
        self.feature_names = ["eye_aspect_ratio", "cheek_redness", "mouth_opening", "eye_circularity", "lip_line_curvature"]
    
    def train(self, X: np.ndarray, y: np.ndarray, n_estimators: int = 100, random_state: int = 42) -> Dict[str, Any]:
        if len(X) >= 4:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        else:
            X_train, X_test, y_train, y_test = X.copy(), X.copy(), y.copy(), y.copy()
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        self.model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        return {"accuracy": accuracy, "n_samples": len(X), "feature_importances": dict(zip(self.feature_names, self.model.feature_importances_))}
    
    def predict(self, features: List[float]) -> Tuple[int, float]:
        if self.model is None:
            raise ValueError("Model not loaded")
        arr = np.array(features).reshape(1, -1)
        pred = self.model.predict(arr)[0]
        probs = self.model.predict_proba(arr)[0]
        return int(pred), float(max(probs))
    
    def predict_with_details(self, features: List[float]) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not loaded")
        arr = np.array(features).reshape(1, -1)
        pred = self.model.predict(arr)[0]
        probs = self.model.predict_proba(arr)[0]
        return {"prediction": "drunk" if pred == 1 else "sober", "confidence": float(max(probs)), "probabilities": {"sober": float(probs[0]), "drunk": float(probs[1])}}
    
    def save(self, path: str) -> None:
        if self.model:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
    
    def load(self, path: str) -> None:
        self.model = joblib.load(path)

def train_model(data_dir: str = "data", model_path: str = DEFAULT_MODEL_PATH, n_estimators: int = 100) -> Dict[str, Any]:
    project_root = get_project_root()
    data_path = os.path.join(project_root, data_dir)
    model_save_path = os.path.join(project_root, model_path)
    
    print(f"Project root: {project_root}")
    print(f"Data path: {data_path}")
    
    X, y = [], []
    
    for label, label_val in [("drunk", 1), ("sober", 0)]:
        label_dir = os.path.join(data_path, label)
        print(f"Looking in: {label_dir}")
        
        if not os.path.exists(label_dir):
            print(f"Dir not found: {label_dir}")
            continue
        
        for ext in ["jpg", "jpeg", "png"]:
            for path in glob.glob(os.path.join(label_dir, f"*.{ext}")):
                img = load_image_as_array(path)
                if img is not None:
                    f = features.extract_features(img)
                    if f is not None:
                        X.append(f)
                        y.append(label_val)
    
    print(f"Loaded {len(X)} images")
    
    if len(X) < 2:
        raise ValueError(f"Not enough data! Found {len(X)} images.")
    
    X, y = np.array(X), np.array(y)
    m = DetectionModel()
    r = m.train(X, y, n_estimators)
    m.save(model_save_path)
    r["model_path"] = model_save_path
    return r

def load_model(path: str = DEFAULT_MODEL_PATH) -> DetectionModel:
    project_root = get_project_root()
    full_path = os.path.join(project_root, path)
    m = DetectionModel()
    m.load(full_path)
    return m

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()
    r = train_model(args.data_dir, args.model_path, args.n_estimators)
    print(f"Done! Accuracy: {r['accuracy']:.2f}")
