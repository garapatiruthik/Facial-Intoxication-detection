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
from .features import extract_features

DEFAULT_MODEL_PATH = "models/detection_model.pkl"

def load_image_as_array(path: str) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(path)
        return img
    except Exception:
        return None

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
    X, y = [], []
    for label, label_val in [("drunk", 1), ("sober", 0)]:
        for ext in ["jpg", "png"]:
            for path in glob.glob(os.path.join(data_dir, label, f"*.{ext}")):
                img = load_image_as_array(path)
                if img is not None:
                    f = extract_features(img)
                    if f: X.append(f); y.append(label_val)
    if len(X) < 2: raise ValueError("Not enough data!")
    X, y = np.array(X), np.array(y)
    m = DetectionModel()
    r = m.train(X, y, n_estimators)
    m.save(model_path)
    r["model_path"] = model_path
    return r

def predict(image_path: str, model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    img = load_image_as_array(image_path)
    if img is None: raise FileNotFoundError("Image not found")
    f = extract_features(img)
    if f is None: raise ValueError("No face detected")
    m = DetectionModel()
    m.load(model_path)
    return m.predict_with_details(f)

def load_model(path: str = DEFAULT_MODEL_PATH) -> DetectionModel:
    m = DetectionModel()
    m.load(path)
    return m

if __name__ == "__main__":
    import sys, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()
    r = train_model(args.data_dir, args.model_path, args.n_estimators)
    print(f"Done! Accuracy: {r['accuracy']:.2f}")
