"""
Model module for INERB drunk/sober detection.

This module handles model training, saving, loading, and prediction
using a Random Forest classifier.
"""

import os
import glob
import cv2
import numpy as np
import joblib
from typing import Tuple, List, Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from .features import extract_features


# Default model path
DEFAULT_MODEL_PATH = "models/detection_model.pkl"


class DetectionModel:
    """Drunk/Sober detection model wrapper."""
    
    def __init__(self, model: Optional[RandomForestClassifier] = None):
        """
        Initialize the detection model.
        
        Args:
            model: Pre-trained RandomForestClassifier. If None, must be loaded or trained.
        """
        self.model = model
        self.feature_names = [
            "eye_aspect_ratio",
            "cheek_redness", 
            "mouth_opening",
            "eye_circularity",
            "lip_line_curvature",
        ]
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train the detection model.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target labels (0=sober, 1=drunk).
            n_estimators: Number of trees in the forest.
            random_state: Random state for reproducibility.
            
        Returns:
            Dictionary with training results and metrics.
        """
        # Split data for evaluation
        if len(X) >= 4:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )
        else:
            # Not enough samples for proper split - use all for training
            # and warn the user
            X_train, X_test = X.copy(), X.copy()
            y_train, y_test = y.copy(), y.copy()
            import warnings
            warnings.warn(
                f"Dataset too small ({len(X)} samples). Model trained on all data "
                "without validation split. Results may not be reliable."
            )
        
        # Train the model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            "accuracy": accuracy,
            "n_samples": len(X),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "feature_importances": dict(zip(
                self.feature_names,
                self.model.feature_importances_.tolist()
            ))
        }
    
    def predict(self, features: List[float]) -> Tuple[int, float]:
        """
        Make a prediction on a single set of features.
        
        Args:
            features: List of 5 feature values.
            
        Returns:
            Tuple of (prediction, confidence).
            prediction: 0 for sober, 1 for drunk.
            confidence: Probability of the predicted class.
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        features_array = np.array(features).reshape(1, -1)
        
        # Get prediction and probabilities
        prediction = self.model.predict(features_array)[0]
        probabilities = self.model.predict_proba(features_array)[0]
        confidence = max(probabilities)
        
        return int(prediction), float(confidence)
    
    def predict_with_details(self, features: List[float]) -> Dict[str, Any]:
        """
        Make a prediction with detailed probability information.
        
        Args:
            features: List of 5 feature values.
            
        Returns:
            Dictionary with prediction results and probabilities.
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        features_array = np.array(features).reshape(1, -1)
        
        prediction = self.model.predict(features_array)[0]
        probabilities = self.model.predict_proba(features_array)[0]
        
        return {
            "prediction": "drunk" if prediction == 1 else "sober",
            "prediction_code": int(prediction),
            "confidence": float(max(probabilities)),
            "probabilities": {
                "sober": float(probabilities[0]),
                "drunk": float(probabilities[1]),
            }
        }
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model.
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        joblib.dump(self.model, path)
    
    def load(self, path: str) -> None:
        """
        Load a model from a file.
        
        Args:
            path: Path to the saved model.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = joblib.load(path)


# Module-level functions for convenience

def train_model(
    data_dir: str = "data",
    model_path: str = DEFAULT_MODEL_PATH,
    n_estimators: int = 100,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Train the detection model from images in data directories.
    
    Args:
        data_dir: Root directory containing 'drunk' and 'sober' subdirectories.
        model_path: Path to save the trained model.
        n_estimators: Number of trees in the forest.
        test_size: Fraction of data to use for testing.
        
    Returns:
        Dictionary with training results.
    """
    X, y = [], []
    
    # Load drunk images (label = 1)
    drunk_dir = os.path.join(data_dir, "drunk")
    for path in glob.glob(os.path.join(drunk_dir, "*.jpg")):
        img = cv2.imread(path)
        if img is not None:
            features = extract_features(img)
            if features is not None:
                X.append(features)
                y.append(1)
    
    # Load sober images (label = 0)
    sober_dir = os.path.join(data_dir, "sober")
    for path in glob.glob(os.path.join(sober_dir, "*.jpg")):
        img = cv2.imread(path)
        if img is not None:
            features = extract_features(img)
            if features is not None:
                X.append(features)
                y.append(0)
    
    if len(X) < 2:
        raise ValueError(
            "Not enough data! Add more images to 'data/drunk' and 'data/sober' directories."
        )
    
    X = np.array(X)
    y = np.array(y)
    
    # Train the model
    detector = DetectionModel()
    result = detector.train(X, y, n_estimators=n_estimators)
    
    # Save the model
    detector.save(model_path)
    result["model_path"] = model_path
    
    return result


def predict(image_path: str, model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """
    Make a prediction on a single image.
    
    Args:
        image_path: Path to the input image.
        model_path: Path to the trained model.
        
    Returns:
        Dictionary with prediction results.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Extract features
    features = extract_features(img)
    if features is None:
        raise ValueError(f"No face detected in image: {image_path}")
    
    # Load model and predict
    detector = DetectionModel()
    detector.load(model_path)
    
    return detector.predict_with_details(features)


def save_model(model: RandomForestClassifier, path: str) -> None:
    """
    Save a trained model to a file.
    
    Args:
        model: Trained RandomForestClassifier.
        path: Path to save the model.
    """
    detector = DetectionModel(model=model)
    detector.save(path)


def load_model(path: str = DEFAULT_MODEL_PATH) -> DetectionModel:
    """
    Load a trained model from a file.
    
    Args:
        path: Path to the saved model.
        
    Returns:
        Loaded DetectionModel instance.
    """
    detector = DetectionModel()
    detector.load(path)
    return detector


# CLI entry points

def main_train():
    """Command-line interface for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train INERB detection model")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Model output path")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    
    args = parser.parse_args()
    
    result = train_model(
        data_dir=args.data_dir,
        model_path=args.model_path,
        n_estimators=args.n_estimators
    )
    
    print(f"Training complete!")
    print(f"Accuracy: {result['accuracy']:.2f}")
    print(f"Samples: {result['n_samples']}")
    print(f"Model saved to: {result['model_path']}")


def main_predict():
    """Command-line interface for prediction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict drunk/sober from image")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Model path")
    
    args = parser.parse_args()
    
    result = predict(args.image_path, args.model_path)
    
    print(f"Image: {args.image_path}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probabilities: Sober={result['probabilities']['sober']:.2%}, "
          f"Drunk={result['probabilities']['drunk']:.2%}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        # Remove 'predict' from arguments and call main_predict
        sys.argv.pop(1)
        main_predict()
    else:
        main_train()
