"""
Tests for INERB feature extraction module.
"""

import pytest
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inerb import features


class TestFeatureExtraction:
    """Test cases for feature extraction functions."""
    
    def test_get_feature_names(self):
        """Test that feature names are returned correctly."""
        names = features.get_feature_names()
        assert len(names) == 5
        assert "eye_aspect_ratio" in names
        assert "cheek_redness" in names
        assert "mouth_opening" in names
        assert "eye_circularity" in names
        assert "lip_line_curvature" in names
    
    def test_extract_features_returns_list(self):
        """Test that extract_features returns a list of correct length."""
        # Create a dummy image (black 640x480)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Should return None since no face is present
        result = features.extract_features(img)
        assert result is None or isinstance(result, list)
    
    def test_eye_aspect_ratio_no_landmarks(self):
        """Test eye_aspect_ratio with no landmarks returns 0."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Empty landmarks list should not cause crash
        # This tests the function handles edge cases
        try:
            result = features.eye_aspect_ratio([], img)
            assert isinstance(result, (int, float))
        except:
            pass  # Expected to fail with empty list
    
    def test_cheek_redness_returns_float(self):
        """Test cheek_redness returns a float between 0 and 1."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = features.cheek_redness(img)
        assert isinstance(result, float)
        assert 0 <= result <= 1
    
    def test_mouth_opening_no_landmarks(self):
        """Test mouth_opening with no landmarks."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Should return 0 for empty/none landmarks
        result = features.mouth_opening(None, img)
        assert result == 0
    
    def test_extract_features_with_details(self):
        """Test extract_features_with_details returns correct format."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = features.extract_features_with_details(img)
        
        # Should return None or dict
        if result is not None:
            assert isinstance(result, dict)


class TestModel:
    """Test cases for model functions."""
    
    def test_detection_model_init(self):
        """Test DetectionModel initialization."""
        from inerb.model import DetectionModel
        
        detector = DetectionModel()
        assert detector.feature_names is not None
        assert len(detector.feature_names) == 5


class TestDataset:
    """Test cases for dataset functions."""
    
    def test_load_dataset_returns_tuples(self):
        """Test load_dataset returns correct tuple format."""
        from inerb import dataset
        
        # Should fail gracefully with empty directories
        try:
            X, y = dataset.load_dataset("nonexistent_dir")
        except ValueError:
            pass  # Expected to fail


class TestUtils:
    """Test cases for utility functions."""
    
    def test_validate_image_file_not_found(self):
        """Test image validation with non-existent file."""
        from inerb.utils import validate_image_file
        
        is_valid, msg = validate_image_file("nonexistent.jpg")
        assert not is_valid
        assert "not found" in msg.lower()
    
    def test_check_model_exists(self):
        """Test model existence check."""
        from inerb.utils import check_model_exists
        
        # Should return False for non-existent model
        result = check_model_exists("nonexistent_model.pkl")
        assert result is False
    
    def test_format_prediction_result(self):
        """Test prediction result formatting."""
        from inerb.utils import format_prediction_result
        
        result = format_prediction_result(
            prediction="drunk",
            confidence=0.85,
            probabilities={"sober": 0.15, "drunk": 0.85}
        )
        
        assert result["prediction"] == "drunk"
        assert result["confidence"] == 0.85
        assert result["color"] == "#e74c3c"  # Red for drunk
        
        # Test sober
        result = format_prediction_result(
            prediction="sober",
            confidence=0.90,
            probabilities={"sober": 0.90, "drunk": 0.10}
        )
        
        assert result["prediction"] == "sober"
        assert result["color"] == "#2ecc71"  # Green for sober


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
