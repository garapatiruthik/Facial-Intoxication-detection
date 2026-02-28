# INERB - Drunk/Sober Detection

A professional machine learning application that detects whether a person is drunk or sober based on facial features extracted using Mediapipe.

## Project Overview

INERB (INERB is Not Evidence-based Research) uses computer vision and machine learning to analyze facial characteristics and determine if someone has been drinking. The system extracts 5 key facial features and uses a Random Forest classifier for prediction.

## Features

- **Face Mesh Extraction**: Uses Mediapipe for precise facial landmark detection
- **5 Key Features**:
  - Eye Aspect Ratio (EAR) - measures eye openness
  - Cheek Redness - detects facial flushing
  - Mouth Opening - measures jaw relaxation
  - Eye Circularity - analyzes eye shape
  - Lip Line Curvature - measures lip position changes
- **Random Forest Classifier**: Robust machine learning model with 100 estimators
- **Web Interface**: Streamlit-based user interface for easy testing
- **CLI Support**: Command-line tools for training and prediction

## Project Structure

```
inerb/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt         # Python dependencies
├── setup.py                 # Package installation
├── .gitignore               # Git ignore patterns
├── config/
│   └── config.yaml          # Configuration file
├── src/
│   └── inerb/
│       ├── __init__.py      # Package initialization
│       ├── features.py      # Feature extraction module
│       ├── model.py         # Model training and prediction
│       ├── dataset.py       # Dataset handling
│       └── utils.py         # Utility functions
├── models/
│   └── .gitkeep             # Models directory placeholder
├── data/
│   ├── drunk/               # Drunk face images for training
│   └── sober/               # Sober face images for training
├── app/
│   ├── __init__.py
│   └── streamlit_app.py     # Streamlit web interface
├── tests/
│   ├── __init__.py
│   └── test_features.py     # Unit tests
└── notebooks/
    └── training.ipynb       # Training notebook
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Clone or download the project
git clone https://github.com/yourusername/Facial-Intoxication-detection.git
cd Facial-Intoxication-detection

# Install dependencies
pip install -r requirements.txt

# Or install the package in development mode
pip install -e .
```

## Deployment

### Deploy to Streamlit Cloud (Recommended)

Streamlit Cloud is free for public repositories and is the easiest way to deploy your app.

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository: `yourusername/Facial-Intoxication-detection`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
   - Click "Deploy"

3. **Your app will be live at**: `https://your-app-name.streamlit.app`

### Local Deployment

```bash
# Run locally
streamlit run app/streamlit_app.py
```

## Usage

### 1. Prepare Training Data

Create two directories for training data:
- `data/drunk/` - Place images of drunk faces (JPG format)
- `data/sober/` - Place images of sober faces (JPG format)

**Tips for good training data:**
- Use clear, well-lit photos
- Front-facing photos work best
- Include various angles and lighting conditions
- Aim for at least 10+ images per class

### 2. Train the Model

#### Option A: Using Python CLI

```bash
python -m inerb.model
```

#### Option B: Using the Training Notebook

```bash
jupyter notebook notebooks/training.ipynb
```

#### Option C: Using Streamlit

Run the web app - it will prompt you to train if no model exists:

```bash
streamlit run app/streamlit_app.py
```

### 3. Run the Web Application

```bash
streamlit run app/streamlit_app.py
```

This will open a web interface where you can:
- Upload two face images
- See predictions with confidence scores
- View extracted feature values

### 4. Use the Python API

```python
import cv2
from inerb import features, model

# Extract features from an image
img = cv2.imread("face.jpg")
features_list = features.extract_features(img)

# Load model and predict
detector = model.load_model("models/detection_model.pkl")
result = detector.predict_with_details(features_list)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

## Model Information

- **Algorithm**: Random Forest Classifier
- **Number of Trees**: 100
- **Features Used**: 5 facial features
- **Training Data Format**: JPG images in `data/drunk` and `data/sober`

## Configuration

Edit `config/config.yaml` to customize:
- Model parameters
- Dataset paths
- App settings
- Feature names

## Testing

Run the test suite:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

## Disclaimer

This project is for educational and research purposes only. The predictions made by this system should not be used for legal, medical, or safety-critical decisions. Alcohol impairment detection requires professional assessment.
