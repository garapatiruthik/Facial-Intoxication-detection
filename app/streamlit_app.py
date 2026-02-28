"""
INEB - Drunk/Sober Detection Web Application

A Streamlit web interface for detecting whether a person is drunk or sober
based on facial features extracted using Mediapipe.
"""

import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Try to import from package, fall back to src path for development
try:
    from inerb import features, model, utils
except ImportError:
    import sys
    # Add src to path for development
    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from inerb import features, model, utils


# Configuration
MODEL_PATH = "models/detection_model.pkl"
DEFAULT_MODEL_PATH = "models/detection_model.pkl"

# Page configuration
st.set_page_config(
    page_title="INEB - Drunk/Sober Detection",
    page_icon="🍺",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_model():
    """Load the trained model."""
    try:
        return model.load_model(MODEL_PATH)
    except Exception as e:
        return None


def process_image(uploaded_file):
    """Process uploaded image and return prediction result."""
    # Read image from uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, "Could not decode image"
    
    # Extract features
    try:
        extracted_features = features.extract_features(img)
        
        if extracted_features is None:
            return None, "No face detected in the image. Please upload a clear face photo."
        
        # Load model and make prediction
        detector = load_model()
        
        if detector is None:
            return None, "Model not found. Please train the model first."
        
        result = detector.predict_with_details(extracted_features)
        
        # Also get feature details for display
        feature_details = features.extract_features_with_details(img)
        
        return {
            "result": result,
            "features": feature_details,
            "image": img,
        }, None
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"


def display_prediction_card(result, features_data, image, title="Image"):
    """Display prediction result in a styled card."""
    
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(title)
        st.image(image_rgb, use_container_width=True)
    
    with col2:
        st.subheader("Prediction Results")
        
        # Display prediction with color
        prediction = result["prediction"]
        confidence = result["confidence"]
        color = "#e74c3c" if prediction == "drunk" else "#2ecc71"
        emoji = "🍺" if prediction == "drunk" else "💚"
        
        st.markdown(
            f"""
            <div style="padding: 20px; background-color: {color}20; border-radius: 10px; border-left: 5px solid {color};">
                <h2 style="color: {color}; margin: 0;">{emoji} {prediction.upper()}</h2>
                <p style="font-size: 18px; margin: 10px 0 0 0;">Confidence: {confidence*100:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display probability bars
        st.markdown("### Probability Breakdown")
        
        probs = result["probabilities"]
        
        # Sober probability
        st.markdown("**Sober**:")
        st.progress(probs["sober"])
        st.markdown(f"_{probs['sober']*100:.1f}%_")
        
        # Drunk probability
        st.markdown("**Drunk**:")
        st.progress(probs["drunk"])
        st.markdown(f"_{probs['drunk']*100:.1f}%_")
        
        # Display extracted features
        if features_data:
            st.markdown("### Extracted Features")
            feature_cols = st.columns(2)
            
            for idx, (name, value) in enumerate(features_data.items()):
                with feature_cols[idx % 2]:
                    st.metric(
                        label=name.replace("_", " ").title(),
                        value=f"{value:.4f}"
                    )


def main():
    """Main application function."""
    
    # Header
    st.title("🍺 INEB - Facial Intoxication (Drunk/Sober) Detection")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        """
        **INEB** is a machine learning system that detects whether a person 
        is drunk or sober based on facial features.
        
        ### Features Used:
        - Eye Aspect Ratio
        - Cheek Redness
        - Mouth Opening
        - Eye Circularity
        - Lip Line Curvature
        
        ### Model:
        Random Forest Classifier with 100 estimators
        """
    )
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"""
            **Model not found!** 
            
            Please train the model first by running:
            ```
            python -m inerb.model
            ```
            
            Or use the training notebook in `notebooks/training.ipynb`
            """
        )
        
        # Show training button
        if st.button("Train Model Now"):
            try:
                with st.spinner("Training model... This may take a while..."):
                    result = model.train_model(
                        data_dir="data",
                        model_path=MODEL_PATH,
                        n_estimators=100
                    )
                st.success(f"Model trained successfully! Accuracy: {result['accuracy']*100:.1f}%")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                st.info("Please add images to data/drunk and data/sober directories first.")
        
        return
    
    # Main content
    st.header("Upload Face Images")
    st.markdown("Upload two images of the same person to analyze their state.")
    
    # File uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Image 1")
        uploaded_file1 = st.file_uploader(
            "Choose first image",
            type=['jpg', 'jpeg', 'png'],
            key="file1"
        )
    
    with col2:
        st.subheader("Image 2")
        uploaded_file2 = st.file_uploader(
            "Choose second image",
            type=['jpg', 'jpeg', 'png'],
            key="file2"
        )
    
    # Process images
    results = []
    
    if uploaded_file1 is not None:
        result1, error1 = process_image(uploaded_file1)
        
        if error1:
            st.error(f"Image 1: {error1}")
        elif result1:
            results.append(("Image 1", result1, result1["features"], result1["image"]))
    
    if uploaded_file2 is not None:
        result2, error2 = process_image(uploaded_file2)
        
        if error2:
            st.error(f"Image 2: {error2}")
        elif result2:
            results.append(("Image 2", result2, result2["features"], result2["image"]))
    
    # Display results
    if results:
        st.markdown("---")
        st.header("Analysis Results")
        
        for title, result, features_data, img in results:
            display_prediction_card(result["result"], features_data, img, title)
            st.markdown("---")
        
        # Summary if both images analyzed
        if len(results) == 2:
            st.header("Summary")
            
            predictions = [r[1]["result"]["prediction"] for r in results]
            confidences = [r[1]["result"]["confidence"] for r in results]
            
            # Determine overall assessment
            drunk_count = predictions.count("drunk")
            sober_count = predictions.count("sober")
            
            if drunk_count > sober_count:
                overall = "Likely Drunk"
                color = "#e74c3c"
            elif sober_count > drunk_count:
                overall = "Likely Sober"
                color = "#2ecc71"
            else:
                overall = "Inconclusive"
                color = "#f39c12"
            
            avg_confidence = sum(confidences) / len(confidences)
            
            st.markdown(
                f"""
                <div style="padding: 20px; background-color: {color}20; border-radius: 10px; text-align: center;">
                    <h2 style="color: {color}; margin: 0;">{overall}</h2>
                    <p>Average Confidence: {avg_confidence*100:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Instructions when no images uploaded
    if not uploaded_file1 and not uploaded_file2:
        st.info("👆 Please upload images to start the analysis.")
        
        st.markdown("""
        ### How to Use:
        1. Upload a clear photo of a face (preferably frontal view)
        2. The system will extract facial features using Mediapipe
        3. The trained model will predict if the person appears drunk or sober
        4. Upload two images to compare both analyses
        
        ### Tips for Best Results:
        - Use clear, well-lit photos
        - Face should be clearly visible
        - Front-facing photos work best
        - Avoid blurry or low-resolution images
        """)


if __name__ == "__main__":
    main()
