"""
INEB - Drunk/Sober Detection Web Application
"""

import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

try:
    from inerb import features, model, utils
except ImportError:
    import sys
    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from inerb import features, model, utils

MODEL_PATH = "models/detection_model.pkl"

st.set_page_config(page_title="INEB - Drunk/Sober Detection", page_icon="🍺", layout="wide")

def load_model():
    try:
        return model.load_model(MODEL_PATH)
    except:
        return None

def process_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Could not decode image"
    try:
        extracted_features = features.extract_features(img)
        if extracted_features is None:
            return None, "No face detected"
        detector = load_model()
        if detector is None:
            return None, "Model not found"
        result = detector.predict_with_details(extracted_features)
        feature_details = features.extract_features_with_details(img)
        return {"result": result, "features": feature_details, "image": img}, None
    except Exception as e:
        return None, f"Error: {str(e)}"

def display_prediction_card(result, features_data, image, title="Image"):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader(title)
        st.image(image_rgb, use_container_width=True)
    with col2:
        st.subheader("Prediction Results")
        prediction = result["prediction"]
        confidence = result["confidence"]
        color = "#e74c3c" if prediction == "drunk" else "#2ecc71"
        emoji = "🍺" if prediction == "drunk" else "💚"
        st.markdown(f"<div style='padding:20px;background-color:{color}20;border-radius:10px;border-left:5px solid {color};'><h2 style='color:{color};margin:0;'>{emoji} {prediction.upper()}</h2><p>Confidence: {confidence*100:.1f}%</p></div>", unsafe_allow_html=True)
        probs = result["probabilities"]
        st.markdown("**Sober:**"); st.progress(probs["sober"]); st.markdown(f"_{probs['sober']*100:.1f}%_")
        st.markdown("**Drunk:**"); st.progress(probs["drunk"]); st.markdown(f"_{probs['drunk']*100:.1f}%_")
        if features_data:
            st.markdown("### Features")
            for name, value in features_data.items():
                st.metric(name.replace("_", " ").title(), f"{value:.4f}")

def main():
    st.title("🍺 INEB - Facial Intoxication Detection")
    st.markdown("---")
    st.sidebar.title("About")
    st.sidebar.info("**INEB** detects drunk/sober from facial features.")
    
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found! Please train first.")
        if st.button("Train Model"):
            try:
                with st.spinner("Training..."):
                    result = model.train_model(data_dir="data", model_path=MODEL_PATH)
                st.success(f"Done! Accuracy: {result['accuracy']*100:.1f}%")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
        return
    
    st.header("Upload Face Images")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image 1")
        uploaded_file1 = st.file_uploader("Choose first image", type=['jpg', 'jpeg', 'png'], key="file1")
    with col2:
        st.subheader("Image 2")
        uploaded_file2 = st.file_uploader("Choose second image", type=['jpg', 'jpeg', 'png'], key="file2")
    
    results = []
    if uploaded_file1:
        result1, error1 = process_image(uploaded_file1)
        if error1: st.error(f"Image 1: {error1}")
        elif result1: results.append(("Image 1", result1))
    if uploaded_file2:
        result2, error2 = process_image(uploaded_file2)
        if error2: st.error(f"Image 2: {error2}")
        elif result2: results.append(("Image 2", result2))
    
    if results:
        st.markdown("---")
        st.header("Analysis Results")
        for title, r in results:
            display_prediction_card(r["result"], r["features"], r["image"], title)
            st.markdown("---")
    
    if not uploaded_file1 and not uploaded_file2:
        st.info("👆 Upload images to analyze.")

if __name__ == "__main__":
    main()
