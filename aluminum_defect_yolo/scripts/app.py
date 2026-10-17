import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Aluminum Defect Detection",
    layout="wide"
)

# Header
st.title("Aluminum Surface Defect Classifier")
st.markdown("Upload an image or select a sample from your test dataset to classify defects using YOLOv8.")

# Sidebar - Settings & Model Loader
st.sidebar.header("Configuration")
default_model_path = r"D:\Aluminum_defect\runs\classify\results\run_20260827_001652\weights\best.pt"
model_path = st.sidebar.text_input("Model Weights Path", value=default_model_path)
conf_threshold = st.sidebar.slider("Confidence Display Threshold", 0.0, 1.0, 0.20, 0.05)

@st.cache_resource
def load_yolo_model(path):
    if not Path(path).exists():
        st.error(f"Model file not found at: {path}")
        return None
    return YOLO(path)

model = load_yolo_model(model_path)

# Class mappings fallback
CLASS_NAMES = {
    0: 'Clean sample', 1: 'Be injured by a collision', 2: 'Coating cracking',
    3: 'Convex powder', 4: 'Dirty spot', 5: 'Drain bottom',
    6: 'Non-conducting', 7: 'Orange peel', 8: 'Pitting',
    9: 'Scuffing', 10: 'Transverse strip dented'
}

# Image source selector
input_mode = st.radio("Choose Input Method", ["Upload Image", "Pick from Test Dataset"], horizontal=True)
input_image = None
image_name = ""

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an aluminum sheet image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")
        image_name = uploaded_file.name
else:
    test_dir = Path(r"D:\Aluminum_defect\aluminum_defect_yolo\data_cls\test")
    if test_dir.exists():
        test_images = list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.png"))
        if test_images:
            selected_path = st.selectbox("Select sample test image", test_images, format_func=lambda x: f"{x.parent.name} / {x.name}")
            input_image = Image.open(selected_path).convert("RGB")
            image_name = selected_path.name
        else:
            st.warning("No images found in test directory.")
    else:
        st.warning(f"Test directory not found at: {test_dir}")

# Inference & Display
if input_image and model:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Image")
        st.image(input_image, caption=image_name, use_container_width=True)

    # Run Prediction
    with st.spinner("Analyzing surface..."):
        results = model.predict(source=np.array(input_image), verbose=False)
        result = results[0]

        top1_id = int(result.probs.top1)
        top1_conf = float(result.probs.top1conf.item())
        top1_name = result.names.get(top1_id, CLASS_NAMES.get(top1_id, "Unknown"))

        # Collect top predictions
        prob_data = result.probs.data.cpu().numpy()
        probs_df = pd.DataFrame({
            "Class": [result.names.get(i, CLASS_NAMES.get(i, f"Class {i}")) for i in range(len(prob_data))],
            "Confidence": prob_data
        }).sort_values(by="Confidence", ascending=True)

    with col2:
        st.subheader("Classification Result")
        
        # Result Card
        if "Clean" in top1_name:
            st.success(f"**Status:** {top1_name} ({top1_conf:.1%})")
        else:
            st.error(f"**Defect Detected:** {top1_name} ({top1_conf:.1%})")

        # Confidence Bar Chart
        st.markdown("##### Class Probabilities")
        fig = px.bar(
            probs_df.tail(6), # Top 6 classes
            x="Confidence",
            y="Class",
            orientation="h",
            text=probs_df.tail(6)["Confidence"].apply(lambda x: f"{x:.1%}"),
            color="Confidence",
            color_continuous_scale="Blues"
        )
        fig.update_layout(
            showlegend=False,
            height=320,
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)