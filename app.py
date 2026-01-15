import streamlit as st
import tempfile
import cv2
import os
import numpy as np  # Fixed: Added missing import
from PIL import Image
from yolov8_detector import YOLOv8Detector

# --- Configuration ---
# Ensure this file exists in your project directory
MODEL_WEIGHTS = 'yolov8n.pt' 

# --- Initialize Detector (Caching for performance) ---
@st.cache_resource
def load_detector():
    """Load the YOLOv8 model once and cache it."""
    if not os.path.exists(MODEL_WEIGHTS):
        return None
    detector = YOLOv8Detector(model_path=MODEL_WEIGHTS)
    return detector

detector = load_detector()

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Community Vision: AI Hazard Detection Prototype",
    layout="wide"
)

st.title("🚦 Community Vision: AI-Powered Hazard Detection")
st.markdown("Prototype application for real-time urban pedestrian safety monitoring using **YOLOv8**.")

# File Uploader
st.sidebar.header("Upload File")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image or video file...",
    type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov']
)

# Confidence Slider
st.sidebar.header("Model Settings")
confidence = st.sidebar.slider(
    'Confidence Threshold', 0.0, 1.0, 0.25, 0.05
)

if uploaded_file is not None and detector is not None:
    file_type = uploaded_file.type.split('/')[0]

    st.subheader(f"Processing Uploaded {file_type.capitalize()}")
    
    col1, col2 = st.columns(2)
    
    # --- Image Processing ---
    if file_type == 'image':
        original_image = Image.open(uploaded_file)
        with col1:
            st.markdown("**Original Image**")
            st.image(original_image, caption="Uploaded Image", use_container_width=True)

        try:
            # Convert PIL image to OpenCV format
            frame = np.array(original_image.convert('RGB'))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Process with detection
            detected_frame = detector.detect_and_draw(frame, conf=confidence)
            
            # Convert back to RGB for Streamlit
            detected_image = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.markdown("**AI Hazard Detection Result**")
                st.image(detected_image, caption="Detected Hazards", use_container_width=True)
                
            st.success("Image processing complete!")
            
        except Exception as e:
            st.error(f"Error processing image: {e}")

    # --- Video Processing ---
    elif file_type == 'video':
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name

        st.info("Processing video. This may take a moment...")
        video_placeholder = st.empty()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_file_name = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_file_name, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            detected_frame = detector.detect_and_draw(frame, conf=confidence)
            out.write(detected_frame)

            if frame_count % max(1, int(fps/2)) == 0: 
                cv_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(cv_rgb, caption="Processing Video...", use_container_width=True)
            
            frame_count += 1

        cap.release()
        out.release()
        os.unlink(video_path)
        
        st.success("Video processing complete!")

        with open(output_file_name, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name=output_file_name,
                mime="video/mp4"
            )

        if os.path.exists(output_file_name):
            os.unlink(output_file_name)

elif detector is None:
    st.error(f"🚨 **Error:** The model file '{MODEL_WEIGHTS}' was not found.")
else:
    st.info("👆 Please upload an image or video file to begin.")
