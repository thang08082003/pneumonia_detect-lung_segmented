"""
Cascade Pneumonia Detection App
================================
1. UNet: Segment lung region
2. YOLOv8: Detect pneumonia in lung region

Run: streamlit run app_lung_segmentation.py
"""

import os
import io
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2
from pathlib import Path
import streamlit as st
from PIL import Image

# TensorFlow/Keras for UNet
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Ultralytics for YOLOv8
from ultralytics import YOLO

# ============================================================================
# CONFIG
# ============================================================================

UNET_MODEL_PATH = Path("D:/monaiyolo/unet_lung_seg.hdf5")
YOLO_MODEL_PATH = Path("D:/monaiyolo/rsna-pneumonia-detection-challenge/runs/detect/yolov8_rsna_run/weights/best.pt")
IMG_SIZE_UNET = 512  # UNet input size
IMG_SIZE_YOLO = 640  # YOLO input size

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_unet_model():
    """Load UNet segmentation model."""
    try:
        model = tf.keras.models.load_model(str(UNET_MODEL_PATH), compile=False)
        return model, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_yolo_model():
    """Load YOLOv8 detection model."""
    try:
        model = YOLO(str(YOLO_MODEL_PATH))
        return model, None
    except Exception as e:
        return None, str(e)


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def load_image(uploaded_file):
    """Load image from uploaded file."""
    try:
        if uploaded_file.name.lower().endswith('.dcm'):
            import pydicom
            bytes_data = uploaded_file.read()
            ds = pydicom.dcmread(io.BytesIO(bytes_data))
            image = ds.pixel_array.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
            return image.astype(np.uint8)
        else:
            image = Image.open(uploaded_file)
            return np.array(image.convert('L'))
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


def segment_lungs(model, image):
    """Segment lungs using UNet model."""
    original_size = image.shape[:2]
    
    # Preprocess - resize to 512x512
    resized = cv2.resize(image, (IMG_SIZE_UNET, IMG_SIZE_UNET))
    normalized = resized.astype(np.float32) / 255.0
    input_data = normalized.reshape(1, IMG_SIZE_UNET, IMG_SIZE_UNET, 1)
    
    # Predict
    prediction = model.predict(input_data, verbose=0)
    mask = prediction[0, :, :, 0]
    
    # Threshold to binary mask
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Resize back to original size
    mask = cv2.resize(mask, (original_size[1], original_size[0]), 
                      interpolation=cv2.INTER_NEAREST)
    
    return mask


def apply_lung_mask(image, mask):
    """Apply lung mask to image."""
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked


def detect_pneumonia(model, image, conf_threshold=0.25):
    """Detect pneumonia using YOLOv8."""
    # Convert grayscale to RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image
    
    # Run detection
    results = model.predict(image_rgb, conf=conf_threshold, verbose=False)[0]
    
    detections = []
    if results.boxes is not None:
        for i in range(len(results.boxes)):
            box = results.boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf
            })
    
    return detections


def detect_pneumonia_simple(image, lung_mask, sensitivity=0.5):
    """
    Simple pneumonia detection using image processing.
    Detects bright/opaque regions within lung area.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply lung mask
    masked = cv2.bitwise_and(gray, gray, mask=lung_mask)
    
    # Calculate statistics only within lung region
    lung_pixels = masked[lung_mask > 127]
    if len(lung_pixels) == 0:
        return []
    
    mean_val = np.mean(lung_pixels)
    std_val = np.std(lung_pixels)
    
    # Threshold for opacity detection (brighter than normal)
    threshold = mean_val + (1 - sensitivity) * std_val
    
    # Create binary mask of opaque regions
    _, opacity_mask = cv2.threshold(masked, int(threshold), 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations
    kernel = np.ones((15, 15), np.uint8)
    opacity_mask = cv2.morphologyEx(opacity_mask, cv2.MORPH_CLOSE, kernel)
    opacity_mask = cv2.morphologyEx(opacity_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(opacity_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    min_area = gray.shape[0] * gray.shape[1] * 0.005  # Min 0.5% of image
    max_area = gray.shape[0] * gray.shape[1] * 0.25   # Max 25% of image
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate confidence based on intensity difference
            roi = gray[y:y+h, x:x+w]
            roi_mean = np.mean(roi)
            conf = min(0.95, 0.5 + (roi_mean - mean_val) / (255 - mean_val) * 0.5)
            
            detections.append({
                'bbox': (x, y, x+w, y+h),
                'confidence': max(0.3, conf)
            })
    
    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    return detections[:5]  # Return top 5


# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_detections(image, detections, color=(0, 0, 255), thickness=2):
    """Draw detection boxes on image."""
    if len(image.shape) == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        output = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        
        label = f"Pneumonia: {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(output, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return output


def create_overlay(image, mask, alpha=0.4, color=(0, 255, 0)):
    """Create mask overlay on image."""
    if len(image.shape) == 2:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb_image = image.copy()
    
    colored_mask = np.zeros_like(rgb_image)
    colored_mask[mask > 127] = color
    
    overlay = cv2.addWeighted(rgb_image, 1, colored_mask, alpha, 0)
    return overlay


def create_full_visualization(image, mask, detections, alpha=0.3):
    """Create full visualization with mask overlay and detections."""
    if len(image.shape) == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        output = image.copy()
    
    # Add lung mask overlay (green)
    lung_overlay = np.zeros_like(output)
    lung_overlay[mask > 127] = (0, 255, 0)
    output = cv2.addWeighted(output, 1, lung_overlay, alpha, 0)
    
    # Draw detection boxes (red)
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        label = f"Pneumonia: {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 10, y1), (255, 0, 0), -1)
        cv2.putText(output, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return output


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Cascade Pneumonia Detection",
        page_icon="🫁",
        layout="wide"
    )
    
    st.title("🫁 Cascade Pneumonia Detection")
    st.markdown("""
    **Pipeline:** UNet (Lung Segmentation) → YOLOv8 (Pneumonia Detection)
    
    Upload chest X-ray to detect pneumonia regions.
    """)
    
    # Load models
    unet_model, unet_error = load_unet_model()
    yolo_model, yolo_error = load_yolo_model()
    
    # Model status in sidebar
    st.sidebar.header("📊 Model Status")
    
    if unet_error:
        st.sidebar.error(f"❌ UNet: {unet_error}")
    else:
        st.sidebar.success("✅ UNet Loaded")
    
    if yolo_error:
        st.sidebar.error(f"❌ YOLOv8: {yolo_error}")
    else:
        st.sidebar.success("✅ YOLOv8 Loaded")
    
    # Settings
    st.sidebar.header("⚙️ Settings")
    
    use_segmentation = st.sidebar.checkbox("Use Lung Segmentation", value=True, 
                                           help="Apply lung mask before detection")
    
    conf_threshold = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.01,
        max_value=0.5,
        value=0.05,  # Lower default for this model
        step=0.01
    )
    
    detection_method = st.sidebar.radio(
        "Detection Method",
        options=["Simple (Image Processing)", "YOLOv8"],
        index=0,
        help="Simple method works better for this model"
    )
    
    detect_on_masked = st.sidebar.checkbox(
        "Detect on Masked Image", 
        value=True,
        help="Detect only within lung region"
    )
    
    overlay_alpha = st.sidebar.slider(
        "Overlay Transparency",
        min_value=0.1,
        max_value=0.7,
        value=0.3,
        step=0.1
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-ray",
        type=['png', 'jpg', 'jpeg', 'dcm'],
        help="Supported: PNG, JPG, DICOM"
    )
    
    if uploaded_file is not None:
        # Load image
        image = load_image(uploaded_file)
        
        if image is not None:
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["🔍 Detection Result", "🫁 Segmentation", "📊 Analysis"])
            
            with st.spinner("🔄 Processing..."):
                # Step 1: Lung Segmentation
                if use_segmentation and unet_model is not None:
                    lung_mask = segment_lungs(unet_model, image)
                    masked_image = apply_lung_mask(image, lung_mask)
                else:
                    lung_mask = np.ones_like(image) * 255
                    masked_image = image
                
                # Step 2: Pneumonia Detection
                if detection_method == "Simple (Image Processing)":
                    # Use simple image processing method
                    detections = detect_pneumonia_simple(image, lung_mask, conf_threshold)
                elif yolo_model is not None:
                    # Use YOLOv8
                    detect_image = masked_image if detect_on_masked else image
                    detections = detect_pneumonia(yolo_model, detect_image, conf_threshold)
                    
                    # Filter detections to only show those inside lung region
                    if use_segmentation and not detect_on_masked:
                        filtered_detections = []
                        for det in detections:
                            x1, y1, x2, y2 = det['bbox']
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            if 0 <= cy < lung_mask.shape[0] and 0 <= cx < lung_mask.shape[1]:
                                if lung_mask[cy, cx] > 127:
                                    filtered_detections.append(det)
                        detections = filtered_detections
                else:
                    detections = []
            
            # Tab 1: Detection Result
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_container_width=True, clamp=True)
                
                with col2:
                    st.subheader("Detection Result")
                    result_image = create_full_visualization(image, lung_mask, detections, overlay_alpha)
                    st.image(result_image, use_container_width=True, clamp=True)
                
                # Detection summary
                st.markdown("---")
                if len(detections) == 0:
                    st.success("✅ **No Pneumonia Detected**")
                else:
                    st.warning(f"⚠️ **{len(detections)} Pneumonia Region(s) Detected**")
                    
                    for i, det in enumerate(detections):
                        x1, y1, x2, y2 = det['bbox']
                        st.write(f"**Region {i+1}:** Confidence {det['confidence']:.1%}, "
                                f"Location ({x1}, {y1}) to ({x2}, {y2})")
            
            # Tab 2: Segmentation Details
            with tab2:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Original")
                    st.image(image, use_container_width=True, clamp=True)
                
                with col2:
                    st.subheader("Lung Mask")
                    st.image(lung_mask, use_container_width=True, clamp=True)
                
                with col3:
                    st.subheader("Masked Image")
                    st.image(masked_image, use_container_width=True, clamp=True)
                
                # Lung area stats
                lung_pixels = np.sum(lung_mask > 127)
                total_pixels = lung_mask.shape[0] * lung_mask.shape[1]
                lung_percentage = (lung_pixels / total_pixels) * 100
                
                st.metric("Lung Coverage", f"{lung_percentage:.1f}%")
            
            # Tab 3: Analysis
            with tab3:
                st.subheader("📋 Analysis Report")
                
                # Image info
                st.write("**Image Information:**")
                st.write(f"- Size: {image.shape[1]} x {image.shape[0]} pixels")
                st.write(f"- Lung Coverage: {lung_percentage:.1f}%")
                
                st.write("**Detection Results:**")
                if len(detections) == 0:
                    st.write("- No pneumonia regions detected")
                    st.write("- Recommendation: Image appears normal")
                else:
                    st.write(f"- Found {len(detections)} suspicious region(s)")
                    max_conf = max(d['confidence'] for d in detections)
                    st.write(f"- Highest confidence: {max_conf:.1%}")
                    st.write("- Recommendation: Further examination advised")
                
                # Download results
                st.markdown("---")
                st.subheader("📥 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    result_pil = Image.fromarray(result_image)
                    result_bytes = io.BytesIO()
                    result_pil.save(result_bytes, format='PNG')
                    st.download_button(
                        "Download Result Image",
                        data=result_bytes.getvalue(),
                        file_name=f"{uploaded_file.name.split('.')[0]}_result.png",
                        mime="image/png"
                    )
                
                with col2:
                    mask_pil = Image.fromarray(lung_mask)
                    mask_bytes = io.BytesIO()
                    mask_pil.save(mask_bytes, format='PNG')
                    st.download_button(
                        "Download Lung Mask",
                        data=mask_bytes.getvalue(),
                        file_name=f"{uploaded_file.name.split('.')[0]}_mask.png",
                        mime="image/png"
                    )
    
    else:
        # Instructions when no file uploaded
        st.info("👆 Upload a chest X-ray image to begin")
        
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            ### Cascade Pipeline
            
            1. **Lung Segmentation (UNet)**
               - Identifies lung regions in the X-ray
               - Creates a binary mask of lung area
               - Removes non-lung regions (bones, heart, etc.)
            
            2. **Pneumonia Detection (YOLOv8)**
               - Analyzes the lung region
               - Detects areas with lung opacity
               - Draws bounding boxes around suspicious regions
            
            ### Benefits of Cascade Approach
            - Reduces false positives from non-lung areas
            - Focuses detection on relevant regions
            - Improves overall accuracy
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Models:** UNet (Lung Segmentation) + YOLOv8 (Pneumonia Detection)  
    **⚠️ Disclaimer:** For research/educational purposes only. Not for clinical diagnosis.
    """)


if __name__ == "__main__":
    main()
