"""
Cascade Pneumonia Detection - UNet + Mask R-CNN
================================================
1. UNet: Segment lung region
2. Mask R-CNN: Detect pneumonia with segmentation masks

MUST RUN WITH Python 3.9 (venv39):
    .\venv39\Scripts\Activate.ps1
    streamlit run app_maskrcnn_cascade.py
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

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# Disable eager to avoid legacy TF1 graph issues with the UNet model
tf.compat.v1.disable_eager_execution()
from tensorflow.compat.v1.keras import backend as K

# Mask R-CNN
from mrcnn import model as modellib
from mrcnn.config import Config

# ============================================================================
# CONFIG
# ============================================================================

UNET_MODEL_PATH = Path("D:/monaiyolo/unet_lung_seg.hdf5")
MASKRCNN_MODEL_PATH = Path("D:/monaiyolo/mask_rcnn_pneumonia_0015.h5")
IMG_SIZE_UNET = 512
ORIG_SIZE = 1024  # RSNA images are 1024x1024

# ============================================================================
# MASK R-CNN CONFIG
# ============================================================================

class PneumoniaConfig(Config):
    NAME = "pneumonia"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # Background + Pneumonia
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # Scales similar to notebook (larger anchors)
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    DETECTION_MIN_CONFIDENCE = 0.05  # lower default

class InferenceConfig(PneumoniaConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_unet_model():
    """Load UNet segmentation model."""
    try:
        holder = {}

        def build():
            g = tf.Graph()
            with g.as_default():
                sess = tf.compat.v1.Session(graph=g)
                with sess.as_default():
                    m = tf.keras.models.load_model(str(UNET_MODEL_PATH), compile=False)
                    K.set_session(sess)
            holder['graph'] = g
            holder['sess'] = sess
            holder['model'] = m

        build()
        holder['rebuild'] = build  # keep rebuild function
        return holder, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_maskrcnn_model():
    """Load Mask R-CNN detection model with its own graph/session."""
    try:
        holder = {}

        def build():
            g = tf.Graph()
            with g.as_default():
                sess = tf.compat.v1.Session(graph=g)
                with sess.as_default():
                    K.set_session(sess)
                    config = InferenceConfig()
                    m = modellib.MaskRCNN(mode='inference', config=config, model_dir='.')
                    m.load_weights(str(MASKRCNN_MODEL_PATH), by_name=True)
            holder['graph'] = g
            holder['sess'] = sess
            holder['model'] = m

        build()
        holder['rebuild'] = build
        return holder, None
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

def segment_lungs(holder, image):
    """Segment lungs using UNet model. Rebuild session if needed."""
    # If session was closed (e.g., Streamlit rerun), rebuild
    sess = holder.get('sess')
    graph = holder.get('graph')
    model = holder.get('model')
    if sess is None or getattr(sess, '_closed', False):
        holder['rebuild']()
        sess = holder['sess']
        graph = holder['graph']
        model = holder['model']

    original_size = image.shape[:2]
    resized = cv2.resize(image, (IMG_SIZE_UNET, IMG_SIZE_UNET))
    normalized = resized.astype(np.float32) / 255.0
    input_data = normalized.reshape(1, IMG_SIZE_UNET, IMG_SIZE_UNET, 1)
    # Ensure prediction runs inside the captured graph
    with graph.as_default():
        with sess.as_default():
            K.set_session(sess)
            prediction = model.predict(input_data, verbose=0)
    mask = prediction[0, :, :, 0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return mask

def detect_pneumonia_maskrcnn(holder, image, conf_threshold=0.5):
    """Detect pneumonia using Mask R-CNN. Rebuild session if closed."""
    sess = holder.get('sess')
    graph = holder.get('graph')
    model = holder.get('model')
    if sess is None or getattr(sess, '_closed', False):
        holder['rebuild']()
        sess = holder['sess']
        graph = holder['graph']
        model = holder['model']

    original_size = image.shape[:2]

    # Convert to RGB
    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = image

    # Run detection inside graph/session
    with graph.as_default():
        with sess.as_default():
            K.set_session(sess)
            results = model.detect([rgb], verbose=0)[0]
    
    detections = []
    for i in range(len(results['rois'])):
        if results['scores'][i] >= conf_threshold:
            y1, x1, y2, x2 = results['rois'][i]
            
            # ROIs are already in original image coordinates
            x1_orig = int(x1)
            y1_orig = int(y1)
            x2_orig = int(x2)
            y2_orig = int(y2)

            # Clamp to original image size
            x1_orig = max(0, min(original_size[1]-1, x1_orig))
            x2_orig = max(0, min(original_size[1]-1, x2_orig))
            y1_orig = max(0, min(original_size[0]-1, y1_orig))
            y2_orig = max(0, min(original_size[0]-1, y2_orig))

            # Resize mask back
            mask_resized = None
            if results['masks'].size > 0:
                # masks returned by Mask R-CNN are already in original image size
                mask_resized = results['masks'][:, :, i].astype(np.uint8)

            detections.append({
                'bbox': (x1_orig, y1_orig, x2_orig, y2_orig),
                'confidence': float(results['scores'][i]),
                'mask': mask_resized
            })
    
    return detections

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization(image, lung_mask, detections, show_lung_mask=True, alpha=0.4):
    """Create full visualization."""
    if len(image.shape) == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        output = image.copy()
    
    # Draw lung mask overlay (green)
    if show_lung_mask and lung_mask is not None:
        lung_overlay = np.zeros_like(output)
        lung_overlay[lung_mask > 127] = (0, 200, 0)
        output = cv2.addWeighted(output, 1, lung_overlay, alpha * 0.5, 0)
    
    # Draw detection masks and boxes (red)
    for i, det in enumerate(detections):
        # Draw mask
        if det.get('mask') is not None:
            mask_overlay = np.zeros_like(output)
            mask_overlay[det['mask'] > 0] = (255, 50, 50)
            output = cv2.addWeighted(output, 1, mask_overlay, alpha, 0)
        
        # Draw bbox
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        # Label
        label = f"Pneumonia: {det['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 10, y1), (255, 0, 0), -1)
        cv2.putText(output, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return output

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="Cascade Detection (Mask R-CNN)", page_icon="🫁", layout="wide")
    
    st.title("🫁 Cascade Pneumonia Detection")
    st.markdown("**Pipeline:** UNet (Lung Segmentation) → Mask R-CNN (Pneumonia Detection)")
    
    # Load models
    unet_model, unet_error = load_unet_model()
    maskrcnn_model, maskrcnn_error = load_maskrcnn_model()
    
    # Status
    st.sidebar.header("📊 Model Status")
    if unet_error:
        st.sidebar.error(f"❌ UNet: {unet_error}")
    else:
        st.sidebar.success("✅ UNet Loaded")
    
    if maskrcnn_error:
        st.sidebar.error(f"❌ Mask R-CNN: {maskrcnn_error}")
    else:
        st.sidebar.success("✅ Mask R-CNN Loaded")
    
    # Settings
    st.sidebar.header("⚙️ Settings")
    
    use_segmentation = st.sidebar.checkbox("Use Lung Segmentation", value=True)
    
    conf_threshold = st.sidebar.slider("Detection Confidence", 0.05, 0.25, 0.1, 0.01)
    
    show_lung_mask = st.sidebar.checkbox("Show Lung Mask Overlay", value=True)
    
    overlay_alpha = st.sidebar.slider("Overlay Transparency", 0.1, 0.7, 0.4, 0.1)
    
    # Upload
    uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=['png', 'jpg', 'jpeg', 'dcm'])
    
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        
        if image is not None:
            tab1, tab2, tab3 = st.tabs(["🔍 Detection", "🫁 Segmentation", "📊 Report"])
            
            with st.spinner("🔄 Processing..."):
                # Lung segmentation
                if use_segmentation and unet_model:
                    lung_mask = segment_lungs(unet_model, image)
                else:
                    lung_mask = np.ones_like(image) * 255
                
                # Pneumonia detection
                if maskrcnn_model:
                    detections = detect_pneumonia_maskrcnn(maskrcnn_model, image, conf_threshold)
                else:
                    detections = []
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original")
                    st.image(image, clamp=True)
                with col2:
                    st.subheader("Detection Result")
                    result = create_visualization(image, lung_mask, detections, show_lung_mask, overlay_alpha)
                    st.image(result, clamp=True)
                
                st.markdown("---")
                if len(detections) == 0:
                    st.success("✅ No Pneumonia Detected")
                else:
                    st.warning(f"⚠️ {len(detections)} Pneumonia Region(s) Found")
                    for i, det in enumerate(detections):
                        st.write(f"**Region {i+1}:** Confidence {det['confidence']:.1%}")
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Lung Mask")
                    st.image(lung_mask, clamp=True)
                with col2:
                    st.subheader("Masked Image")
                    masked = cv2.bitwise_and(image, image, mask=lung_mask)
                    st.image(masked, clamp=True)
            
            with tab3:
                st.subheader("📋 Analysis Report")
                st.write(f"**Image Size:** {image.shape[1]} x {image.shape[0]}")
                lung_coverage = np.sum(lung_mask > 127) / (lung_mask.shape[0] * lung_mask.shape[1]) * 100
                st.write(f"**Lung Coverage:** {lung_coverage:.1f}%")
                st.write(f"**Detections:** {len(detections)}")
                
                if detections:
                    max_conf = max(d['confidence'] for d in detections)
                    st.write(f"**Max Confidence:** {max_conf:.1%}")
                
                # Download
                st.markdown("---")
                result_pil = Image.fromarray(result)
                result_bytes = io.BytesIO()
                result_pil.save(result_bytes, format='PNG')
                st.download_button("📥 Download Result", result_bytes.getvalue(), 
                                  f"{uploaded_file.name.split('.')[0]}_result.png", "image/png")
    else:
        st.info("👆 Upload a chest X-ray image to begin")
    
    st.markdown("---")
    st.markdown("**Models:** UNet + Mask R-CNN | ⚠️ For research only, not for clinical use")

if __name__ == "__main__":
    main()

