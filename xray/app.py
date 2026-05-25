"""
Cascade Pneumonia Detection - Flask
===================================
UNet + Mask R-CNN API for Docker Space
"""

import os
import io
import base64
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import threading

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.disable_eager_execution()
from tensorflow.compat.v1.keras import backend as K

# Mask R-CNN
from mrcnn import model as modellib
from mrcnn.config import Config

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile apps
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Thread lock để tránh TensorFlow issues
model_lock = threading.Lock()

# ============================================================================
# CONFIG
# ============================================================================

UNET_MODEL_PATH = Path("unet_lung_seg.hdf5")
MASKRCNN_MODEL_PATH = Path("mask_rcnn_pneumonia_0015.h5")

if not UNET_MODEL_PATH.exists():
    UNET_MODEL_PATH = Path("models/unet_lung_seg.hdf5")
if not MASKRCNN_MODEL_PATH.exists():
    MASKRCNN_MODEL_PATH = Path("models/mask_rcnn_pneumonia_0015.h5")

IMG_SIZE_UNET = 512
ORIG_SIZE = 1024

# ============================================================================
# MASK R-CNN CONFIG
# ============================================================================

class PneumoniaConfig(Config):
    NAME = "pneumonia"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    DETECTION_MIN_CONFIDENCE = 0.05

class InferenceConfig(PneumoniaConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# ============================================================================
# MODEL LOADING
# ============================================================================

unet_model_holder = None
maskrcnn_model_holder = None

def load_unet_model():
    """Load UNet segmentation model."""
    global unet_model_holder
    if unet_model_holder is not None:
        return unet_model_holder, None
    
    try:
        if not UNET_MODEL_PATH.exists():
            return None, f"Model file not found: {UNET_MODEL_PATH}"
        
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
        holder['rebuild'] = build
        unet_model_holder = holder
        return holder, None
    except Exception as e:
        return None, str(e)

def load_maskrcnn_model():
    """Load Mask R-CNN detection model."""
    global maskrcnn_model_holder
    if maskrcnn_model_holder is not None:
        return maskrcnn_model_holder, None
    
    try:
        if not MASKRCNN_MODEL_PATH.exists():
            return None, f"Model file not found: {MASKRCNN_MODEL_PATH}"
        
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
        maskrcnn_model_holder = holder
        return holder, None
    except Exception as e:
        return None, str(e)

# Load models immediately
print("Loading models...")
unet_model, unet_error = load_unet_model()
maskrcnn_model, maskrcnn_error = load_maskrcnn_model()

if unet_error:
    print(f"UNet error: {unet_error}")
else:
    print("✅ UNet loaded")

if maskrcnn_error:
    print(f"Mask R-CNN error: {maskrcnn_error}")
else:
    print("✅ Mask R-CNN loaded")
print("Models loaded!")

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def load_image_from_bytes(file_bytes, filename):
    """Load image from bytes."""
    try:
        if filename.lower().endswith('.dcm'):
            import pydicom
            ds = pydicom.dcmread(io.BytesIO(file_bytes))
            image = ds.pixel_array.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
            return image.astype(np.uint8)
        else:
            image = Image.open(io.BytesIO(file_bytes))
            return np.array(image.convert('L'))
    except Exception as e:
        return None

def load_image_from_base64(base64_str):
    """Load image from base64 string."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
        return np.array(image.convert('L'))
    except Exception as e:
        return None

def segment_lungs(holder, image):
    """Segment lungs using UNet model."""
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
    
    with graph.as_default():
        with sess.as_default():
            K.set_session(sess)
            prediction = model.predict(input_data, verbose=0)
    
    mask = prediction[0, :, :, 0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return mask

def detect_pneumonia_maskrcnn(holder, image, conf_threshold=0.5):
    """Detect pneumonia using Mask R-CNN."""
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

    # Run detection inside graph/session (không resize, detect trực tiếp như bản gốc)
    with graph.as_default():
        with sess.as_default():
            K.set_session(sess)
            results = model.detect([rgb], verbose=0)[0]
    
    # DEBUG: log raw scores and rois
    try:
        print("[DEBUG] Mask R-CNN raw scores:", results.get('scores'))
        print("[DEBUG] Mask R-CNN raw rois:", results.get('rois'))
    except Exception as _:
        pass
    
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
    
    # DEBUG: log filtered detections count
    try:
        print(f"[DEBUG] Filtered detections (conf_threshold={conf_threshold}): {len(detections)}")
    except Exception as _:
        pass
    
    return detections

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
    
    # Draw detection masks and boxes (Pneumonia = red)
    for i, det in enumerate(detections):
        # Draw mask
        if det.get('mask') is not None:
            mask_overlay = np.zeros_like(output)
            # Pure red mask for pneumonia regions (OpenCV dùng BGR, nên đỏ là (0, 0, 255))
            mask_overlay[det['mask'] > 0] = (0, 0, 255)
            # Tăng alpha nhẹ để mask nổi bật hơn
            output = cv2.addWeighted(output, 1, mask_overlay, min(0.9, alpha + 0.2), 0)
        
        # Draw bbox (đỏ đậm) - BGR: (0, 0, 255)
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Label
        label = f"Pneumonia: {det['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 0, 255), -1)
        cv2.putText(output, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return output

def image_to_base64(image):
    """Convert numpy image to base64 string."""
    if len(image.shape) == 3:
        # Convert BGR to RGB if needed
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
    else:
        image_rgb = image
    
    pil_image = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# ============================================================================
# HTML TEMPLATE
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🫁 Cascade Pneumonia Detection</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #2563eb; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
        .upload-area:hover { border-color: #2563eb; }
        input[type="file"] { margin: 10px; }
        button { background: #2563eb; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #1d4ed8; }
        .result { margin-top: 20px; }
        .image-container { display: flex; gap: 20px; margin: 20px 0; }
        .image-box { flex: 1; }
        .image-box img { max-width: 100%; border: 1px solid #ccc; border-radius: 5px; }
        .info { background: #f3f4f6; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .loading { display: none; text-align: center; }
    </style>
</head>
<body>
    <h1>🫁 Cascade Pneumonia Detection</h1>
    <p><strong>Pipeline:</strong> UNet (Lung Segmentation) → Mask R-CNN (Pneumonia Detection)</p>
    
    <div class="upload-area">
        <h3>📤 Upload Chest X-ray Image</h3>
        <input type="file" id="imageInput" accept="image/*,.dcm">
        <br>
        <label>Confidence Threshold: <input type="range" id="confThreshold" min="0.05" max="0.25" step="0.01" value="0.1"></label>
        <span id="confValue">0.1</span>
        <br><br>
        <button onclick="processImage()">🔍 Detect Pneumonia</button>
    </div>
    
    <div class="loading" id="loading">
        <p>🔄 Processing...</p>
    </div>
    
    <div class="result" id="result"></div>
    
    <script>
        document.getElementById('confThreshold').addEventListener('input', function(e) {
            document.getElementById('confValue').textContent = e.target.value;
        });
        
        async function processImage() {
            const fileInput = document.getElementById('imageInput');
            const confThreshold = document.getElementById('confThreshold').value;
            const resultDiv = document.getElementById('result');
            const loadingDiv = document.getElementById('loading');
            
            if (!fileInput.files[0]) {
                alert('Please select an image file');
                return;
            }
            
            loadingDiv.style.display = 'block';
            resultDiv.innerHTML = '';
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('conf_threshold', confThreshold);
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    resultDiv.innerHTML = `
                        <div class="image-container">
                            <div class="image-box">
                                <h3>Original Image</h3>
                                <img src="data:image/png;base64,${data.original_image}" alt="Original">
                            </div>
                            <div class="image-box">
                                <h3>Detection Result</h3>
                                <img src="data:image/png;base64,${data.result_image}" alt="Result">
                            </div>
                        </div>
                        <div class="info">
                            <h3>📊 Analysis Report</h3>
                            <p><strong>Detections:</strong> ${data.num_detections}</p>
                            <p><strong>Image Size:</strong> ${data.image_size[0]} x ${data.image_size[1]}</p>
                            ${data.detections.length > 0 ? '<p><strong>Regions Found:</strong></p><ul>' + data.detections.map((d, i) => `<li>Region ${i+1}: Confidence ${(d.confidence * 100).toFixed(1)}%</li>`).join('') + '</ul>' : '<p>✅ No Pneumonia Detected</p>'}
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `<div class="info" style="background: #fee; color: #c00;"><strong>Error:</strong> ${data.error || data.message || 'Unknown error'}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="info" style="background: #fee; color: #c00;"><strong>Error:</strong> ${error.message}</div>`;
            } finally {
                loadingDiv.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route("/")
def index():
    """Serve HTML frontend."""
    return render_template_string(HTML_TEMPLATE)

@app.route("/detect", methods=["POST"])
def detect_pneumonia():
    """
    Detect pneumonia in uploaded chest X-ray.
    Supports both multipart/form-data (file upload) and JSON (base64 image).
    """
    global unet_model, maskrcnn_model
    
    try:
        # Check if models are loaded
        if maskrcnn_model is None:
            return jsonify({
                'success': False,
                'error': 'Mask R-CNN model not loaded'
            }), 500
        
        # Parse request parameters
        if request.is_json:
            # JSON request with base64 image
            data = request.get_json()
            image_base64 = data.get('image_base64') or data.get('image')
            if not image_base64:
                return jsonify({
                    'success': False,
                    'error': 'No image_base64 provided'
                }), 400
            
            image = load_image_from_base64(image_base64)
            if image is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to decode base64 image'
                }), 400
            
            confidence = float(data.get('confidence', data.get('conf_threshold', 0.1)))
            use_segmentation = data.get('use_segmentation', True)
            show_lung_mask = data.get('show_lung_mask', True)
            overlay_alpha = float(data.get('overlay_alpha', 0.4))
            return_format = data.get('return_format', 'json')
        else:
            # Multipart form-data with file upload
            if 'file' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No file provided'
                }), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
            
            file_bytes = file.read()
            image = load_image_from_bytes(file_bytes, file.filename)
            if image is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to load image'
                }), 400
            
            confidence = float(request.form.get('conf_threshold', request.form.get('confidence', 0.1)))
            use_segmentation = request.form.get('use_segmentation', 'true').lower() == 'true'
            show_lung_mask = request.form.get('show_lung_mask', 'true').lower() == 'true'
            overlay_alpha = float(request.form.get('overlay_alpha', 0.4))
            return_format = request.form.get('return_format', 'json')
        
        # Validate parameters
        confidence = max(0.05, min(1.0, confidence))
        overlay_alpha = max(0.1, min(0.7, overlay_alpha))
        
        # DEBUG: log request parameters
        try:
            print(f"[DEBUG] /detect params: confidence={confidence}, use_segmentation={use_segmentation}, show_lung_mask={show_lung_mask}, overlay_alpha={overlay_alpha}, return_format={return_format}")
            print(f"[DEBUG] /detect image shape: {image.shape}")
        except Exception as _:
            pass
        
        # Process with thread lock to avoid TensorFlow issues
        with model_lock:
            # Load models if needed
            unet_model, unet_error = load_unet_model()
            maskrcnn_model, maskrcnn_error = load_maskrcnn_model()
            
            if unet_error:
                return jsonify({
                    'success': False,
                    'error': f'UNet error: {unet_error}'
                }), 500
            if maskrcnn_error:
                return jsonify({
                    'success': False,
                    'error': f'Mask R-CNN error: {maskrcnn_error}'
                }), 500
            
            # Lung segmentation
            if use_segmentation and unet_model is not None:
                lung_mask = segment_lungs(unet_model, image)
            else:
                lung_mask = np.ones_like(image) * 255
            
            # Pneumonia detection
            detections = detect_pneumonia_maskrcnn(maskrcnn_model, image, confidence)
        
        # DEBUG: log detections summary
        try:
            print(f"[DEBUG] /detect total detections: {len(detections)}")
        except Exception as _:
            pass
        
        # Create visualizations
        # 1) Ảnh chỉ có lung mask (không có bbox/mask pneumonia)
        lung_only_image = create_visualization(image, lung_mask, [], show_lung_mask=True, alpha=overlay_alpha)
        # 2) Ảnh chỉ có detection (bbox + mask pneumonia, không overlay lung mask)
        detection_only_image = create_visualization(image, lung_mask, detections, show_lung_mask=False, alpha=overlay_alpha)
        # 3) Ảnh tổng hợp như hiện tại (lung mask + detection)
        result_image = create_visualization(image, lung_mask, detections, show_lung_mask, overlay_alpha)
        
        # Calculate lung coverage
        lung_coverage = np.sum(lung_mask > 127) / (lung_mask.shape[0] * lung_mask.shape[1]) * 100
        
        # Prepare detections for JSON (remove mask arrays)
        detections_json = []
        for det in detections:
            detections_json.append({
                'bbox': list(det['bbox']),
                'confidence': float(det['confidence'])
            })
        
        # Return response
        if return_format == 'image':
            # Return image directly
            result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            result_pil.save(buffer, format='PNG')
            buffer.seek(0)
            return send_file(buffer, mimetype='image/png')
        else:
            # Return JSON with base64 images
            original_b64 = image_to_base64(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image)
            result_b64 = image_to_base64(result_image)
            lung_only_b64 = image_to_base64(lung_only_image)
            detection_only_b64 = image_to_base64(detection_only_image)
            
            return jsonify({
                'success': True,
                'detections': detections_json,
                'num_detections': len(detections),
                # Ảnh tổng hợp (lung mask + detection)
                'result_image': result_b64,
                'result_image_base64': result_b64,  # Alias for compatibility
                # Ảnh chỉ lung mask
                'lung_mask_image': lung_only_b64,
                'lung_mask_image_base64': lung_only_b64,
                # Ảnh chỉ detection (bbox + mask pneumonia)
                'detection_image': detection_only_b64,
                'detection_image_base64': detection_only_b64,
                # Ảnh gốc
                'original_image': original_b64,
                'original_image_base64': original_b64,  # Alias for compatibility
                'lung_coverage': round(float(lung_coverage), 2),
                'image_size': {
                    'width': int(image.shape[1]),
                    'height': int(image.shape[0])
                }
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'unet_loaded': unet_model is not None,
        'maskrcnn_loaded': maskrcnn_model is not None
    })

@app.route("/models/info", methods=["GET"])
def models_info():
    """Get information about loaded models."""
    return jsonify({
        'models': {
            'unet': {
                'loaded': unet_model is not None,
                'path': str(UNET_MODEL_PATH),
                'purpose': 'Lung Segmentation'
            },
            'maskrcnn': {
                'loaded': maskrcnn_model is not None,
                'path': str(MASKRCNN_MODEL_PATH),
                'purpose': 'Pneumonia Detection'
            }
        },
        'supported_formats': ['png', 'jpg', 'jpeg', 'dcm'],
        'default_settings': {
            'confidence': 0.1,
            'use_segmentation': True,
            'show_lung_mask': True,
            'overlay_alpha': 0.4
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
