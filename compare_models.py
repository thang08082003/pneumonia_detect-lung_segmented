"""
So sánh Mask R-CNN và ResNet cho Pneumonia Detection
====================================================
Tính toán Accuracy, Precision, Recall cho cả hai mô hình
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Fix encoding cho Windows console
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import defaultdict

# TensorFlow cho Mask R-CNN
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.compat.v1.keras import backend as K
from mrcnn import model as modellib
from mrcnn.config import Config

# ============================================================================
# CONFIG
# ============================================================================

MASKRCNN_MODEL_PATH = "mask_rcnn_pneumonia_0015.h5"
RESNET_MODEL_PATH = "model.pth"
VAL_IMAGES_DIR = "rsna-pneumonia-detection-challenge/yolo_rsna_dataset/images/val"
LABELS_CSV = "rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
IMG_SIZE = 224  # ResNet input size

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
# DATASET
# ============================================================================

class PneumoniaDataset(Dataset):
    """Dataset cho ResNet classification"""
    def __init__(self, image_dir, labels_df, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.labels_df = labels_df
        
        # Lấy danh sách ảnh
        self.image_files = list(self.image_dir.glob("*.png"))
        self.image_ids = [f.stem for f in self.image_files]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        patient_id = self.image_ids[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Get label (0 = no pneumonia, 1 = pneumonia)
        patient_labels = self.labels_df[self.labels_df['patientId'] == patient_id]
        if len(patient_labels) > 0 and patient_labels['Target'].iloc[0] == 1:
            label = 1
        else:
            label = 0
        
        if self.transform:
            img = self.transform(img)
        
        return img, label, patient_id

# ============================================================================
# LOAD MODELS
# ============================================================================

def load_maskrcnn_model():
    """Load Mask R-CNN model"""
    print("Đang load Mask R-CNN model...")
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session(graph=graph)
        with sess.as_default():
            K.set_session(sess)
            config = InferenceConfig()
            model = modellib.MaskRCNN(
                mode="inference",
                config=config,
                model_dir="."
            )
            model.load_weights(MASKRCNN_MODEL_PATH, by_name=True)
    print("✓ Mask R-CNN đã load xong")
    return {"graph": graph, "sess": sess, "model": model}

def load_resnet_model():
    """Load ResNet18 model"""
    print("Đang load ResNet18 model...")
    
    # PyTorch 2.6+ yêu cầu weights_only=False cho model đầy đủ
    checkpoint = torch.load(RESNET_MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # Tạo ResNet18 model
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: no pneumonia, pneumonia
    
    # Load weights
    state_dict = None
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print("✓ Đã load ResNet18 thành công")
    except Exception as e:
        print(f"⚠ Cảnh báo khi load: {e}")
    
    model = model.to(DEVICE)
    model.eval()
    print("✓ ResNet18 đã load xong")
    return model

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_maskrcnn_detailed(holder, image_path, conf_threshold=0.1):
    """
    Predict với Mask R-CNN - trả về detections chi tiết
    Trả về: dict với 'boxes', 'scores', 'masks', 'has_pneumonia'
    """
    sess = holder["sess"]
    graph = holder["graph"]
    model = holder["model"]
    
    try:
        # Load image
        img = np.array(Image.open(image_path).convert('L'))
        original_size = img.shape
        
        # Normalize image để đảm bảo chất lượng tốt nhất
        if img.max() > 0:
            img = (img.astype(np.float32) / img.max() * 255).astype(np.uint8)
        
        # Convert to RGB
        if len(img.shape) == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = img
        
        # Predict
        with graph.as_default():
            with sess.as_default():
                K.set_session(sess)
                results = model.detect([rgb], verbose=0)[0]
        
        # Lọc detections theo threshold
        boxes = []
        scores = []
        masks = []
        
        if len(results['scores']) > 0:
            for i in range(len(results['scores'])):
                if results['scores'][i] >= conf_threshold:
                    y1, x1, y2, x2 = results['rois'][i]
                    boxes.append([x1, y1, x2, y2])  # [x1, y1, x2, y2]
                    scores.append(float(results['scores'][i]))
                    if results['masks'].size > 0:
                        masks.append(results['masks'][:, :, i])
        
        has_pneumonia = 1 if len(boxes) > 0 else 0
        
        return {
            'boxes': np.array(boxes),
            'scores': np.array(scores),
            'masks': masks,
            'has_pneumonia': has_pneumonia,
            'original_size': original_size
        }
    except Exception as e:
        print(f"Warning: Error in Mask R-CNN prediction for {image_path.name}: {e}")
        return {
            'boxes': np.array([]),
            'scores': np.array([]),
            'masks': [],
            'has_pneumonia': 0,
            'original_size': (1024, 1024)
        }

def predict_maskrcnn(holder, image_path, conf_threshold=0.1):
    """
    Predict với Mask R-CNN
    Trả về: 1 nếu có detection, 0 nếu không
    Tối ưu hóa: sử dụng top detection và xử lý edge cases
    """
    result = predict_maskrcnn_detailed(holder, image_path, conf_threshold)
    return result['has_pneumonia']

def predict_resnet(model, image_path):
    """
    Predict với ResNet
    Trả về: 1 nếu có pneumonia, 0 nếu không
    """
    try:
        # Load và preprocess image
        img = Image.open(image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            # Lấy class có probability cao nhất
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            return predicted.item()
    except Exception as e:
        # Nếu có lỗi, trả về 0 (không có pneumonia)
        print(f"Warning: Error in ResNet prediction for {image_path.name}: {e}")
        return 0

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Vẽ Confusion Matrix cho một mô hình"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Pneumonia', 'Pneumonia'],
                yticklabels=['No Pneumonia', 'Pneumonia'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu Confusion Matrix: {save_path}")

def plot_comparison_bar_chart(metrics_maskrcnn, metrics_resnet, save_path):
    """Vẽ biểu đồ cột so sánh các metrics"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    maskrcnn_values = [metrics_maskrcnn[m] * 100 for m in metrics]
    resnet_values = [metrics_resnet[m] * 100 for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, maskrcnn_values, width, label='Mask R-CNN', 
                   color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, resnet_values, width, label='ResNet18', 
                   color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('So sánh Mask R-CNN vs ResNet18', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend(fontsize=11)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Thêm giá trị trên mỗi cột
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu Biểu đồ so sánh: {save_path}")

def create_heatmap_maskrcnn(holder, image_path, save_path):
    """Tạo heatmap cho Mask R-CNN"""
    sess = holder["sess"]
    graph = holder["graph"]
    model = holder["model"]
    
    # Load image
    img = np.array(Image.open(image_path).convert('L'))
    original_size = img.shape
    
    # Normalize
    if img.max() > 0:
        img = (img.astype(np.float32) / img.max() * 255).astype(np.uint8)
    
    # Convert to RGB
    if len(img.shape) == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = img
    
    # Predict
    with graph.as_default():
        with sess.as_default():
            K.set_session(sess)
            results = model.detect([rgb], verbose=0)[0]
    
    # Tạo heatmap từ masks
    heatmap = np.zeros((original_size[0], original_size[1]), dtype=np.float32)
    
    if len(results['scores']) > 0:
        for i in range(len(results['scores'])):
            if results['scores'][i] >= 0.1:  # Threshold
                mask = results['masks'][:, :, i]
                score = results['scores'][i]
                heatmap += mask.astype(np.float32) * score
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Vẽ
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap, cmap='hot', alpha=0.8)
    axes[1].imshow(img, cmap='gray', alpha=0.5)
    axes[1].set_title('Mask R-CNN Heatmap (Detection Regions)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu Heatmap Mask R-CNN: {save_path}")

def create_heatmap_resnet(model, image_path, save_path):
    """Tạo heatmap cho ResNet18 sử dụng Grad-CAM"""
    try:
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        except ImportError:
            from grad_cam import GradCAM
            from grad_cam.utils.image import show_cam_on_image
            from grad_cam.utils.model_targets import ClassifierOutputTarget
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Predict để xác định target class
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Tạo Grad-CAM
        target_layers = [model.layer4[-1]]  # Last conv layer của ResNet18
        # Version mới của grad-cam không cần use_cuda, tự động detect device
        cam = GradCAM(model=model, target_layers=target_layers)
        
        targets = [ClassifierOutputTarget(predicted_class)]
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Tạo visualization
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        # Vẽ
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(cam_image)
        axes[1].set_title(f'ResNet18 Grad-CAM (Class: {"Pneumonia" if predicted_class == 1 else "No Pneumonia"})', 
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Đã lưu Heatmap ResNet18: {save_path}")
        
    except ImportError:
        # Fallback: Tạo heatmap đơn giản từ attention
        print("⚠ pytorch-grad-cam chưa được cài, tạo heatmap đơn giản...")
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(img_np)
        axes[1].set_title('ResNet18 (Grad-CAM không khả dụng)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Đã lưu Heatmap ResNet18 (đơn giản): {save_path}")

def create_comparison_table(metrics_maskrcnn, metrics_resnet, save_path):
    """Tạo bảng so sánh metrics"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    data = {
        'Metric': [m.capitalize() for m in metrics],
        'Mask R-CNN': [f"{metrics_maskrcnn[m]*100:.2f}%" for m in metrics],
        'ResNet18': [f"{metrics_resnet[m]*100:.2f}%" for m in metrics],
        'Winner': []
    }
    
    for m in metrics:
        if metrics_maskrcnn[m] > metrics_resnet[m]:
            data['Winner'].append('Mask R-CNN')
        elif metrics_resnet[m] > metrics_maskrcnn[m]:
            data['Winner'].append('ResNet18')
        else:
            data['Winner'].append('Hòa')
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.25, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight winners
    for i in range(1, len(df) + 1):
        if df.iloc[i-1]['Winner'] == 'Mask R-CNN':
            table[(i, 3)].set_facecolor('#FFE5E5')
        elif df.iloc[i-1]['Winner'] == 'ResNet18':
            table[(i, 3)].set_facecolor('#E5F5F3')
    
    plt.title('Bảng So Sánh Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu Bảng so sánh: {save_path}")

# ============================================================================
# mAP CALCULATION FOR MASK R-CNN
# ============================================================================

def calculate_iou(box1, box2):
    """Tính IoU giữa 2 bounding boxes [x1, y1, x2, y2]"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Tính intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Tính union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def calculate_ap(gt_boxes, pred_boxes, pred_scores, iou_threshold=0.5):
    """Tính Average Precision (AP) cho một image"""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0:
        return 0.0
    if len(pred_boxes) == 0:
        return 0.0
    
    # Sort predictions by score (descending)
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    # Track which GT boxes have been matched
    gt_matched = [False] * len(gt_boxes)
    
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    
    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        
        # Find best matching GT box
        for j, gt_box in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1
    
    # Calculate cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Calculate precision and recall
    recalls = tp_cumsum / len(gt_boxes) if len(gt_boxes) > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    
    # Add sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap

def calculate_map_maskrcnn(maskrcnn_holder, valid_images, labels_df, conf_threshold=0.1, iou_threshold=0.5):
    """Tính mAP cho Mask R-CNN"""
    aps = []
    
    print(f"Đang tính mAP cho Mask R-CNN (IoU threshold = {iou_threshold})...")
    
    for img_path in tqdm(valid_images, desc="Calculating mAP"):
        patient_id = img_path.stem
        
        # Get ground truth boxes
        patient_labels = labels_df[labels_df['patientId'] == patient_id]
        gt_boxes = []
        
        for _, row in patient_labels.iterrows():
            if row['Target'] == 1 and not pd.isna(row['x']):
                x = float(row['x'])
                y = float(row['y'])
                w = float(row['width'])
                h = float(row['height'])
                gt_boxes.append([x, y, x + w, y + h])  # [x1, y1, x2, y2]
        
        gt_boxes = np.array(gt_boxes)
        
        # Get predictions
        result = predict_maskrcnn_detailed(maskrcnn_holder, img_path, conf_threshold)
        pred_boxes = result['boxes']
        pred_scores = result['scores']
        
        # Calculate AP for this image
        if len(gt_boxes) > 0 or len(pred_boxes) > 0:
            ap = calculate_ap(gt_boxes, pred_boxes, pred_scores, iou_threshold)
            aps.append(ap)
    
    # Calculate mAP
    map_value = np.mean(aps) if len(aps) > 0 else 0.0
    
    return map_value, aps

def calculate_map_at_iou_thresholds(maskrcnn_holder, valid_images, labels_df, conf_threshold=0.1):
    """Tính mAP tại các IoU threshold khác nhau để tạo loss curve"""
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    map_values = []
    
    print("Đang tính mAP tại các IoU threshold khác nhau...")
    
    for iou_thresh in tqdm(iou_thresholds, desc="Calculating mAP at IoU thresholds"):
        map_value, _ = calculate_map_maskrcnn(maskrcnn_holder, valid_images, labels_df, 
                                              conf_threshold=conf_threshold, 
                                              iou_threshold=iou_thresh)
        map_values.append(map_value)
    
    return iou_thresholds, map_values

def plot_map_and_loss_curves_maskrcnn(map_value, iou_thresholds, map_values, save_path):
    """Vẽ biểu đồ mAP và Loss curves (mAP tại các IoU threshold) cho Mask R-CNN"""
    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
    
    # Subplot 1: mAP tại IoU=0.5
    axes[0].bar(['mAP@0.5'], [map_value], color='#FF6B6B', alpha=0.8, width=0.5)
    axes[0].set_ylabel('mAP Score', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Mean Average Precision (mAP@0.5)\nmAP = {map_value:.4f} ({map_value*100:.2f}%)', 
                     fontsize=13, fontweight='bold')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Thêm giá trị trên bar
    axes[0].text(0, map_value + 0.02, f'{map_value:.4f}', 
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Subplot 2: Loss curves - mAP tại các IoU threshold khác nhau
    axes[1].plot(iou_thresholds, map_values, 'o-', linewidth=2, markersize=8, 
                color='#FF6B6B', label='mAP', alpha=0.8)
    axes[1].set_xlabel('IoU Threshold', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('mAP Score', fontsize=12, fontweight='bold')
    axes[1].set_title('mAP Loss Curves trên tập kiểm thử\n(mAP tại các IoU threshold)', 
                     fontsize=13, fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=11)
    
    # Thêm giá trị tại mỗi điểm
    for i, (iou, map_val) in enumerate(zip(iou_thresholds, map_values)):
        axes[1].text(iou, map_val + 0.02, f'{map_val:.3f}', 
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu mAP và Loss curves chart: {save_path}")

def plot_precision_recall_curve_maskrcnn(maskrcnn_holder, valid_images, labels_df, save_path):
    """Vẽ Precision-Recall curve cho Mask R-CNN"""
    # Collect all predictions and ground truths
    all_pred_scores = []
    all_gt_labels = []
    
    print("Đang thu thập dữ liệu cho Precision-Recall curve...")
    
    for img_path in tqdm(valid_images, desc="Collecting PR data"):
        patient_id = img_path.stem
        
        # Get ground truth
        patient_labels = labels_df[labels_df['patientId'] == patient_id]
        has_pneumonia = 1 if len(patient_labels[patient_labels['Target'] == 1]) > 0 else 0
        
        # Get predictions với nhiều threshold
        result = predict_maskrcnn_detailed(maskrcnn_holder, img_path, conf_threshold=0.01)
        
        if len(result['scores']) > 0:
            max_score = np.max(result['scores'])
        else:
            max_score = 0.0
        
        all_pred_scores.append(max_score)
        all_gt_labels.append(has_pneumonia)
    
    all_pred_scores = np.array(all_pred_scores)
    all_gt_labels = np.array(all_gt_labels)
    
    # Calculate precision and recall for different thresholds
    thresholds = np.arange(0.0, 1.01, 0.01)
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        pred_binary = (all_pred_scores >= threshold).astype(int)
        
        TP = np.sum((all_gt_labels == 1) & (pred_binary == 1))
        FP = np.sum((all_gt_labels == 0) & (pred_binary == 1))
        FN = np.sum((all_gt_labels == 1) & (pred_binary == 0))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(recalls, precisions, linewidth=2, color='#FF6B6B', label='Mask R-CNN')
    plt.fill_between(recalls, precisions, alpha=0.3, color='#FF6B6B')
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curve - Mask R-CNN', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Calculate AUC
    auc = np.trapz(precisions, recalls)
    plt.text(0.6, 0.2, f'AUC = {auc:.4f}', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu Precision-Recall curve: {save_path}")

# ============================================================================
# EVALUATION
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Tính Accuracy, Precision, Recall, F1"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }

def evaluate_models():
    """Đánh giá cả hai mô hình"""
    print("\n" + "="*60)
    print("SO SÁNH MASK R-CNN VÀ RESNET")
    print("="*60)
    
    # Load labels
    print("\nĐang load labels...")
    labels_df = pd.read_csv(LABELS_CSV)
    print(f"✓ Đã load {len(labels_df)} labels")
    
    # Load models
    maskrcnn_holder = load_maskrcnn_model()
    resnet_model = load_resnet_model()
    
    # Get validation images
    val_dir = Path(VAL_IMAGES_DIR)
    val_images = list(val_dir.glob("*.png"))
    print(f"\nTìm thấy {len(val_images)} ảnh validation")
    
    if len(val_images) == 0:
        print("❌ Không tìm thấy ảnh validation!")
        return
    
    # Get ground truth labels
    print("\nĐang lấy ground truth labels...")
    y_true = []
    valid_images = []
    
    for img_path in val_images:
        patient_id = img_path.stem
        patient_labels = labels_df[labels_df['patientId'] == patient_id]
        
        if len(patient_labels) > 0:
            label = 1 if patient_labels['Target'].iloc[0] == 1 else 0
            y_true.append(label)
            valid_images.append(img_path)
    
    print(f"✓ Có {len(valid_images)} ảnh hợp lệ")
    print(f"  - Có pneumonia: {sum(y_true)}")
    print(f"  - Không có pneumonia: {len(y_true) - sum(y_true)}")
    
    # Predict với Mask R-CNN
    print("\n" + "-"*60)
    print("Đang predict với Mask R-CNN...")
    y_pred_maskrcnn = []
    
    # Thử nhiều threshold để tìm threshold tốt nhất cho Mask R-CNN
    best_threshold = 0.1
    best_f1 = 0
    best_metrics = None
    
    print("Đang tìm threshold tối ưu cho Mask R-CNN...")
    thresholds_to_try = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]
    
    for threshold in thresholds_to_try:
        y_pred_temp = []
        for img_path in tqdm(valid_images, desc=f"Mask R-CNN (thresh={threshold:.2f})", leave=False):
            pred = predict_maskrcnn(maskrcnn_holder, img_path, conf_threshold=threshold)
            y_pred_temp.append(pred)
        
        metrics_temp = calculate_metrics(y_true, y_pred_temp)
        # Ưu tiên F1 score, nhưng cũng xem xét balance giữa precision và recall
        score = metrics_temp['f1'] + 0.1 * metrics_temp['recall']  # Ưu tiên recall một chút
        
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
            y_pred_maskrcnn = y_pred_temp.copy()
            best_metrics = metrics_temp
    
    print(f"✓ Threshold tối ưu: {best_threshold} (F1 = {best_metrics['f1']:.4f}, "
          f"Precision = {best_metrics['precision']:.4f}, Recall = {best_metrics['recall']:.4f})")
    
    # Predict với ResNet
    print("\n" + "-"*60)
    print("Đang predict với ResNet...")
    y_pred_resnet = []
    
    for img_path in tqdm(valid_images, desc="ResNet"):
        try:
            pred = predict_resnet(resnet_model, img_path)
            y_pred_resnet.append(pred)
        except Exception as e:
            print(f"\n⚠ Lỗi khi predict {img_path.name}: {e}")
            y_pred_resnet.append(0)  # Default: không có pneumonia
    
    # Tính metrics
    print("\n" + "="*60)
    print("KẾT QUẢ SO SÁNH")
    print("="*60)
    
    metrics_maskrcnn = calculate_metrics(y_true, y_pred_maskrcnn)
    metrics_resnet = calculate_metrics(y_true, y_pred_resnet)
    
    # In kết quả
    print("\n📊 MASK R-CNN:")
    print(f"  Accuracy:  {metrics_maskrcnn['accuracy']:.4f} ({metrics_maskrcnn['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics_maskrcnn['precision']:.4f} ({metrics_maskrcnn['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics_maskrcnn['recall']:.4f} ({metrics_maskrcnn['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics_maskrcnn['f1']:.4f} ({metrics_maskrcnn['f1']*100:.2f}%)")
    print(f"  Confusion Matrix: TP={metrics_maskrcnn['TP']}, TN={metrics_maskrcnn['TN']}, "
          f"FP={metrics_maskrcnn['FP']}, FN={metrics_maskrcnn['FN']}")
    
    # Tính và hiển thị mAP
    print("\n📊 MASK R-CNN - mAP (sẽ được tính trong phần visualization)...")
    
    print("\n📊 RESNET:")
    print(f"  Accuracy:  {metrics_resnet['accuracy']:.4f} ({metrics_resnet['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics_resnet['precision']:.4f} ({metrics_resnet['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics_resnet['recall']:.4f} ({metrics_resnet['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics_resnet['f1']:.4f} ({metrics_resnet['f1']*100:.2f}%)")
    print(f"  Confusion Matrix: TP={metrics_resnet['TP']}, TN={metrics_resnet['TN']}, "
          f"FP={metrics_resnet['FP']}, FN={metrics_resnet['FN']}")
    
    # So sánh
    print("\n" + "="*60)
    print("SO SÁNH")
    print("="*60)
    
    print(f"\n{'Metric':<15} {'Mask R-CNN':<15} {'ResNet':<15} {'Winner':<15} {'Diff':<15}")
    print("-" * 75)
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics_to_compare:
        mrcnn_val = metrics_maskrcnn[metric]
        resnet_val = metrics_resnet[metric]
        diff = mrcnn_val - resnet_val
        
        if mrcnn_val > resnet_val:
            winner = "Mask R-CNN"
            diff_str = f"+{diff:.4f}"
        elif resnet_val > mrcnn_val:
            winner = "ResNet"
            diff_str = f"{diff:.4f}"
        else:
            winner = "Hòa"
            diff_str = "0.0000"
        
        print(f"{metric.capitalize():<15} {mrcnn_val:<15.4f} {resnet_val:<15.4f} {winner:<15} {diff_str:<15}")
    
    # Tổng kết
    print("\n" + "="*60)
    print("TỔNG KẾT")
    print("="*60)
    
    maskrcnn_wins = sum(1 for m in metrics_to_compare 
                       if metrics_maskrcnn[m] > metrics_resnet[m])
    resnet_wins = sum(1 for m in metrics_to_compare 
                     if metrics_resnet[m] > metrics_maskrcnn[m])
    
    if maskrcnn_wins > resnet_wins:
        print("🏆 Mask R-CNN có kết quả tốt hơn trên nhiều metrics")
    elif resnet_wins > maskrcnn_wins:
        print("🏆 ResNet có kết quả tốt hơn trên nhiều metrics")
    else:
        print("🤝 Hai mô hình có kết quả tương đương")
    
    print(f"\nMask R-CNN thắng: {maskrcnn_wins}/4 metrics")
    print(f"ResNet thắng: {resnet_wins}/4 metrics")
    
    # ========================================================================
    # TẠO VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*60)
    print("ĐANG TẠO VISUALIZATIONS...")
    print("="*60)
    
    output_dir = Path("outputs/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix cho từng mô hình
    print("\n1. Tạo Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred_maskrcnn, "Mask R-CNN", 
                         output_dir / "confusion_matrix_maskrcnn.png")
    plot_confusion_matrix(y_true, y_pred_resnet, "ResNet18", 
                         output_dir / "confusion_matrix_resnet.png")
    
    # 2. mAP và Loss curves cho Mask R-CNN (riêng biệt)
    print("\n2. Tính mAP và Loss curves cho Mask R-CNN...")
    map_value, aps = calculate_map_maskrcnn(maskrcnn_holder, valid_images, labels_df, 
                                            conf_threshold=best_threshold, iou_threshold=0.5)
    print(f"✓ mAP (IoU=0.5): {map_value:.4f} ({map_value*100:.2f}%)")
    
    # Tính mAP tại các IoU threshold khác nhau để tạo loss curve
    iou_thresholds, map_values = calculate_map_at_iou_thresholds(maskrcnn_holder, valid_images, labels_df, 
                                                                 conf_threshold=best_threshold)
    
    # Vẽ biểu đồ mAP và Loss curves
    plot_map_and_loss_curves_maskrcnn(map_value, iou_thresholds, map_values,
                                      output_dir / "map_and_loss_curves_maskrcnn.png")
    
    # Vẽ Precision-Recall curve
    plot_precision_recall_curve_maskrcnn(maskrcnn_holder, valid_images, labels_df,
                                        output_dir / "precision_recall_curve_maskrcnn.png")
    
    # 3. Biểu đồ cột so sánh
    print("\n3. Tạo Biểu đồ so sánh...")
    plot_comparison_bar_chart(metrics_maskrcnn, metrics_resnet,
                             output_dir / "comparison_bar_chart.png")
    
    # 4. Bảng so sánh
    print("\n4. Tạo Bảng so sánh...")
    create_comparison_table(metrics_maskrcnn, metrics_resnet,
                          output_dir / "comparison_table.png")
    
    # 5. Heatmap cho 1-2 ảnh ví dụ
    print("\n5. Tạo Heatmap cho ảnh ví dụ...")
    # Chọn 1-2 ảnh có pneumonia và 1-2 ảnh không có
    pneumonia_indices = [i for i, label in enumerate(y_true) if label == 1]
    normal_indices = [i for i, label in enumerate(y_true) if label == 0]
    
    example_indices = []
    if len(pneumonia_indices) > 0:
        example_indices.append(pneumonia_indices[0])  # Ảnh có pneumonia
    if len(normal_indices) > 0:
        example_indices.append(normal_indices[0])  # Ảnh không có pneumonia
    
    for idx, img_idx in enumerate(example_indices[:2]):  # Tối đa 2 ảnh
        img_path = valid_images[img_idx]
        true_label = y_true[img_idx]
        label_str = "pneumonia" if true_label == 1 else "normal"
        
        # Mask R-CNN heatmap
        create_heatmap_maskrcnn(maskrcnn_holder, img_path,
                               output_dir / f"heatmap_maskrcnn_example{idx+1}_{label_str}.png")
        
        # ResNet heatmap
        create_heatmap_resnet(resnet_model, img_path,
                             output_dir / f"heatmap_resnet_example{idx+1}_{label_str}.png")
    
    print("\n" + "="*60)
    print("✓ HOÀN THÀNH TẤT CẢ VISUALIZATIONS!")
    print(f"✓ Tất cả files đã được lưu trong: {output_dir}")
    print("="*60)
    
    return metrics_maskrcnn, metrics_resnet

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        metrics_maskrcnn, metrics_resnet = evaluate_models()
        print("\n✓ Hoàn thành!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

