"""
Đánh giá Mask R-CNN với mAP và Loss Curves
===========================================
Chỉ tập trung vào Mask R-CNN, tính mAP và vẽ loss curves trên tập kiểm thử
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
import matplotlib.pyplot as plt

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
VAL_IMAGES_DIR = "rsna-pneumonia-detection-challenge/yolo_rsna_dataset/images/val"
LABELS_CSV = "rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"

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
# LOAD MODEL
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

# ============================================================================
# PREDICTION
# ============================================================================

def predict_maskrcnn_detailed(holder, image_path, conf_threshold=0.1):
    """
    Predict với Mask R-CNN - trả về detections chi tiết
    Trả về: dict với 'boxes', 'scores', 'masks'
    """
    sess = holder["sess"]
    graph = holder["graph"]
    model = holder["model"]
    
    try:
        # Load image
        img = np.array(Image.open(image_path).convert('L'))
        original_size = img.shape
        
        # Normalize image
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
        
        return {
            'boxes': np.array(boxes),
            'scores': np.array(scores),
            'masks': masks,
            'original_size': original_size
        }
    except Exception as e:
        print(f"Warning: Error in Mask R-CNN prediction for {image_path.name}: {e}")
        return {
            'boxes': np.array([]),
            'scores': np.array([]),
            'masks': [],
            'original_size': (1024, 1024)
        }

# ============================================================================
# mAP CALCULATION
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
    
    print(f"Đang tính mAP (IoU threshold = {iou_threshold})...")
    
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

def build_loss_and_map_curves_from_aps(aps):
    """
    Tạo loss curve và mAP curve từ danh sách AP từng ảnh.
    - Loss được định nghĩa đơn giản là: loss = 1 - AP  (AP cao => loss thấp)
    - mAP curve: chính là AP theo index ảnh.
    """
    aps = np.array(aps, dtype=np.float32)
    
    # Nếu không có AP nào (trường hợp lỗi), trả về giá trị mặc định
    if aps.size == 0:
        return [0.0], [1.0], 1.0, 0.0
    
    loss_per_image = 1.0 - aps
    map_per_image = aps
    
    avg_loss = float(np.mean(loss_per_image))
    avg_map = float(np.mean(map_per_image))
    
    return loss_per_image.tolist(), map_per_image.tolist(), avg_loss, avg_map

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_map_and_loss_curves(loss_per_image, map_per_image, avg_loss, avg_map, save_path):
    """
    Vẽ line chart so sánh Loss và mAP trên cùng 1 biểu đồ (2 đường):
    - Trục Y trái: Loss
    - Trục Y phải: mAP
    - Trục X: index của ảnh (giả sử giống như epoch).
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x = list(range(len(loss_per_image)))
    
    # Đường Loss (màu tím giống ví dụ)
    color_loss = '#9B59B6'
    ax1.plot(x, loss_per_image, '-', color=color_loss, linewidth=1.5, label='Loss')
    ax1.axhline(y=avg_loss, color=color_loss, linestyle='--', linewidth=1.2, alpha=0.6,
                label=f'Avg Loss = {avg_loss:.4f}')
    ax1.set_xlabel('Samples (image index)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (1 - AP)', fontsize=12, fontweight='bold', color=color_loss)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(alpha=0.3)
    
    # Đường mAP (màu xanh cyan)
    ax2 = ax1.twinx()
    color_map = '#00CED1'
    ax2.plot(x, map_per_image, '-', color=color_map, linewidth=1.5, label='mAP')
    ax2.axhline(y=avg_map, color=color_map, linestyle='--', linewidth=1.2, alpha=0.6,
                label=f'Avg mAP = {avg_map:.4f}')
    ax2.set_ylabel('mAP', fontsize=12, fontweight='bold', color=color_map)
    ax2.tick_params(axis='y', labelcolor=color_map)
    ax2.set_ylim([0, 1.0])
    
    # Tiêu đề + legend gộp
    plt.title('Loss và mAP của Mask R-CNN trên tập kiểm thử', fontsize=14, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu biểu đồ Loss và mAP (line chart): {save_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("ĐÁNH GIÁ MASK R-CNN VỚI mAP VÀ LOSS CURVES")
    print("="*60)
    
    # Load labels
    print("\nĐang load labels...")
    labels_df = pd.read_csv(LABELS_CSV)
    print(f"✓ Đã load {len(labels_df)} labels")
    
    # Load model
    maskrcnn_holder = load_maskrcnn_model()
    
    # Get validation images
    val_dir = Path(VAL_IMAGES_DIR)
    val_images = list(val_dir.glob("*.png"))
    print(f"\nTìm thấy {len(val_images)} ảnh validation")
    
    if len(val_images) == 0:
        print("❌ Không tìm thấy ảnh validation!")
        return
    
    # Filter valid images (có trong labels)
    valid_images = []
    for img_path in val_images:
        patient_id = img_path.stem
        patient_labels = labels_df[labels_df['patientId'] == patient_id]
        if len(patient_labels) > 0:
            valid_images.append(img_path)
    
    print(f"✓ Có {len(valid_images)} ảnh hợp lệ")
    
    # Sử dụng threshold cố định để chạy nhanh
    best_threshold = 0.15
    print(f"\n✓ Sử dụng threshold cố định: {best_threshold}")
    
    # Tính mAP@0.15 và tạo loss/mAP curves từ AP từng ảnh
    print("\n" + "="*60)
    print("TÍNH mAP VÀ LOSS CURVES (TỪ AP TỪNG ẢNH)")
    print("="*60)
    avg_map, aps = calculate_map_maskrcnn(
        maskrcnn_holder, valid_images, labels_df,
        conf_threshold=best_threshold, iou_threshold=0.15
    )
    print(f"\n✓ Average mAP@0.15: {avg_map:.4f} ({avg_map*100:.2f}%)")
    
    loss_per_image, map_per_image, avg_loss, avg_map2 = build_loss_and_map_curves_from_aps(aps)
    
    # In kết quả
    print("\n" + "="*60)
    print("KẾT QUẢ")
    print("="*60)
    print(f"\nAverage mAP@0.15: {avg_map2:.4f} ({avg_map2*100:.2f}%)")
    print(f"Average Loss (1 - AP): {avg_loss:.4f}")
    
    # Vẽ biểu đồ
    print("\n" + "="*60)
    print("TẠO BIỂU ĐỒ")
    print("="*60)
    
    output_dir = Path("outputs/maskrcnn_map")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_map_and_loss_curves(loss_per_image, map_per_image, avg_loss, avg_map2,
                            output_dir / "map_and_loss_curves.png")
    
    print("\n" + "="*60)
    print("✓ HOÀN THÀNH!")
    print(f"✓ Biểu đồ đã được lưu tại: {output_dir / 'map_and_loss_curves.png'}")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

