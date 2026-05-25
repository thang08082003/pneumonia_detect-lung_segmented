"""
Đánh giá ResNet50 với Loss và mAP
==================================
Tính loss và mAP cho ResNet50 trên tập kiểm thử
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
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models

# ============================================================================
# CONFIG
# ============================================================================

RESNET_MODEL_PATH = "full_model.h5"
VAL_IMAGES_DIR = "rsna-pneumonia-detection-challenge/yolo_rsna_dataset/images/val"
LABELS_CSV = "rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_resnet50_model():
    """Load ResNet50 model từ full_model.h5"""
    print("Đang load ResNet50 model...")
    
    # Tạo ResNet50 model
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: no pneumonia, pneumonia
    
    # Load weights từ .h5 file (Keras format)
    try:
        import h5py
        with h5py.File(RESNET_MODEL_PATH, 'r') as f:
            # Đọc weights từ Keras model
            # Cần convert từ Keras format sang PyTorch format
            print("⚠ Đang load từ Keras .h5 format...")
            # Thử load trực tiếp nếu có converter
            try:
                # Nếu file là PyTorch format được lưu với extension .h5
                checkpoint = torch.load(RESNET_MODEL_PATH, map_location=DEVICE, weights_only=False)
                
                state_dict = None
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'weights' in checkpoint:
                        state_dict = checkpoint['weights']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                model.load_state_dict(state_dict, strict=False)
                print("✓ Đã load ResNet50 thành công")
            except Exception as e:
                print(f"⚠ Lỗi khi load: {e}")
                print("⚠ Sử dụng model mặc định (chưa được train)")
    except Exception as e:
        print(f"⚠ Không thể load weights: {e}")
        print("⚠ Sử dụng model mặc định")
    
    model = model.to(DEVICE)
    model.eval()
    print("✓ ResNet50 đã load xong")
    return model

# ============================================================================
# PREDICTION
# ============================================================================

def predict_resnet50(model, image_path):
    """Predict với ResNet50 - trả về probabilities và loss"""
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
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'logits': outputs[0].cpu().numpy()
        }
    except Exception as e:
        print(f"Warning: Error in ResNet50 prediction for {image_path.name}: {e}")
        return {
            'predicted_class': 0,
            'confidence': 0.0,
            'probabilities': np.array([1.0, 0.0]),
            'logits': np.array([0.0, 0.0])
        }

# ============================================================================
# LOSS CALCULATION
# ============================================================================

def calculate_loss_resnet50(model, valid_images, labels_df):
    """Tính loss trên validation set"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_losses = []
    all_labels = []
    all_predictions = []
    
    print("Đang tính loss trên tập kiểm thử...")
    
    for img_path in tqdm(valid_images, desc="Calculating loss"):
        patient_id = img_path.stem
        
        # Get ground truth label
        patient_labels = labels_df[labels_df['patientId'] == patient_id]
        label = 1 if len(patient_labels[patient_labels['Target'] == 1]) > 0 else 0
        
        # Predict
        result = predict_resnet50(model, img_path)
        
        # Calculate loss
        logits_tensor = torch.tensor(result['logits']).unsqueeze(0).to(DEVICE)
        label_tensor = torch.tensor([label], dtype=torch.long).to(DEVICE)
        
        loss = criterion(logits_tensor, label_tensor)
        all_losses.append(loss.item())
        all_labels.append(label)
        all_predictions.append(result['predicted_class'])
    
    avg_loss = np.mean(all_losses)
    
    return avg_loss, all_losses, all_labels, all_predictions

# ============================================================================
# mAP CALCULATION (Classification mAP)
# ============================================================================

def calculate_map_resnet50(model, valid_images, labels_df, conf_threshold=0.5):
    """Tính mAP cho ResNet50 (classification mAP)"""
    all_scores = []
    all_labels = []
    
    print(f"Đang tính mAP cho ResNet50 (confidence threshold = {conf_threshold})...")
    
    for img_path in tqdm(valid_images, desc="Calculating mAP"):
        patient_id = img_path.stem
        
        # Get ground truth label
        patient_labels = labels_df[labels_df['patientId'] == patient_id]
        label = 1 if len(patient_labels[patient_labels['Target'] == 1]) > 0 else 0
        
        # Predict
        result = predict_resnet50(model, img_path)
        
        # Lấy confidence cho class pneumonia (class 1)
        pneumonia_prob = result['probabilities'][1]
        
        all_scores.append(pneumonia_prob)
        all_labels.append(label)
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Sort by score descending
    sorted_indices = np.argsort(all_scores)[::-1]
    sorted_scores = all_scores[sorted_indices]
    sorted_labels = all_labels[sorted_indices]
    
    # Calculate precision and recall at different thresholds
    thresholds = np.arange(0.0, 1.01, 0.01)
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        pred_binary = (sorted_scores >= threshold).astype(int)
        
        TP = np.sum((sorted_labels == 1) & (pred_binary == 1))
        FP = np.sum((sorted_labels == 0) & (pred_binary == 1))
        FN = np.sum((sorted_labels == 1) & (pred_binary == 0))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP using trapezoidal rule
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls_sorted = recalls[sorted_indices]
    precisions_sorted = precisions[sorted_indices]
    
    # Calculate AP (Area Under Precision-Recall Curve)
    ap = np.trapz(precisions_sorted, recalls_sorted)
    
    return ap, precisions, recalls

def calculate_map_per_batch(model, valid_images, labels_df, batch_size=100, conf_threshold=0.15):
    """Tính mAP theo từng batch để có nhiều điểm vẽ curve"""
    map_per_batch = []
    
    print(f"Đang tính mAP theo từng batch (batch_size={batch_size})...")
    
    for i in tqdm(range(0, len(valid_images), batch_size), desc="mAP per batch"):
        batch_images = valid_images[i:i+batch_size]
        if len(batch_images) == 0:
            continue
        
        # Tính mAP cho batch này
        map_value, _, _ = calculate_map_resnet50(model, batch_images, labels_df, conf_threshold=conf_threshold)
        map_per_batch.append(map_value)
    
    return map_per_batch

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_loss_and_map_curves(loss_per_image, map_per_batch, avg_loss, avg_map, save_path):
    """Vẽ line chart với 2 đường: Loss và mAP trên cùng 1 biểu đồ"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Trục X: số lượng ảnh (hoặc batch index)
    x_loss = range(len(loss_per_image))
    
    # Tính x cho mAP (mỗi điểm mAP tương ứng với một batch)
    batch_size = len(loss_per_image) // len(map_per_batch) if len(map_per_batch) > 0 else 100
    x_map = [i * batch_size for i in range(len(map_per_batch))]
    
    # Trục Y bên trái: Loss curve
    color_loss = '#9B59B6'  # Màu tím như trong hình
    line1 = ax1.plot(x_loss, loss_per_image, '-', linewidth=1.5, 
                     color=color_loss, alpha=0.7, label='Loss')
    ax1.axhline(y=avg_loss, color=color_loss, linestyle='--', linewidth=1.5, alpha=0.5,
                label=f'Avg Loss = {avg_loss:.4f}')
    ax1.set_xlabel('Image Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold', color=color_loss)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(alpha=0.3)
    
    # Trục Y bên phải: mAP curve
    ax2 = ax1.twinx()
    color_map = '#00CED1'  # Màu cyan như trong hình
    line2 = ax2.plot(x_map, map_per_batch, '-', linewidth=1.5,
                     color=color_map, alpha=0.7, label='mAP')
    ax2.axhline(y=avg_map, color=color_map, linestyle='--', linewidth=1.5, alpha=0.5,
                label=f'Avg mAP = {avg_map:.4f}')
    ax2.set_ylabel('mAP', fontsize=12, fontweight='bold', color=color_map)
    ax2.tick_params(axis='y', labelcolor=color_map)
    ax2.set_ylim([0, max(1.0, max(map_per_batch) * 1.1) if len(map_per_batch) > 0 else 1.0])
    
    # Tiêu đề và legend
    plt.title('Loss and mAP of ResNet50', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu biểu đồ Loss và mAP curves: {save_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("ĐÁNH GIÁ RESNET50 VỚI LOSS VÀ mAP")
    print("="*60)
    
    # Load labels
    print("\nĐang load labels...")
    labels_df = pd.read_csv(LABELS_CSV)
    print(f"✓ Đã load {len(labels_df)} labels")
    
    # Load model
    resnet_model = load_resnet50_model()
    
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
    
    # Tính Loss
    print("\n" + "="*60)
    print("TÍNH LOSS")
    print("="*60)
    avg_loss, loss_per_image, all_labels, all_predictions = calculate_loss_resnet50(
        resnet_model, valid_images, labels_df)
    print(f"\n✓ Average Loss: {avg_loss:.4f}")
    
    # Tính mAP tại confidence threshold = 0.15 (tổng thể)
    print("\n" + "="*60)
    print("TÍNH mAP")
    print("="*60)
    avg_map_value, precisions, recalls = calculate_map_resnet50(
        resnet_model, valid_images, labels_df, conf_threshold=0.15)
    print(f"\n✓ mAP@0.15 (tổng thể): {avg_map_value:.4f} ({avg_map_value*100:.2f}%)")
    
    # Tính mAP theo từng batch để có nhiều điểm vẽ curve
    print("\n" + "="*60)
    print("TÍNH mAP THEO BATCH (ĐỂ VẼ CURVE)")
    print("="*60)
    batch_size = max(50, len(valid_images) // 20)  # Chia thành khoảng 20 batch
    map_per_batch = calculate_map_per_batch(
        resnet_model, valid_images, labels_df, batch_size=batch_size, conf_threshold=0.15)
    print(f"\n✓ Đã tính mAP cho {len(map_per_batch)} batch")
    
    # In kết quả
    print("\n" + "="*60)
    print("KẾT QUẢ")
    print("="*60)
    print(f"\nAverage Loss: {avg_loss:.4f}")
    print(f"\nAverage mAP@0.15: {avg_map_value:.4f} ({avg_map_value*100:.2f}%)")
    print(f"\nmAP theo batch (để vẽ curve): {len(map_per_batch)} điểm")
    
    # Tính các metrics khác
    TP = np.sum((np.array(all_labels) == 1) & (np.array(all_predictions) == 1))
    TN = np.sum((np.array(all_labels) == 0) & (np.array(all_predictions) == 0))
    FP = np.sum((np.array(all_labels) == 0) & (np.array(all_predictions) == 1))
    FN = np.sum((np.array(all_labels) == 1) & (np.array(all_predictions) == 0))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  Confusion Matrix: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    
    # Vẽ biểu đồ
    print("\n" + "="*60)
    print("TẠO BIỂU ĐỒ")
    print("="*60)
    
    output_dir = Path("outputs/resnet_loss_map")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_loss_and_map_curves(loss_per_image, map_per_batch, avg_loss, avg_map_value,
                            output_dir / "loss_and_map_curves.png")
    
    print("\n" + "="*60)
    print("✓ HOÀN THÀNH!")
    print(f"✓ Biểu đồ đã được lưu tại: {output_dir / 'loss_and_map_curves.png'}")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

