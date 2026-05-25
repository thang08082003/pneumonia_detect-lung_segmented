# Hướng dẫn So sánh Mask R-CNN và ResNet

## Mô tả
Script `compare_models.py` so sánh hiệu suất giữa hai mô hình:
- **Mask R-CNN**: `mask_rcnn_pneumonia_0015.h5` (detection model)
- **ResNet**: `model.pth` (classification model)

## Yêu cầu
- Python 3.9 (khuyến nghị sử dụng venv39)
- Các thư viện: torch, torchvision, tensorflow, mrcnn, pandas, numpy, opencv-python, PIL, tqdm

## Cách sử dụng

### 1. Kích hoạt môi trường ảo
```powershell
.\venv39\Scripts\Activate.ps1
```

### 2. Chạy script
```bash
python compare_models.py
```

## Chức năng

### Tự động tối ưu hóa
- **Mask R-CNN**: Script tự động tìm threshold tối ưu (0.05 - 0.2) để đạt F1 score cao nhất
- **ResNet**: Sử dụng model đã được train sẵn

### Metrics được tính toán
- **Accuracy**: Độ chính xác tổng thể
- **Precision**: Độ chính xác khi dự đoán có pneumonia
- **Recall**: Tỷ lệ phát hiện được các trường hợp có pneumonia
- **F1-Score**: Trung bình điều hòa của Precision và Recall
- **Confusion Matrix**: TP, TN, FP, FN

### Kết quả
Script sẽ hiển thị:
1. Metrics chi tiết cho từng mô hình
2. Bảng so sánh trực tiếp
3. Tổng kết mô hình nào tốt hơn

## Lưu ý
- Đảm bảo các file model tồn tại:
  - `mask_rcnn_pneumonia_0015.h5`
  - `model.pth`
- Dataset validation phải có trong:
  - `rsna-pneumonia-detection-challenge/yolo_rsna_dataset/images/val/`
- File labels:
  - `rsna-pneumonia-detection-challenge/stage_2_train_labels.csv`

## Tối ưu hóa cho Mask R-CNN
Script đã được tối ưu để Mask R-CNN hoạt động tốt nhất:
- Tự động tìm threshold tối ưu
- Xử lý edge cases
- Normalize ảnh đầu vào
- Ưu tiên recall một chút để phát hiện tốt hơn

