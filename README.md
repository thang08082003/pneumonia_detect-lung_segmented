---
title: Cascade Pneumonia Detection
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.20.0
app_file: app.py
pinned: false
license: mit
---

# 🫁 Cascade Pneumonia Detection

Ứng dụng phát hiện viêm phổi sử dụng pipeline cascade kết hợp UNet và Mask R-CNN.

## 🚀 Tính năng

- **UNet**: Phân đoạn vùng phổi từ ảnh X-quang
- **Mask R-CNN**: Phát hiện và phân đoạn các vùng viêm phổi
- Hỗ trợ định dạng: PNG, JPG, JPEG, DICOM (.dcm)
- Giao diện trực quan với overlay masks và bounding boxes

## 📋 Yêu cầu

- Python 3.9
- TensorFlow 2.10.0
- Streamlit 1.20.0
- Các dependencies khác xem trong `requirements.txt`

## 🔧 Cài đặt và chạy local

```bash
# Tạo virtual environment
python -m venv venv39
.\venv39\Scripts\Activate.ps1  # Windows PowerShell
# hoặc
source venv39/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy ứng dụng
streamlit run app.py
```

## 📦 Deploy lên Hugging Face Spaces

### Bước 1: Chuẩn bị files

1. **Tạo repository trên Hugging Face Spaces**
   - Vào https://huggingface.co/spaces
   - Click "New Space"
   - Chọn SDK: **Streamlit**
   - Đặt tên space của bạn

2. **Upload các files cần thiết:**
   ```
   - app.py (file chính)
   - requirements.txt
   - README.md (file này)
   - unet_lung_seg.hdf5 (model UNet)
   - mask_rcnn_pneumonia_0015.h5 (model Mask R-CNN)
   ```

### Bước 2: Upload models

**Quan trọng:** Các file model (.h5 và .hdf5) thường rất lớn (>100MB). Bạn có 2 cách:

#### Cách 1: Sử dụng Git LFS (Khuyến nghị)

```bash
# Cài đặt Git LFS
git lfs install

# Clone repository của bạn
git clone https://huggingface.co/spaces/your-username/your-space-name
cd your-space-name

# Track các file model lớn
git lfs track "*.h5"
git lfs track "*.hdf5"

# Copy models vào thư mục
cp /path/to/unet_lung_seg.hdf5 .
cp /path/to/mask_rcnn_pneumonia_0015.h5 .

# Commit và push
git add .
git commit -m "Add models"
git push
```

#### Cách 2: Upload qua giao diện web

1. Vào trang Space của bạn trên Hugging Face
2. Click tab "Files and versions"
3. Click "Add file" → "Upload files"
4. Upload các file model (có thể mất thời gian nếu file lớn)

### Bước 3: Cấu hình Space

Đảm bảo file `README.md` có metadata đúng:
```yaml
---
title: Cascade Pneumonia Detection
emoji: 🫁
sdk: streamlit
sdk_version: 1.20.0
app_file: app.py
---
```

### Bước 4: Chờ build

Hugging Face sẽ tự động build và deploy ứng dụng của bạn. Quá trình này có thể mất 5-10 phút.

## 📁 Cấu trúc thư mục trên Hugging Face Spaces

```
your-space/
├── app.py                 # File chính
├── requirements.txt       # Dependencies
├── README.md             # Documentation
├── unet_lung_seg.hdf5    # UNet model
└── mask_rcnn_pneumonia_0015.h5  # Mask R-CNN model
```

## ⚠️ Lưu ý

1. **Kích thước models**: Các file model có thể rất lớn. Đảm bảo sử dụng Git LFS để tránh vấn đề với Git.

2. **GPU**: Hugging Face Spaces miễn phí không có GPU. Nếu cần GPU, bạn có thể upgrade lên CPU Basic hoặc cao hơn.

3. **Thời gian load**: Lần đầu tiên load models có thể mất vài phút. Sử dụng `@st.cache_resource` để cache models.

4. **Giới hạn**: 
   - Free tier: 16GB RAM, 2 CPU cores
   - Models lớn có thể cần nhiều RAM hơn

## 🔍 Troubleshooting

### Model không load được
- Kiểm tra đường dẫn file model trong `app.py`
- Đảm bảo file model đã được upload đúng
- Kiểm tra logs trong Hugging Face Space

### Lỗi memory
- Giảm kích thước input image
- Sử dụng CPU Basic tier (có nhiều RAM hơn)

### Lỗi dependencies
- Kiểm tra `requirements.txt` có đầy đủ packages
- Xem logs build để biết package nào bị lỗi

## 📝 License

MIT License - Chỉ dùng cho mục đích nghiên cứu, không dùng cho mục đích lâm sàng.

## 🙏 Credits

- UNet model: Trained on Montgomery and Shenzhen datasets
- Mask R-CNN: Based on Matterport implementation
- Dataset: RSNA Pneumonia Detection Challenge

# pneumonia-lungsegmented
