# 🚀 Hướng dẫn Deploy lên Hugging Face Spaces

## Bước 1: Tạo Space mới

1. Vào https://huggingface.co/spaces
2. Click **"New Space"**
3. Điền thông tin:
   - **Space name**: `pneumonia-detection` (hoặc tên bạn muốn)
   - **SDK**: Chọn **Streamlit**
   - **Visibility**: Public hoặc Private
4. Click **"Create Space"**

## Bước 2: Upload files

### Cách 1: Upload qua web (Dễ nhất)

1. Vào trang Space vừa tạo
2. Click tab **"Files and versions"**
3. Click **"Add file"** → **"Upload files"**
4. Upload các file sau:
   - ✅ `app.py`
   - ✅ `requirements.txt`
   - ✅ `README.md`
   - ✅ `unet_lung_seg.hdf5` (file model lớn)
   - ✅ `mask_rcnn_pneumonia_0015.h5` (file model lớn)

**Lưu ý**: Upload models có thể mất nhiều thời gian (10-30 phút tùy kích thước)

### Cách 2: Sử dụng Git (Khuyến nghị cho models lớn)

```bash
# Cài Git LFS (nếu chưa có)
git lfs install

# Clone repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Track các file lớn với Git LFS
git lfs track "*.h5"
git lfs track "*.hdf5"

# Copy files vào thư mục
cp D:/monaiyolo/app.py .
cp D:/monaiyolo/requirements.txt .
cp D:/monaiyolo/README.md .
cp D:/monaiyolo/unet_lung_seg.hdf5 .
cp D:/monaiyolo/mask_rcnn_pneumonia_0015.h5 .

# Commit và push
git add .
git commit -m "Initial commit: Add pneumonia detection app"
git push
```

## Bước 3: Chờ build

- Hugging Face sẽ tự động build ứng dụng
- Xem tiến trình ở tab **"Logs"**
- Thời gian build: 5-15 phút
- Khi build xong, ứng dụng sẽ tự động chạy!

## Bước 4: Kiểm tra

1. Vào tab **"App"** để xem ứng dụng
2. Upload một ảnh X-quang để test
3. Nếu có lỗi, xem tab **"Logs"** để debug

## ⚠️ Lưu ý quan trọng

1. **Kích thước models**: 
   - Nếu models > 100MB, **BẮT BUỘC** dùng Git LFS
   - Upload qua web có thể fail với file lớn

2. **RAM/CPU**:
   - Free tier: 16GB RAM, 2 CPU cores
   - Nếu models quá lớn, có thể cần upgrade

3. **Thời gian load lần đầu**:
   - Load models lần đầu có thể mất 2-5 phút
   - Sử dụng `@st.cache_resource` để cache

## 🔧 Troubleshooting

### Lỗi "Model file not found"
- Kiểm tra tên file model trong `app.py` có đúng không
- Đảm bảo file đã được upload vào root của Space

### Lỗi memory
- Models quá lớn cho free tier
- Thử giảm kích thước models hoặc upgrade tier

### Build failed
- Xem logs để biết package nào lỗi
- Kiểm tra `requirements.txt` có đúng không

## 📞 Hỗ trợ

Nếu gặp vấn đề, xem:
- Hugging Face Spaces docs: https://huggingface.co/docs/hub/spaces
- Streamlit docs: https://docs.streamlit.io

