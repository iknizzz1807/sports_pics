# Sports Classifier - Ứng Dụng Phân Loại Ảnh Thể Thao

Ứng dụng web sử dụng trí tuệ nhân tạo để phân loại các môn thể thao từ hình ảnh. Hệ thống sử dụng mô hình Vision Transformer (ViT) kết hợp với thuật toán K-Nearest Neighbors (KNN) để nhận diện.

## Tính Năng

- Upload ảnh thể thao và nhận kết quả phân loại tức thì
- Sử dụng mô hình ViT pre-trained từ Google (`google/vit-base-patch16-224-in21k`)
- Giao diện web đơn giản, dễ sử dụng
- API REST với FastAPI

## Cấu Trúc Dự Án

```
sports_pics/
├── backend/
│   ├── main.py           # FastAPI server
│   ├── core.py           # Logic phân loại (ViT + KNN)
│   ├── requirements.txt  # Các thư viện Python cần thiết
│   └── artifacts/        # Các file embeddings đã train sẵn
│       ├── vit_embeddings.pkl
│       └── vgg16_embeddings.pkl
├── frontend/
│   ├── index.html        # Giao diện chính
│   ├── styles.css        # CSS styling
│   └── scripts.js        # JavaScript xử lý upload và gọi API
└── README.md
```

## Yêu Cầu Hệ Thống

- Python 3.10+
- Các thư viện Python (xem `requirements.txt`)

## Cài Đặt

### 1. Clone dự án

```bash
git clone <repository-url>
cd sports_pics
```

### 2. Tạo môi trường ảo và cài đặt dependencies

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc: .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 3. Chạy Backend Server

```bash
python main.py
```

Server sẽ chạy tại `http://localhost:8000`

### 4. Mở Frontend

Mở file `frontend/index.html` trực tiếp trong trình duyệt hoặc sử dụng một web server đơn giản:

```bash
cd frontend
python -m http.server 3000
```

Truy cập `http://localhost:3000`

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Kiểm tra trạng thái API |
| POST | `/api/predict` | Phân loại ảnh thể thao |

### Ví dụ gọi API

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "image=@path/to/sports_image.jpg"
```

**Response:**
```json
{
  "label": "football",
  "confidence": 0.95,
  "method": "ViT + KNN"
}
```

## Công Nghệ Sử Dụng

**Backend:**
- FastAPI - Web framework
- PyTorch - Deep learning framework
- Transformers (Hugging Face) - Vision Transformer model
- Scikit-learn - KNN classifier
- Pillow - Xử lý ảnh

**Frontend:**
- HTML5 / CSS3 / JavaScript

## Cách Hoạt Động

1. Người dùng upload ảnh qua giao diện web
2. Ảnh được gửi đến backend qua API `/api/predict`
3. Backend sử dụng Vision Transformer để trích xuất đặc trưng (embeddings) từ ảnh
4. Vector đặc trưng được chuẩn hóa và đưa vào mô hình KNN để phân loại
5. Kết quả (tên môn thể thao + độ tin cậy) được trả về frontend

## Giấy Phép

MIT License
