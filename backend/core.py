import os
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Chỉ import thư viện cần thiết cho ViT
import torch
from transformers import ViTImageProcessor, ViTModel

# Cấu hình đường dẫn artifacts
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


class SportsPredictor:
    def __init__(self):
        print(">>> Khởi động hệ thống nhận diện thể thao (ViT Only)...")

        self.classes = None
        self.model_knn = None
        self.scaler = None
        self.device = "cpu"  # Chạy trên CPU cho đơn giản và ổn định

        # 1. Lấy danh sách tên Class từ file HOG (như bạn yêu cầu)
        self._load_class_names()

        # 2. Load dữ liệu ViT và train KNN
        self._load_vit_and_train_knn()

        # 3. Khởi tạo ViT Extractor (HuggingFace)
        print(">>> Đang tải ViT Pre-trained model (có thể mất vài giây)...")
        self.vit_processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.vit_model = (
            ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            .to(self.device)
            .eval()
        )

        print(">>> Hệ thống sẵn sàng!")

    def _load_class_names(self):
        try:
            path = os.path.join(ARTIFACTS_DIR, "vgg16_embeddings.pkl")
            if not os.path.exists(path):
                print(f"CẢNH BÁO: Không tìm thấy file {path} để lấy tên class!")
                return

            with open(path, "rb") as f:
                data = pickle.load(f)

            if "class_names" in data:
                self.classes = data["class_names"]
                print(f"✅ Đã lấy danh sách class từ VGG16: {self.classes}")
            else:
                print("⚠️ File tồn tại nhưng không có key 'class_names'.")

        except Exception as e:
            print(f"Lỗi khi đọc class từ HOG: {e}")

    def _load_vit_and_train_knn(self):
        """
        Load embeddings của ViT để train KNN.
        """
        try:
            path = os.path.join(ARTIFACTS_DIR, "vit_embeddings.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Không tìm thấy file artifact quan trọng: {path}"
                )

            with open(path, "rb") as f:
                data = pickle.load(f)

            # Chuẩn hóa dữ liệu
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(data["X_train"])
            y_train = data["y_train"]

            # Train KNN
            # Lưu ý: n_neighbors=2 dựa trên code cũ của bạn
            self.model_knn = KNeighborsClassifier(n_neighbors=2)
            self.model_knn.fit(X_train, y_train)

            print("✅ Đã train xong KNN với dữ liệu ViT.")

        except Exception as e:
            print(f"CRITICAL ERROR - Không thể load ViT KNN: {e}")
            self.model_knn = None

    def extract_vit_features(self, image_bytes):
        """
        Trích xuất đặc trưng ảnh sử dụng Vision Transformer
        """
        try:
            # Convert bytes sang ảnh PIL RGB
            img = Image.open(BytesIO(image_bytes)).convert("RGB")

            # Preprocess
            inputs = self.vit_processor(images=img, return_tensors="pt").to(self.device)

            # Forward pass qua model
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                # Lấy vector [CLS] token đại diện cho cả ảnh
                cls_emb = outputs.last_hidden_state[:, 0, :]

            return cls_emb.cpu().numpy()
        except Exception as e:
            print(f"Lỗi khi trích xuất đặc trưng ViT: {e}")
            raise e

    def predict(self, image_bytes):
        if self.model_knn is None:
            return {"error": "Model chưa được khởi tạo thành công."}

        # 1. Trích xuất đặc trưng
        features = self.extract_vit_features(image_bytes)

        # 2. Scale đặc trưng (phải dùng scaler đã fit lúc train)
        features_scaled = self.scaler.transform(features)

        # 3. Dự đoán bằng KNN
        probas = self.model_knn.predict_proba(features_scaled)[0]
        idx = np.argmax(probas)
        confidence = probas[idx]

        # 4. Map index sang tên class
        if self.classes:
            # Đảm bảo index không vượt quá độ dài list class
            if idx < len(self.classes):
                label_name = self.classes[idx]
            else:
                label_name = f"Unknown Class ID {idx}"
        else:
            # Fallback nếu không load được class name
            label_name = str(idx)

        return {
            "label": label_name,
            "confidence": float(confidence),
            "method": "ViT + KNN",
        }


# Singleton instance
predictor = SportsPredictor()
