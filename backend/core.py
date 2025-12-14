import os

# --- FIX QUAN TRỌNG 1: Tắt GPU để tránh lỗi "No PTX compilation provider" ---
# Dòng này phải đặt TRƯỚC KHI import tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Deep Learning Libs
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

import torch
from transformers import ViTImageProcessor, ViTModel

# Cấu hình đường dẫn artifacts
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

class SportsPredictor:
    def __init__(self):
        print("Loading models & artifacts... This may take a while.")
        self.models = {}
        self.scalers = {}
        self.classes = None
        
        # 1. Load Data & Train KNN in-memory
        self._load_hog_knn()
        self._load_vgg_knn()
        self._load_vit_knn()
        
        # 2. Init Deep Learning extractors
        # VGG16
        print("Initializing VGG16 (CPU Mode)...")
        self.vgg_base = VGG16(weights='imagenet', include_top=False, pooling='avg')
        
        # ViT
        print("Initializing ViT...")
        # Vì đã tắt GPU bằng biến môi trường ở trên, Torch cũng sẽ dùng CPU hoặc cần chỉ định rõ
        self.device = "cpu" 
        self.vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(self.device).eval()
        
        print("System Ready!")

    def _load_hog_knn(self):
        try:
            path = os.path.join(ARTIFACTS_DIR, 'hog_features.pkl')
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(data['X_train'])
            clf = KNeighborsClassifier(n_neighbors=20)
            clf.fit(X_train, data['y_train'])
            
            self.models['HOG'] = clf
            self.scalers['HOG'] = scaler
            if 'class_names' in data: 
                self.classes = data.get('class_names')
        except Exception as e:
            print(f"Warning: Could not load HOG: {e}")

    def _load_vgg_knn(self):
        try:
            path = os.path.join(ARTIFACTS_DIR, 'vgg16_embeddings.pkl')
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            scaler = StandardScaler()
            X_train = scaler.fit_transform(data['X_train'])
            clf = KNeighborsClassifier(n_neighbors=20)
            clf.fit(X_train, data['y_train'])
            
            self.models['VGG16'] = clf
            self.scalers['VGG16'] = scaler
            if not self.classes and 'class_names' in data:
                self.classes = data['class_names']
        except Exception as e:
            print(f"Warning: Could not load VGG16: {e}")

    def _load_vit_knn(self):
        try:
            path = os.path.join(ARTIFACTS_DIR, 'vit_embeddings.pkl')
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            scaler = StandardScaler()
            X_train = scaler.fit_transform(data['X_train'])
            clf = KNeighborsClassifier(n_neighbors=20)
            clf.fit(X_train, data['y_train'])
            
            self.models['ViT'] = clf
            self.scalers['ViT'] = scaler
        except Exception as e:
            print(f"Warning: Could not load ViT: {e}")

    # --- Feature Extraction Functions ---

    def extract_hog(self, image_bytes):
        # Logic HOG
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # --- FIX LỖI 2: Resize về 224x224 để khớp số lượng features (26244) ---
        img = cv2.resize(img, (224, 224)) 

        from skimage.feature import hog
        features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
        return features.reshape(1, -1)

    def extract_vgg(self, image_bytes):
        # Logic VGG
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        
        # --- FIX LỖI 3: Dùng trực tiếp model() thay vì .predict() ---
        # Điều này giúp tránh xung đột luồng trong FastAPI và lỗi session
        emb = self.vgg_base(arr, training=False).numpy()
        return emb

    def extract_vit(self, image_bytes):
        # Logic ViT
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        inputs = self.vit_processor(images=img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            
        return cls_emb.cpu().numpy()

    # --- Main Prediction Function ---
    
    def predict(self, image_bytes, method):
        if method not in self.models:
            raise ValueError(f"Method {method} not ready or invalid")
            
        # 1. Extract features
        if method == 'HOG':
            features = self.extract_hog(image_bytes)
        elif method == 'VGG16':
            features = self.extract_vgg(image_bytes)
        elif method == 'ViT':
            features = self.extract_vit(image_bytes)
            
        # 2. Scale features
        features = self.scalers[method].transform(features)
        
        # 3. Predict KNN
        clf = self.models[method]
        probas = clf.predict_proba(features)[0]
        idx = np.argmax(probas)
        confidence = probas[idx]
        
        # Fallback labels
        label_name = self.classes[idx] if self.classes is not None else str(idx)
        
        return {
            "label": label_name,
            "confidence": float(confidence),
            "method": method
        }

# Singleton instance
predictor = SportsPredictor()
