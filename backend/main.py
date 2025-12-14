from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core import predictor

app = FastAPI()

# Cấu hình CORS để frontend gọi được API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Sports Classifier API is running"}

@app.post("/api/predict")
async def predict_sport(
    image: UploadFile = File(...),
    method: str = Form(...)
):
    try:
        # Đọc file ảnh
        image_bytes = await image.read()
        
        # Gọi model xử lý
        result = predictor.predict(image_bytes, method)
        
        return result
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
