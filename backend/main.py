from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core import predictor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Sports Classifier API (ViT Only) is running"}


@app.post("/api/predict")
async def predict_sport(image: UploadFile = File(...)):
    try:
        # Đọc file ảnh
        image_bytes = await image.read()

        # Gọi model xử lý (không cần truyền method nữa)
        result = predictor.predict(image_bytes)

        return result
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
