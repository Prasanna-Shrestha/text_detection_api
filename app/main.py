from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import requests
from app.trocr_utils import load_model, predict_word

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TrOCR model once globally
model, processor = load_model()

CROP_API_URL = "https://your-crop-api.onrender.com/crop/"  # replace with your deployed Crop API URL


@app.post("/predict/")
async def predict_text(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Call Crop API to get bounding boxes
    try:
        response = requests.post(CROP_API_URL, files={"file": contents})
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop API error: {str(e)}")

    boxes = response.json()
    if not boxes:
        return {"text": ""}

    # Crop words and run TrOCR
    recognized_words = []
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        word_img = image.crop((x, y, x + w, y + h))
        text = predict_word(word_img, model, processor)
        if text:
            recognized_words.append(text)

    full_text = " ".join(recognized_words)
    return {"text": full_text}
