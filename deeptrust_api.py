from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import logging

# ✅ IMPORTANT: Import SAME model used in training
from model import get_model

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI()

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Device
# -------------------------------
device = torch.device("cpu")
logger.info(f"Using device: {device}")

# -------------------------------
# Load Model (FIXED)
# -------------------------------
MODEL_PATH = "saved_models/best_model.pth"

try:
    logger.info(f"Loading model from {MODEL_PATH}...")

    # ✅ Use SAME architecture
    model = get_model(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # ✅ Load correct state_dict
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    logger.info("✅ Model loaded successfully!")

except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise e

# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# Prediction (FIXED)
# -------------------------------
def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        prob = outputs.item()  # already sigmoid

    label = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else 1 - prob

    return label, confidence

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def home():
    return {"message": "DeepTrust API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        prediction, confidence = predict_image(image)

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))