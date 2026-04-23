from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # ✅ ADDED
import torch
import cv2
import numpy as np
from PIL import Image
from model import DeepfakeDetector
from face_extractor import FaceExtractor
from torchvision import transforms
import hashlib

# -------------------------------
# App Init
# -------------------------------
app = FastAPI()

# -------------------------------
# ✅ CORS FIX (IMPORTANT)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (dev mode)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 1. Device (GPU if available)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 2. Load Model
# -------------------------------
model = DeepfakeDetector()

checkpoint = torch.load("saved_models/best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()

# -------------------------------
# 3. Face Extractor
# -------------------------------
extractor = FaceExtractor(target_size=(224, 224))

# -------------------------------
# 4. Transform
# -------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# Optional: Home Route (no more 404)
# -------------------------------
@app.get("/")
def home():
    return {"message": "DeepTrust API is running 🚀"}

# -------------------------------
# 5. Verify Route
# -------------------------------
@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    try:
        # Read file
        content = await file.read()

        # Hash
        file_hash = hashlib.sha256(content).hexdigest()

        # Bytes → OpenCV
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Face extraction
        face = extractor.extract_face_from_array(img)

        if face is None:
            return {"error": "No face detected in the image."}

        # BGR → RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Transform
        input_tensor = transform(face).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            prob = output.item()

        # Decision
        is_fake = prob > 0.5
        confidence = round(prob * 100 if is_fake else (1 - prob) * 100, 2)

        return {
            "video_hash": file_hash,
            "verdict": "FAKE" if is_fake else "REAL",
            "confidence": f"{confidence}%",
            "raw_score": prob
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# 6. Run Server
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)