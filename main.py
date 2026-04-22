from fastapi import FastAPI, UploadFile, File
import torch
import cv2
import numpy as np
from PIL import Image
from model import DeepfakeDetector
from face_extractor import FaceExtractor
from torchvision import transforms
import hashlib
import os

app = FastAPI()

# 1. Setup Device (Utilizing your RTX 3060)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load Model & Extractor
model = DeepfakeDetector()
# Using the best_model.pth you showed in your VS Code screenshot
checkpoint = torch.load("saved_models/best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

extractor = FaceExtractor(target_size=(224, 224))

# 3. Transform (Matches your dataloader.py val_transform)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    # Read file content
    content = await file.read()
    
    # Generate SHA-256 Hash for the Blockchain "Notary"
    file_hash = hashlib.sha256(content).hexdigest()
    
    # Save temp image for the extractor
    temp_name = f"temp_{file.filename}"
    with open(temp_name, "wb") as f:
        f.write(content)
    
    # Extract Face using your MediaPipe logic
    face = extractor.extract_face(temp_name)
    os.remove(temp_name) # Cleanup
    
    if face is None:
        return {"error": "No face detected in the image/frame."}

    # Inference on GPU
    face_pil = Image.fromarray(face)
    input_tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = output.item()
    
    is_fake = prob > 0.5
    confidence = round(prob * 100 if is_fake else (1 - prob) * 100, 2)
    
    return {
        "video_hash": file_hash,
        "verdict": "FAKE" if is_fake else "REAL",
        "confidence": f"{confidence}%",
        "raw_score": prob
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)