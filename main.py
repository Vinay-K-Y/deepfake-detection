from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
from model import DeepfakeDetector
from face_extractor import FaceExtractor
from torchvision import transforms
import hashlib
from web3 import Web3
import json

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model
model = DeepfakeDetector()
checkpoint = torch.load("saved_models/best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Blockchain
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

with open("contract-address.json") as f:
    contract_address = json.load(f)["address"]

with open("DeepfakeStorageABI.json") as f:
    abi = json.load(f)

contract = w3.eth.contract(address=contract_address, abi=abi)
account = w3.eth.accounts[0]

# Face extractor
extractor = FaceExtractor(target_size=(224, 224))

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "DeepTrust API running 🚀"}

@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_hash = hashlib.sha256(content).hexdigest()

        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        face = extractor.extract_face_from_array(img)
        if face is None:
            return {"error": "No face detected"}

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        input_tensor = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = model(input_tensor).item()

        is_fake = prob > 0.5
        verdict = "FAKE" if is_fake else "REAL"
        confidence = round(prob * 100 if is_fake else (1 - prob) * 100, 2)

        # 🔗 STORE
        tx = contract.functions.storeResult(file_hash, verdict).transact({
            'from': account
        })

        w3.eth.wait_for_transaction_receipt(tx)

        print("✅ Stored on blockchain")

        # 🔗 GET LATEST RECORD (SAFE)
        latest_record = None
        try:
            total = contract.functions.getTotalRecords().call()

            if total > 0:
                latest_record = contract.functions.getRecord(total - 1).call()
                print("📦 Latest Record:", latest_record)
            else:
                print("⚠️ No records yet")

        except Exception as e:
            print("⚠️ Blockchain read error:", str(e))

        return {
            "video_hash": file_hash,
            "verdict": verdict,
            "confidence": f"{confidence}%",
            "blockchain_record": latest_record
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))