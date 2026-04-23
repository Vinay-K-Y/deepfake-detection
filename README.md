# 🧠 DeepTrust – Deepfake Detection with Blockchain

🔗 GitHub: https://github.com/Vinay-K-Y/deepfake-detection

DeepTrust is an AI-powered system that detects deepfake images and stores the results securely on a blockchain for tamper-proof verification.

---

## 🚀 Features
- Deepfake detection using PyTorch
- Face extraction & preprocessing
- FastAPI backend
- Simple frontend (HTML/CSS/JS)
- Blockchain storage (Hardhat + Solidity)
- Immutable result logging (hash + verdict)

---

## 🏗️ Architecture
Frontend → FastAPI → AI Model → Blockchain

---

## ⚙️ Setup (Step-by-Step)

### 1. Clone

git clone https://github.com/Vinay-K-Y/deepfake-detection

cd deepfake-detection


### 2. Backend Setup

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


### 3. Run Backend

uvicorn main:app --reload

👉 http://127.0.0.1:8000

---

### 4. Blockchain Setup

cd blockchain
npm install
npx hardhat node


### 5. Deploy Contract (new terminal)

cd blockchain
npx hardhat run scripts/deploy.cjs --network localhost


👉 Copy contract address

---

### 6. Configure Contract

Create `contract_address.json` (root):

{
"address": "PASTE_ADDRESS"
}


Copy ABI from:

blockchain/artifacts/contracts/DeepfakeStorage.sol/DeepfakeStorage.json


Save only `"abi"` into:

DeepfakeStorageABI.json


---

### 7. Run Backend Again

uvicorn main:app --reload


---

### 8. Run Frontend
Open:

frontend/index.html


---

## 🧪 Flow
1. Upload image  
2. Model predicts REAL / FAKE  
3. Hash generated  
4. Result stored on blockchain  
5. Transaction mined  

---

## 📊 Sample Output

{
"video_hash": "abc123...",
"verdict": "FAKE",
"confidence": "92.34%"
}


---

## ⚠️ Notes
- Contract address changes on redeploy
- Always update `contract_address.json`
- Keep Hardhat node running

---

## 🛠️ Tech Stack
- FastAPI (Backend)
- PyTorch + OpenCV (AI)
- Hardhat + Solidity (Blockchain)
- Web3.py
- HTML/CSS/JS (Frontend)

---

## 🐞 Troubleshooting
- Contract error → redeploy  
- Blockchain issue → run `npx hardhat node`  
- No output → check model file  
- Frontend error → verify API URL  

---

## 🚀 Future Scope
- Deploy to Polygon / Sepolia
- MetaMask integration
- Better UI/UX
- Model improvements

---

## 👨‍💻 Author
Vinay

---

⭐ Star the repo if you found it useful!
