
# Quick Test Script - Save as test_setup.py

from pathlib import Path
import sys

print("="*60)
print("DEEP TRUST - SETUP VERIFICATION")
print("="*60)

# Check Python version
print(f"\n1. Python Version: {sys.version.split()[0]}")
if sys.version_info < (3, 8):
    print("   ⚠️  Warning: Python 3.8+ recommended")
else:
    print("   ✓ OK")

# Check required files
files = [
    'model.py', 'dataloader.py', 'train.py', 'evaluate.py',
    'deeptrust_api.py', 'index.html', 'script.js', 'style.css'
]

print("\n2. Required Files:")
for file in files:
    exists = Path(file).exists()
    status = "✓" if exists else "✗"
    print(f"   {status} {file}")

# Check folders
folders = ['processed_data', 'saved_models']
print("\n3. Required Folders:")
for folder in folders:
    exists = Path(folder).exists()
    status = "✓" if exists else "✗"
    print(f"   {status} {folder}/")

# Check model
print("\n4. Trained Model:")
model_path = Path('saved_models/best_model.pth')
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ best_model.pth ({size_mb:.1f} MB)")
else:
    print("   ✗ best_model.pth not found")

# Check data
print("\n5. Dataset:")
data_folders = [
    'processed_data/train/real',
    'processed_data/train/fake',
    'processed_data/valid/real',
    'processed_data/valid/fake'
]

for folder in data_folders:
    folder_path = Path(folder)
    if folder_path.exists():
        count = len(list(folder_path.glob('*.jpg'))) + len(list(folder_path.glob('*.png')))
        print(f"   ✓ {folder}: {count:,} images")
    else:
        print(f"   ✗ {folder}: not found")

# Check imports
print("\n6. Required Packages:")
packages = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'fastapi': 'FastAPI',
    'uvicorn': 'Uvicorn',
    'mediapipe': 'MediaPipe',
    'cv2': 'OpenCV',
    'PIL': 'Pillow',
    'sklearn': 'Scikit-learn'
}

for module, name in packages.items():
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} - Run: pip install {module if module != 'cv2' else 'opencv-python'}")

print("\n" + "="*60)
print("Setup verification complete!")
print("="*60)
