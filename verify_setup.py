# verify_setup.py
import sys

print("=" * 50)
print("ENVIRONMENT VERIFICATION")
print("=" * 50)

# Python version
print(f"\n✅ Python: {sys.version}")

# PyTorch
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError:
    print("❌ PyTorch not installed")

# CUDA
try:
    import torch
    if torch.cuda.is_available():
        gpu_name   = torch.cuda.get_device_name(0)
        gpu_mem    = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_ver   = torch.version.cuda
        print(f"✅ CUDA: {cuda_ver}")
        print(f"✅ GPU : {gpu_name} ({gpu_mem:.1f} GB VRAM)")

        # Quick GPU test
        x = torch.tensor([1.0, 2.0]).cuda()
        print(f"✅ GPU Tensor Test: {x} (on {x.device})")
    else:
        print("❌ CUDA not available - check your PyTorch installation")
except Exception as e:
    print(f"❌ GPU Error: {e}")

# torchvision
try:
    import torchvision
    print(f"✅ Torchvision: {torchvision.__version__}")
except ImportError:
    print("❌ Torchvision not installed")

# OpenCV
try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError:
    print("❌ OpenCV not installed")

# MediaPipe
try:
    import mediapipe
    print(f"✅ MediaPipe: {mediapipe.__version__}")
except ImportError:
    print("❌ MediaPipe not installed")

# PIL
try:
    from PIL import Image
    import PIL
    print(f"✅ Pillow: {PIL.__version__}")
except ImportError:
    print("❌ Pillow not installed")

# NumPy
try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError:
    print("❌ NumPy not installed")

# Flask
try:
    import flask
    print(f"✅ Flask: {flask.__version__}")
except ImportError:
    print("❌ Flask not installed")

# Scikit-learn
try:
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
except ImportError:
    print("❌ Scikit-learn not installed")

print("\n" + "=" * 50)
print("✅ All good! Ready to train." if torch.cuda.is_available() else "⚠️  Fix GPU issues before training")
print("=" * 50)