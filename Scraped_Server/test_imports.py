#!/usr/bin/env python3
"""
Quick test to verify all imports work and models can be loaded.
"""

import sys
import time
import numpy as np
import torch

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")

try:
    from ultralytics import YOLO
    print("✓ Ultralytics import OK")
except ImportError as e:
    print(f"✗ Ultralytics import failed: {e}")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__} import OK")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")

# Test YOLO model loading (without camera)
try:
    print("\nTesting YOLO model loading...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = YOLO("yolov8n.pt")
    model.to(device)
    print("✓ YOLO model loaded successfully")
    
    # Test with dummy image
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    device_str = f"cuda:{device.index if hasattr(device, 'index') and device.index is not None else 0}" if device.type == "cuda" else "cpu"
    
    start_time = time.time()
    results = model.predict(source=dummy_img, imgsz=640, conf=0.25, verbose=False, device=device_str)
    end_time = time.time()
    print(f"✓ YOLO prediction test OK ({(end_time - start_time)*1000:.1f}ms)")
    
except Exception as e:
    print(f"✗ YOLO test failed: {e}")

# Test MiDaS model loading
try:
    print("\nTesting MiDaS model loading...")
    start_time = time.time()
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    midas.to(device)
    midas.eval()
    end_time = time.time()
    print(f"✓ MiDaS model loaded successfully ({(end_time - start_time):.1f}s)")
    
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform
    print("✓ MiDaS transforms loaded successfully")
    
    # Test with dummy image
    dummy_img_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    input_batch = transform(dummy_img_rgb).to(device)
    
    start_time = time.time()
    with torch.inference_mode():
        pred = midas(input_batch)
    end_time = time.time()
    print(f"✓ MiDaS prediction test OK ({(end_time - start_time)*1000:.1f}ms)")
    
except Exception as e:
    print(f"✗ MiDaS test failed: {e}")

print("\nTest complete!")