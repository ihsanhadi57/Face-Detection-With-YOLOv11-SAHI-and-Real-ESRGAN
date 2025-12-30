# -*- coding: utf-8 -*-
"""
Script debugging untuk membandingkan:
1. Direct YOLO inference (seperti baseline)
2. SAHI wrapper inference (seperti di Streamlit)
"""

from ultralytics import YOLO
import cv2
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.yolo_wrapper import YOLOv11PoseDetectionModel
from sahi.prediction import PredictionResult

# Configuration
MODEL_PATH = 'models/yolo11s-pose-default/yolo11s_pose/weights/best.pt'
IMAGE_PATH = 'data/dataset/widerface/wider_val/images/0--parade/0_Parade_Parade_0_628.jpg'
CONFIDENCE = 0.5
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print("="*80)
print("DEBUGGING: YOLO Direct vs SAHI Wrapper")
print("="*80)

# Clear GPU cache
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Load image
img = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"\n📷 Image shape: {img.shape}")

# ============================================================================
# TEST 1: Direct YOLO (Baseline - seperti code Anda yang working)
# ============================================================================
print("\n" + "="*80)
print("TEST 1: Direct YOLO Inference (BASELINE)")
print("="*80)

model_direct = YOLO(MODEL_PATH)

# Test dengan berbagai imgsz
for imgsz in [640, 960, 1024, 1280]:
    print(f"\n--- Testing imgsz={imgsz} ---")
    results = model_direct(img, device=DEVICE, imgsz=imgsz, conf=CONFIDENCE, verbose=False)
    
    num_detections = len(results[0].boxes)
    print(f"✅ Detections: {num_detections}")
    
    if num_detections > 0:
        boxes = results[0].boxes
        print(f"   Confidence range: {boxes.conf.min().item():.3f} - {boxes.conf.max().item():.3f}")
        print(f"   Box sizes (WxH):")
        for i, box in enumerate(boxes.xyxy[:5]):  # Show first 5
            w = box[2] - box[0]
            h = box[3] - box[1]
            print(f"      Box {i+1}: {w:.1f}x{h:.1f}px (conf={boxes.conf[i].item():.3f})")

# Pilih imgsz terbaik untuk model yang dilatih dengan 1024
best_imgsz = 1024
print(f"\n🎯 Using imgsz={best_imgsz} for further testing")

results_baseline = model_direct(img, device=DEVICE, imgsz=best_imgsz, conf=CONFIDENCE, verbose=False)
num_baseline = len(results_baseline[0].boxes)
print(f"\n📊 BASELINE RESULT: {num_baseline} faces detected with imgsz={best_imgsz}")

# ============================================================================
# TEST 2: SAHI Wrapper (Standard Detection - seperti di Streamlit)
# ============================================================================
print("\n" + "="*80)
print("TEST 2: SAHI Wrapper - Standard Detection (STREAMLIT TANPA SAHI)")
print("="*80)

# Initialize wrapper dengan berbagai imgsz
for imgsz in [640, 960, 1024, 1280]:
    print(f"\n--- Testing with image_size={imgsz} ---")
    
    wrapper = YOLOv11PoseDetectionModel(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE,
        device=DEVICE,
        image_size=imgsz,
        load_at_init=True
    )
    
    # Perform inference seperti di Streamlit
    wrapper.perform_inference(img_rgb)
    
    # Create object prediction list
    wrapper._create_object_prediction_list_from_original_predictions(
        shift_amount_list=[[0, 0]],
        full_shape_list=[[img_rgb.shape[0], img_rgb.shape[1]]]
    )
    
    try:
        object_prediction_list = wrapper.object_prediction_list[0]
    except IndexError:
        object_prediction_list = []
    
    if not isinstance(object_prediction_list, list):
        object_prediction_list = [object_prediction_list] if object_prediction_list else []
    
    num_detections = len(object_prediction_list)
    print(f"✅ Detections: {num_detections}")
    
    if num_detections > 0:
        confidences = [pred.score.value for pred in object_prediction_list]
        print(f"   Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
        print(f"   Box sizes (WxH):")
        for i, pred in enumerate(object_prediction_list[:5]):
            bbox = pred.bbox
            w = bbox.maxx - bbox.minx
            h = bbox.maxy - bbox.miny
            print(f"      Box {i+1}: {w:.1f}x{h:.1f}px (conf={pred.score.value:.3f})")

# ============================================================================
# TEST 3: Inspect Wrapper Internal Configuration
# ============================================================================
print("\n" + "="*80)
print("TEST 3: Inspect SAHI Wrapper Configuration")
print("="*80)

wrapper_inspect = YOLOv11PoseDetectionModel(
    model_path=MODEL_PATH,
    confidence_threshold=CONFIDENCE,
    device=DEVICE,
    image_size=best_imgsz,
    load_at_init=True
)

print(f"\n🔍 Wrapper attributes:")
print(f"   - confidence_threshold: {wrapper_inspect.confidence_threshold}")
print(f"   - device: {wrapper_inspect.device}")
print(f"   - image_size: {wrapper_inspect.image_size if hasattr(wrapper_inspect, 'image_size') else 'NOT SET'}")
print(f"   - category_mapping: {wrapper_inspect.category_mapping}")
print(f"   - model type: {type(wrapper_inspect.model)}")

# Check if model has imgsz attribute
if hasattr(wrapper_inspect.model, 'overrides'):
    print(f"   - model.overrides: {wrapper_inspect.model.overrides}")

# ============================================================================
# ANALISIS
# ============================================================================
print("\n" + "="*80)
print("ANALISIS")
print("="*80)

print(f"""
Model Training Parameters:
- Training imgsz: 1024
- Training batch: 16

Baseline Result (Direct YOLO):
- Detections with imgsz=1024: {num_baseline}
- Status: ✅ WORKING

Streamlit Result (SAHI Wrapper):
- Need to check if imgsz parameter is actually used
- Check wrapper implementation

Possible Issues:
1. ❓ SAHI wrapper tidak meneruskan image_size ke model
2. ❓ Image preprocessing berbeda antara direct vs wrapper
3. ❓ Confidence threshold diterapkan berbeda
4. ❓ NMS (Non-Maximum Suppression) settings berbeda
""")

print("\n" + "="*80)
print("REKOMENDASI")
print("="*80)
print("""
1. Check YOLOv11PoseDetectionModel.perform_inference() implementation
2. Pastikan image_size parameter benar-benar digunakan saat inference
3. Cek apakah ada preprocessing berbeda (resize, padding, etc)
4. Compare model.predict() parameters antara direct vs wrapper
""")

mem_used = torch.cuda.max_memory_allocated() / 1e6
print(f"\n💾 Peak GPU memory used: {mem_used:.2f} MB")