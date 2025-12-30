import sys
from pathlib import Path
import cv2

# Tambahkan parent directory ke Python path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO

# Test langsung dengan YOLO tanpa SAHI
MODEL_PATH = "models/yolo11s-pose-max/yolo11s_pose/weights/best.pt"
IMAGE_PATH = "data/input/test_1.jpg"

print("="*60)
print("DEBUG: Test Keypoints Detection")
print("="*60)

# Load model
print(f"\n1. Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Load image
print(f"2. Loading image: {IMAGE_PATH}")
image = cv2.imread(IMAGE_PATH)
print(f"   Image shape: {image.shape}")

# Predict
print(f"\n3. Running inference...")
results = model(image, conf=0.5, verbose=False)

result = results[0]
print(f"   Number of detections: {len(result.boxes)}")

# Check keypoints
print(f"\n4. Checking keypoints...")
print(f"   hasattr(result, 'keypoints'): {hasattr(result, 'keypoints')}")

if hasattr(result, 'keypoints') and result.keypoints is not None:
    print(f"   ✓ Keypoints detected!")
    print(f"   Keypoints object type: {type(result.keypoints)}")
    print(f"   Keypoints data shape: {result.keypoints.data.shape}")
    print(f"   Keypoints xy shape: {result.keypoints.xy.shape if hasattr(result.keypoints, 'xy') else 'No xy attr'}")
    print(f"   Keypoints conf shape: {result.keypoints.conf.shape if hasattr(result.keypoints, 'conf') else 'No conf attr'}")
    
    # Print first detection details
    if len(result.boxes) > 0:
        print(f"\n5. First detection details:")
        print(f"   BBox: {result.boxes.xyxy[0].cpu().numpy()}")
        print(f"   Confidence: {result.boxes.conf[0].cpu().numpy()}")
        
        kpts = result.keypoints.data[0].cpu().numpy()
        print(f"   Keypoints shape: {kpts.shape}")
        print(f"   Keypoints data:")
        for i, kpt in enumerate(kpts):
            print(f"      Keypoint {i}: x={kpt[0]:.1f}, y={kpt[1]:.1f}, conf={kpt[2]:.3f}")
else:
    print(f"   ✗ NO KEYPOINTS DETECTED!")
    print(f"   This means the model might not be a pose model or keypoints are not enabled")

# Check model info
print(f"\n6. Model info:")
print(f"   Model task: {model.task if hasattr(model, 'task') else 'Unknown'}")
print(f"   Model names: {model.names}")

print("\n" + "="*60)