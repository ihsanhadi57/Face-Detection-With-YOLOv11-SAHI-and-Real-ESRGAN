"""
Script untuk download dan setup models
"""
import os
import sys
from pathlib import Path

def setup_insightface_models():
    """Setup InsightFace models"""
    print("Setting up InsightFace models...")
    
    try:
        from insightface.app import FaceAnalysis
        
        # Initialize app (akan auto-download models)
        app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        print("InsightFace models downloaded and ready!")
        
        # Test inference
        import numpy as np
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        faces = app.get(test_image)
        print(f"Model test successful! (detected {len(faces)} faces in random image)")
        
    except Exception as e:
        print(f"Error setting up InsightFace: {e}")

def create_sample_images():
    """Create sample test images jika belum ada"""
    import cv2
    import numpy as np
    
    sample_dir = Path("data/input/test_samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create simple test images
    for i in range(3):
        # Create colored test image
        img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        
        # Add text
        cv2.putText(img, f"Test Image {i+1}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "Add real faces for testing", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Save image
        cv2.imwrite(str(sample_dir / f"sample{i+1}.jpg"), img)
    
    print(f"Created 3 sample images in {sample_dir}")
    print("Please replace with real images containing faces for testing!")

def main():
    print("Setting up models and sample data...")
    
    # Setup models
    setup_insightface_models()
    
    # Create sample images
    create_sample_images()
    
    print("Setup complete!")

if __name__ == "__main__":
    main()