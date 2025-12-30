"""
RT-DETR Training untuk WIDERFace menggunakan Ultralytics
"""

import os
import yaml
from pathlib import Path
from ultralytics import RTDETR
import torch

# Konversi WIDERFace ke format YOLO
def convert_widerface_to_yolo(anno_file, img_dir, output_dir, split='train'):
    """
    Konversi annotation WIDERFace ke format YOLO
    Format YOLO: class_id center_x center_y width height (normalized)
    """
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, 'labels', split)
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"Converting {split} annotations...")
    
    with open(anno_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    converted_count = 0
    
    while i < len(lines):
        img_path = lines[i].strip()
        i += 1
        
        num_faces = int(lines[i].strip())
        i += 1
        
        # Get image dimensions
        full_img_path = os.path.join(img_dir, img_path)
        
        if not os.path.exists(full_img_path):
            i += num_faces
            continue
            
        from PIL import Image
        img = Image.open(full_img_path)
        img_w, img_h = img.size
        
        # Create label file
        label_file = img_path.replace('.jpg', '.txt').replace('/', '_')
        label_path = os.path.join(labels_dir, label_file)
        
        valid_boxes = []
        for _ in range(num_faces):
            box_info = lines[i].strip().split()
            i += 1
            
            x, y, w, h = map(float, box_info[:4])
            
            # Skip invalid boxes
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                continue
            
            # Convert to YOLO format (normalized center coordinates)
            center_x = (x + w / 2) / img_w
            center_y = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            
            # Clip to [0, 1]
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            norm_w = max(0, min(1, norm_w))
            norm_h = max(0, min(1, norm_h))
            
            valid_boxes.append(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        
        # Write label file
        if valid_boxes:
            with open(label_path, 'w') as lf:
                lf.writelines(valid_boxes)
            converted_count += 1
    
    print(f"Converted {converted_count} images for {split} set")
    return converted_count

def create_dataset_yaml(data_dir, output_path='widerface.yaml'):
    """Create YOLO dataset configuration"""
    config = {
        'path': str(Path(data_dir).absolute()),
        'train': 'WIDER_train_images',
        'val': 'WIDER_val',
        'names': {
            0: 'face'
        },
        'nc': 1  # number of classes
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Dataset config saved to {output_path}")
    return output_path

def setup_directories():
    """Create necessary directories"""
    dirs = [
        'output/checkpoints',
        'output/logs',
        'eval_results',
        'pipeline_v3_RT-DETR'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def train_rtdetr():
    """Main training function"""
    
    print("=" * 50)
    print("RT-DETR WIDERFace Training")
    print("=" * 50)
    
    # Setup
    setup_directories()
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Paths
    data_dir = "data/widerface"
    train_anno = os.path.join(data_dir, "wider_face_split", "wider_face_train_bbx_gt.txt")
    val_anno = os.path.join(data_dir, "wider_face_split", "wider_face_val_bbx_gt.txt")
    train_img_dir = os.path.join(data_dir, "WIDER_train/images")
    val_img_dir = os.path.join(data_dir, "WIDER_val")
    
    # Convert annotations
    print("\n" + "=" * 50)
    print("Converting WIDERFace to YOLO format...")
    print("=" * 50)
    
    if os.path.exists(train_anno):
        convert_widerface_to_yolo(train_anno, train_img_dir, data_dir, 'train')
    else:
        print(f"WARNING: Training annotation not found: {train_anno}")
    
    if os.path.exists(val_anno):
        convert_widerface_to_yolo(val_anno, val_img_dir, data_dir, 'val')
    else:
        print(f"WARNING: Validation annotation not found: {val_anno}")
    
    # Create dataset config
    dataset_yaml = create_dataset_yaml(data_dir)
    
    # Initialize RT-DETR model
    print("\n" + "=" * 50)
    print("Initializing RT-DETR Model...")
    print("=" * 50)
    
    # Load pretrained model
    model = RTDETR('rtdetr-l.pt')  # or 'rtdetr-x.pt' for larger model
    
    # Training parameters
    print("\n" + "=" * 50)
    print("Training Configuration")
    print("=" * 50)
    print(f"Model: RT-DETR-L")
    print(f"Dataset: WIDERFace")
    print(f"Batch Size: 8")
    print(f"Image Size: 640")
    print(f"Epochs: 50")
    print(f"Device: {device}")
    
    # Start training
    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50)
    
    results = model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        project='output',
        name='rtdetr_widerface',
        exist_ok=True,
        
        # Optimizer settings
        optimizer='AdamW',
        lr0=0.0001,
        weight_decay=0.0001,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        
        # Validation
        val=True,
        save=True,
        save_period=5,
        
        # Other settings
        patience=10,
        workers=4,
        verbose=True,
    )
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best weights saved to: output/rtdetr_widerface/weights/best.pt")
    print(f"Last weights saved to: output/rtdetr_widerface/weights/last.pt")
    
    return results

def validate_model(weights_path='output/rtdetr_widerface/weights/best.pt'):
    """Validate trained model"""
    print("\n" + "=" * 50)
    print("Validating Model...")
    print("=" * 50)
    
    model = RTDETR(weights_path)
    results = model.val(data='widerface.yaml')
    
    print("\nValidation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return results

if __name__ == "__main__":
    # Train
    results = train_rtdetr()
    
    # Validate
    # validate_model()