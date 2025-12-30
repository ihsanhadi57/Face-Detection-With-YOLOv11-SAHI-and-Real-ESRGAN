import torch
import os
from PIL import Image

# Tentukan path model dan gambar
MODEL_PATH = 'models/yolo11s-pose-max/yolo11s_pose/weights/best.pt'
IMAGE_PATH = 'data/input/test_1.jpg'
OUTPUT_DIR = 'data/output/result_yolo_direct'

# Buat direktori output jika belum ada
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Muat model YOLOv5 secara langsung
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)

# Buka gambar
img = Image.open(IMAGE_PATH)

# Lakukan inferensi
results = model(img)

# Simpan hasil gambar dengan bounding box
output_path = os.path.join(OUTPUT_DIR, os.path.basename(IMAGE_PATH))
results.save(save_dir=OUTPUT_DIR, exist_ok=True)

print(f"Hasil inferensi langsung disimpan di direktori: {OUTPUT_DIR}")

# Tampilkan hasil (bounding box, class, confidence)
results.print()
