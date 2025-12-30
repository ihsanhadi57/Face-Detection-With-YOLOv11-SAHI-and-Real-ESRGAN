import time
import torch
from ultralytics import YOLO
from thop import profile
from PIL import Image

# ---------------------------------------------------------
# Konfigurasi Path (ISI BAGIAN INI)
# ---------------------------------------------------------
model_path = "models/yolo11s-pose-max/yolo11s_pose/weights/best.pt"       
image_path = "data/input/test_1.jpg"      
imgsz = 640
device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------

# Load model
model = YOLO(model_path)
model.to(device)

print(f"Model berhasil dimuat: {model_path}")
print("Device:", device)


# ---------------------------------------------------------
# Hitung FLOPs (GFLOPs)
# ---------------------------------------------------------
def calculate_flops(model, imgsz):
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
    try:
        flops, params = profile(model.model, inputs=(dummy,), verbose=False)
        print("\n===== HASIL FLOPS =====")
        print(f"Parameters: {params/1e6:.2f} M")
        print(f"FLOPs: {flops/1e9:.2f} GFLOPs\n")
    except Exception as e:
        print("Gagal menghitung FLOPs (layer tidak kompatibel THOP):", e)

calculate_flops(model, imgsz)


# ---------------------------------------------------------
# Hitung waktu inference untuk 1 gambar
# ---------------------------------------------------------
def measure_single_inference(model, image_path, imgsz, device):
    img = Image.open(image_path).convert("RGB")

    # warmup (1x)
    model.predict(img, imgsz=imgsz, device=device, verbose=False)

    # timing
    start = time.time()
    model.predict(img, imgsz=imgsz, device=device, verbose=False)
    end = time.time()

    elapsed = (end - start) * 1000  # ms
    fps = 1 / (end - start)

    print("\n===== HASIL INFERENCE TIME =====")
    print(f"Gambar     : {image_path}")
    print(f"Inference  : {elapsed:.2f} ms")
    print(f"FPS        : {fps:.2f}\n")

measure_single_inference(model, image_path, imgsz, device)
