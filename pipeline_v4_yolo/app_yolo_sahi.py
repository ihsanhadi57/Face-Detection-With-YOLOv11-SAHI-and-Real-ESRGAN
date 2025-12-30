import sys
import os
import time
import cv2
from pathlib import Path
from glob import glob

# Tambahkan parent directory ke Python path
sys.path.append(str(Path(__file__).parent.parent))

from sahi.predict import get_sliced_prediction
from utils.yolo_wrapper import YOLOv11PoseDetectionModel
from utils.visualization import (
    draw_detections, 
    save_face_crops, 
    create_detection_summary
)

def process_single_image(image_path, detection_model, output_base_dir, config):
    """
    Memproses satu gambar dan menyimpan hasil ke folder terpisah.
    
    Args:
        image_path: path ke gambar input
        detection_model: model yang sudah di-load
        output_base_dir: direktori base untuk output
        config: dictionary konfigurasi
    """
    # Buat folder output berdasarkan nama file
    base_name = Path(image_path).stem
    output_folder = os.path.join(output_base_dir, base_name)
    crops_folder = os.path.join(output_folder, "crop")  # Subfolder untuk crops
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(crops_folder, exist_ok=True)
    
    # Load image info
    image = cv2.imread(image_path)
    if image is None:
        print(f"   ❌ Error: Tidak dapat membaca {image_path}")
        return None
    
    img_height, img_width = image.shape[:2]
    
    print(f"\n{'─'*60}")
    print(f"📸 Memproses: {Path(image_path).name}")
    print(f"📐 Ukuran: {img_width}x{img_height} px")
    
    # Prediksi dengan SAHI
    start_time = time.time()
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=config['slice_height'],
        slice_width=config['slice_width'],
        overlap_height_ratio=config['overlap_ratio'],
        overlap_width_ratio=config['overlap_ratio'],
    )
    processing_time = time.time() - start_time
    
    num_detections = len(result.object_prediction_list)
    print(f"✓ Deteksi selesai: {num_detections} wajah | {processing_time:.2f}s")
    
    if num_detections == 0:
        print("⚠️  Tidak ada wajah terdeteksi")
        # Tetap simpan hasil kosong
        summary_path = os.path.join(output_folder, f"{base_name}_summary.txt")
        create_detection_summary(
            result, image_path, processing_time, summary_path,
            img_width, img_height, 
            config['slice_width'], config['slice_height']
        )
        return {
            'image': image_path,
            'num_faces': 0,
            'time': processing_time,
            'output_folder': output_folder
        }
    
    # Attach keypoints ke setiap detection dari storage
    for detection in result.object_prediction_list:
        bbox = detection.bbox.to_xyxy()
        keypoints = detection_model.get_keypoints_for_bbox(bbox)
        if keypoints is not None:
            detection.keypoints = keypoints
    
    # 1. Visualisasi
    output_viz_path = os.path.join(output_folder, f"{base_name}_with_keypoints.jpg")
    draw_detections(
        image_path, result, output_viz_path,
        show_confidence=config['show_confidence'],
        show_keypoints=config['show_keypoints'],
        box_color=(0, 255, 0),
        text_color=(255, 255, 255),
        kpt_conf_threshold=config['kpt_conf_threshold']
    )
    
    # 2. Crop wajah di subfolder crop/
    saved_crops = save_face_crops(
        image_path, result, crops_folder,
        prefix=f"{base_name}_face"
    )
    
    # 3. Summary
    summary_path = os.path.join(output_folder, f"{base_name}_summary.txt")
    create_detection_summary(
        result, image_path, processing_time, summary_path,
        img_width, img_height,
        config['slice_width'], config['slice_height']
    )
    
    print(f"✓ Hasil disimpan di: {output_folder}/")
    
    return {
        'image': image_path,
        'num_faces': num_detections,
        'time': processing_time,
        'num_crops': len(saved_crops),
        'output_folder': output_folder
    }

def main():
    # ===== KONFIGURASI =====
    INPUT_DIR = "data/input"  # Folder berisi gambar-gambar
    OUTPUT_DIR = "data/output"
    MODEL_PATH = "models/yolo11s-pose-max/yolo11s_pose/weights/best.pt"
    
    # Ekstensi gambar yang didukung
    IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # Parameter SAHI dan Model
    config = {
        'slice_height': 640,
        'slice_width': 640,
        'overlap_ratio': 0.2,
        'confidence_threshold': 0.5,
        'device': 'cuda:0',  # atau 'cpu'
        'show_keypoints': True,
        'show_confidence': True,
        'kpt_conf_threshold': 0.3,
    }
    
    # ===== AMBIL SEMUA GAMBAR =====
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob(os.path.join(INPUT_DIR, ext)))
    
    if not image_files:
        print(f"❌ Tidak ada gambar ditemukan di {INPUT_DIR}")
        print(f"   Format yang didukung: {', '.join(IMAGE_EXTENSIONS)}")
        return
    
    print(f"\n{'='*60}")
    print(f"🚀 BATCH PROCESSING - YOLOv11 FACE POSE DETECTION")
    print(f"{'='*60}")
    print(f"📁 Input folder: {INPUT_DIR}")
    print(f"📁 Output folder: {OUTPUT_DIR}")
    print(f"📊 Total gambar: {len(image_files)}")
    print(f"{'='*60}")
    
    # ===== INISIALISASI MODEL =====
    print("\n🔧 Menginisialisasi model...")
    detection_model = YOLOv11PoseDetectionModel(
        model_path=MODEL_PATH,
        confidence_threshold=config['confidence_threshold'],
        device=config['device'],
        load_at_init=True,
    )
    print("✓ Model berhasil dimuat!\n")
    
    # ===== PROSES SEMUA GAMBAR =====
    results = []
    total_start = time.time()
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] ", end="")
        result = process_single_image(image_path, detection_model, OUTPUT_DIR, config)
        if result:
            results.append(result)
    
    total_time = time.time() - total_start
    
    # ===== SUMMARY KESELURUHAN =====
    print(f"\n\n{'='*60}")
    print("✨ BATCH PROCESSING SELESAI!")
    print(f"{'='*60}")
    
    total_faces = sum(r['num_faces'] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    print(f"📊 Statistik:")
    print(f"   - Total gambar diproses: {len(results)}")
    print(f"   - Total wajah terdeteksi: {total_faces}")
    print(f"   - Waktu total: {total_time:.2f} detik")
    print(f"   - Rata-rata per gambar: {avg_time:.2f} detik")
    print(f"\n📂 Semua hasil disimpan di: {OUTPUT_DIR}/")
    print(f"{'='*60}\n")
    
    # Detail per gambar
    print("📋 Detail per gambar:")
    for r in results:
        img_name = Path(r['image']).name
        print(f"   {img_name:30s} → {r['num_faces']:2d} wajah | {r['time']:.2f}s")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()