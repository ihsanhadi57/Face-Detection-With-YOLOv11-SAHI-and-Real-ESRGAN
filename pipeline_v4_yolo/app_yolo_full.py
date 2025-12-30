import sys
import os
import time
import cv2
from pathlib import Path
import shutil
import numpy as np

# Tambahkan parent directory ke Python path agar bisa import dari utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sahi.predict import get_sliced_prediction
from utils.yolo_wrapper import YOLOv11PoseDetectionModel
from utils.visualization import (
    draw_detections,
    save_face_crops,
    create_detection_summary
)
from utils.enhancer import FaceEnhancer

def main():
    # ===== KONFIGURASI =====
    INPUT_IMAGE = "data/input/foto abel.jpg"  # Path ke SATU gambar
    OUTPUT_DIR = "data/output"
    MODEL_PATH = "models/yolo11s-pose-max/yolo11s_pose/weights/best.pt"
    
    # Parameter SAHI
    SLICE_HEIGHT = 640
    SLICE_WIDTH = 640
    OVERLAP_RATIO = 0.2
    CONFIDENCE_THRESHOLD = 0.5
    DEVICE = 'cuda:0' if 'cuda' in __import__('torch').cuda.get_device_name(0).lower() else 'cpu'
    
    # Parameter Enhancer (default 2x)
    ENHANCER_MODEL_NAME = 'RealESRGAN_x2plus'
    
    # Parameter visualisasi
    SHOW_KEYPOINTS = True
    SHOW_CONFIDENCE = True
    KEYPOINT_CONF_THRESHOLD = 0.3
    
    # ===== BUAT DIREKTORI OUTPUT =====
    base_name = Path(INPUT_IMAGE).stem
    output_folder = os.path.join(OUTPUT_DIR, f"{base_name}_enhance_first_pipeline")
    crops_folder = os.path.join(output_folder, "crops_from_enhanced")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(crops_folder, exist_ok=True)
    
    # ===== LOAD IMAGE =====
    original_image = cv2.imread(INPUT_IMAGE)
    if original_image is None:
        print(f"❌ Error: Tidak dapat membaca gambar {INPUT_IMAGE}")
        return
    
    orig_h, orig_w = original_image.shape[:2]
    print(f"\n{'='*60}")
    print(f"🚀 MEMULAI PIPELINE 'ENHANCE-FIRST'")
    print(f"📸 Gambar Asli: {INPUT_IMAGE} ({orig_w}x{orig_h} px)")
    print(f"{ '='*60}\n")
    
    # ===== 1. INISIALISASI MODEL-MODEL =====
    print("🔧 1. Menginisialisasi model-model...")
    
    print(f"   - Real-ESRGAN ({ENHANCER_MODEL_NAME})...")
    try:
        face_enhancer = FaceEnhancer(model_name=ENHANCER_MODEL_NAME)
        print("   ✓ Model Enhancer berhasil dimuat!")
    except Exception as e:
        print(f"   ❌ Gagal memuat model Enhancer: {e}")
        return
        
    print("   - YOLOv11-Pose...")
    detection_model = YOLOv11PoseDetectionModel(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        load_at_init=True,
    )
    print("   ✓ Model YOLO berhasil dimuat!")
    print("✓ Semua model siap digunakan.\n")

    # ===== 2. ENHANCE GAMBAR PENUH DI AWAL =====
    print(f"✨ 2. Melakukan enhancement pada seluruh gambar (skala {face_enhancer.scale}x)...")
    start_time_enh = time.time()
    
    enhanced_image, success = face_enhancer.enhance_image(original_image)
    
    processing_time_enh = time.time() - start_time_enh
    
    if not success:
        print("❌ Gagal melakukan enhancement pada gambar. Proses dihentikan.")
        return
        
    enh_h, enh_w = enhanced_image.shape[:2]
    print(f"✓ Enhancement selesai dalam {processing_time_enh:.2f} detik.")
    print(f"   Ukuran gambar baru: {enh_w}x{enh_h} px\n")

    # Simpan gambar yang sudah di-enhance ke file sementara untuk diproses SAHI
    enhanced_image_path = os.path.join(output_folder, f"{base_name}_enhanced_temp.jpg")
    cv2.imwrite(enhanced_image_path, enhanced_image)

    # ===== 3. DETEKSI WAJAH PADA GAMBAR HASIL ENHANCEMENT =====
    print(f"🔍 3. Melakukan deteksi wajah pada gambar enhanced...")
    print(f"   - Input untuk SAHI: {Path(enhanced_image_path).name}")
    print(f"   - Slice: {SLICE_WIDTH}x{SLICE_HEIGHT}, Overlap: {OVERLAP_RATIO}, Conf: {CONFIDENCE_THRESHOLD}")

    start_time_det = time.time()
    sahi_result = get_sliced_prediction(
        enhanced_image_path, # Gunakan gambar enhanced
        detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
    )
    
    sahi_result.object_prediction_list = detection_model.attach_keypoints_to_predictions(
        sahi_result.object_prediction_list
    )
    processing_time_det = time.time() - start_time_det

    num_detections = len(sahi_result.object_prediction_list)
    print(f"✓ Deteksi selesai dalam {processing_time_det:.2f} detik. Ditemukan {num_detections} wajah.\n")

    if num_detections == 0:
        print("⚠️  Tidak ada wajah yang terdeteksi pada gambar enhanced.")
    
    # ===== 4. VISUALISASI, CROP, & SUMMARY =====
    print("🎨 4. Membuat visualisasi, crop, dan summary...")
    
    # Gambar deteksi (bbox dan keypoints) pada gambar ENHANCED
    output_viz_path = os.path.join(output_folder, f"{base_name}_enhanced_with_detections.jpg")
    draw_detections(
        enhanced_image_path, # Sumber gambar adalah yang enhanced
        sahi_result,
        output_viz_path,
        show_confidence=SHOW_CONFIDENCE,
        show_keypoints=SHOW_KEYPOINTS,
        box_color=(0, 255, 0),
        text_color=(255, 255, 255),
        kpt_conf_threshold=KEYPOINT_CONF_THRESHOLD,
        draw_skeleton=False,
    )
    print(f"   - Visualisasi deteksi disimpan di: {output_viz_path}")
    
    # Simpan crop wajah dari gambar ENHANCED
    if num_detections > 0:
        saved_crops = save_face_crops(
            enhanced_image_path, # Sumber crop adalah gambar enhanced
            sahi_result,
            crops_folder,
            prefix=f"{base_name}_face"
        )
        print(f"   - {len(saved_crops)} crop wajah disimpan di: {crops_folder}/")

    # Hapus file enhanced sementara
    os.remove(enhanced_image_path)
    
    # Buat file summary
    summary_path = os.path.join(output_folder, f"{base_name}_summary.txt")
    # Gunakan dimensi gambar enhanced untuk summary
    create_detection_summary(
        sahi_result,
        INPUT_IMAGE, # Tetap catat file input original
        processing_time_det,
        summary_path,
        enh_w, enh_h, # Dimensi gambar yang dideteksi
        SLICE_WIDTH,
        SLICE_HEIGHT
    )
    
    # Tambahkan info enhancement ke summary
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write("\n\n--- PIPELINE INFO ---\n")
        f.write("Pipeline Type: Enhance-First\n")
        f.write(f"Original Image Size: {orig_w}x{orig_h}\n")
        f.write(f"Enhanced Image Size: {enh_w}x{enh_h}\n")
        f.write("\n--- ENHANCEMENT INFO ---\n")
        f.write(f"Model: {face_enhancer.model_name}\n")
        f.write(f"Scale: {face_enhancer.scale}x\n")
        f.write(f"Waktu Proses Enhancement: {processing_time_enh:.2f} detik\n")

    print(f"   - Ringkasan proses disimpan di: {summary_path}")

    # ===== SELESAI =====
    total_time = processing_time_enh + processing_time_det
    print(f"\n{'='*60}")
    print("✨ PIPELINE 'ENHANCE-FIRST' SELESAI!")
    print(f"   Waktu Total: {total_time:.2f} detik (Enhance: {processing_time_enh:.2f}s, Deteksi: {processing_time_det:.2f}s)")
    print(f"{ '='*60}")
    print(f"📂 Semua output disimpan di direktori:")
    print(f"   {output_folder}/")
    print(f"   ├── {Path(output_viz_path).name} (Gambar enhanced dengan deteksi)")
    print(f"   ├── {Path(summary_path).name} (Ringkasan teks)")
    if num_detections > 0:
        print(f"   └── {Path(crops_folder).name}/ ({len(saved_crops)} crop dari gambar enhanced)")
    print(f"{ '='*60}\n")


if __name__ == "__main__":
    main()