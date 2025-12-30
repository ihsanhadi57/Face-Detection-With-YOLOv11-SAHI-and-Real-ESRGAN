import sys
import os
import time
import cv2
from pathlib import Path

# Tambahkan parent directory ke Python path
sys.path.append(str(Path(__file__).parent.parent))

from sahi.predict import get_sliced_prediction
from utils.yolo_wrapper import YOLOv11PoseDetectionModel
from utils.visualization import (
    draw_detections, 
    save_face_crops, 
    create_detection_summary
)

def main():
    # ===== KONFIGURASI =====
    INPUT_IMAGE = "data/input/foto abel.jpg"  # Path ke SATU gambar saja
    OUTPUT_DIR = "data/output"
    MODEL_PATH = "models/yolo11s-pose-max/yolo11s_pose/weights/best.pt"
    
    # Parameter SAHI
    SLICE_HEIGHT = 640
    SLICE_WIDTH = 640
    OVERLAP_RATIO = 0.2
    CONFIDENCE_THRESHOLD = 0.6
    DEVICE = 'cuda:0'  # atau 'cpu'
    
    # Parameter visualisasi
    SHOW_KEYPOINTS = True
    SHOW_CONFIDENCE = True
    KEYPOINT_CONF_THRESHOLD = 0.3
    
    # ===== BUAT DIREKTORI OUTPUT PER GAMBAR =====
    base_name = Path(INPUT_IMAGE).stem
    output_folder = os.path.join(OUTPUT_DIR, base_name)
    crops_folder = os.path.join(output_folder, "crop")  # Subfolder untuk crops
    
    # Hapus folder output lama jika ada (untuk hasil yang bersih)
    import shutil
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(crops_folder, exist_ok=True)
    
    # ===== LOAD IMAGE INFO =====
    image = cv2.imread(INPUT_IMAGE)
    if image is None:
        print(f"❌ Error: Tidak dapat membaca gambar {INPUT_IMAGE}")
        return
    
    img_height, img_width = image.shape[:2]
    print(f"\n{'='*60}")
    print(f"📸 Memproses: {INPUT_IMAGE}")
    print(f"📐 Ukuran gambar: {img_width}x{img_height} px")
    print(f"{'='*60}\n")
    
    # ===== INISIALISASI MODEL =====
    print("🔧 Menginisialisasi model YOLOv11-Pose...")
    detection_model = YOLOv11PoseDetectionModel(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        load_at_init=True,
    )
    print("✓ Model berhasil dimuat!\n")
    
    # ===== PREDIKSI DENGAN SAHI =====
    print(f"🔍 Melakukan deteksi dengan SAHI...")
    print(f"   - Slice size: {SLICE_WIDTH}x{SLICE_HEIGHT}")
    print(f"   - Overlap ratio: {OVERLAP_RATIO}")
    print(f"   - Confidence threshold: {CONFIDENCE_THRESHOLD}")

    start_time = time.time()
    result = get_sliced_prediction(
        INPUT_IMAGE,
        detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
    )
    processing_time = time.time() - start_time

    # ✨ KUNCI: Attach keypoints dari cache setelah SAHI merging
    print("\n🔗 Attaching keypoints dari cache...")
    result.object_prediction_list = detection_model.attach_keypoints_to_predictions(
        result.object_prediction_list
    )

    num_detections = len(result.object_prediction_list)
    print(f"\n✓ Deteksi selesai!")
    print(f"   - Waktu proses: {processing_time:.2f} detik")
    print(f"   - Jumlah wajah terdeteksi: {num_detections}\n")

    # Debug: Check keypoints
    print("🔍 Debug - Checking keypoints:")
    for i, det in enumerate(result.object_prediction_list):
        has_kpts = hasattr(det, 'keypoints') and det.keypoints is not None
        print(f"   Detection #{i+1}: has_keypoints = {has_kpts}")
        if has_kpts:
            print(f"      Keypoints shape: {det.keypoints.shape}")
            print(f"      Sample keypoint (left_eye): {det.keypoints[0]}")

    if num_detections == 0:
        print("⚠️  Tidak ada wajah yang terdeteksi!")
        return

    # ===== VISUALISASI DAN SIMPAN =====
    print("\n🎨 Membuat visualisasi...")

    # Visualisasi (keypoints sudah ter-attach)
    output_viz_path = os.path.join(output_folder, f"{base_name}_with_keypoints.jpg")

    draw_detections(
        INPUT_IMAGE,
        result,
        output_viz_path,
        show_confidence=SHOW_CONFIDENCE,
        show_keypoints=SHOW_KEYPOINTS,
        box_color=(0, 255, 0),
        text_color=(255, 255, 255),
        kpt_conf_threshold=KEYPOINT_CONF_THRESHOLD,
        draw_skeleton=False,
    )
    
    # 2. Simpan crop wajah di subfolder crop/
    print("\n✂️  Menyimpan crop wajah...")
    saved_crops = save_face_crops(
        INPUT_IMAGE,
        result,
        crops_folder,  # Simpan di subfolder crop/
        prefix=f"{base_name}_face"
    )
    print(f"✓ {len(saved_crops)} crop wajah disimpan di: {crops_folder}/\n")
    
    # 3. Buat summary
    print("📝 Membuat summary...")
    summary_path = os.path.join(output_folder, f"{base_name}_summary.txt")
    create_detection_summary(
        result,
        INPUT_IMAGE,
        processing_time,
        summary_path,
        img_width,
        img_height,
        SLICE_WIDTH,
        SLICE_HEIGHT
    )
    
    # ===== TAMPILKAN HASIL =====
    print(f"\n{'='*60}")
    print("✨ PROSES SELESAI!")
    print(f"{'='*60}")
    print(f"📂 Semua output disimpan di: {output_folder}/")
    print(f"   ├── {base_name}_with_keypoints.jpg  (visualisasi)")
    print(f"   ├── {base_name}_summary.txt        (ringkasan)")
    print(f"   └── crop/                          ({len(saved_crops)} face crops)")
    print(f"{'='*60}\n")
    
    # Tampilkan detail setiap deteksi
    print("📋 Detail Deteksi:")
    for i, det in enumerate(result.object_prediction_list):
        bbox = [int(c) for c in det.bbox.to_xyxy()]
        conf = det.score.value
        print(f"\n   Wajah #{i+1}:")
        print(f"      BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        print(f"      Confidence: {conf:.3f}")
        
        # Tampilkan keypoints jika ada
        if hasattr(det, 'keypoints') and det.keypoints is not None:
            print(f"      Keypoints: {len(det.keypoints)} points detected")
            for kpt_idx, kpt_name in enumerate(['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']):
                if kpt_idx < len(det.keypoints):
                    x, y, conf_kpt = det.keypoints[kpt_idx]
                    print(f"         - {kpt_name}: ({x:.1f}, {y:.1f}) conf={conf_kpt:.3f}")

if __name__ == "__main__":
    main()