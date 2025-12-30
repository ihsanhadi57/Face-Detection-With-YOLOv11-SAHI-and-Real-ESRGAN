import os
import re
import time
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sahi.predict import get_sliced_prediction
from utils.insightface_wrapper import InsightFaceDetectionModel
from utils.visualization import draw_detections, save_face_crops, create_detection_summary
from utils.enhancer import FaceEnhancer, enhance_face_crops_batch, create_enhancement_summary
from PIL import Image

def main():
    # --- Konfigurasi Awal ---
    detection_model = InsightFaceDetectionModel(confidence_threshold=0.5)
    source_image_dir = "data/input"
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    # --- Pilih Gambar ---
    test_image_name = "13_Interview_Interview_On_Location_13_138.jpg"
    test_image_path = os.path.join(source_image_dir, test_image_name)

    if not os.path.exists(test_image_path):
        print(f"Error: Gambar tidak ditemukan di '{test_image_path}'")
        return

    print(f"Memulai pipeline deteksi dan enhancement untuk: {test_image_path}")

    # --- Persiapan Direktori & Nama File Output ---
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', os.path.splitext(test_image_name)[0])
    result_dir = os.path.join(output_dir, f"result_{clean_name}")
    os.makedirs(result_dir, exist_ok=True)
    
    crops_dir = os.path.join(result_dir, "face_crops")
    visual_output_path = os.path.join(result_dir, f"visual_{clean_name}.png")
    detection_summary_path = os.path.join(result_dir, "detection_summary.txt")
    enhancement_summary_path = os.path.join(result_dir, "enhancement_summary.txt")

    # --- Logika Ukuran Slice Adaptif ---
    base_slice_size = 512
    with Image.open(test_image_path) as img:
        img_width, img_height = img.size

    slice_height = img_height // 2 if img_height < base_slice_size * 1.5 else base_slice_size
    slice_width = img_width // 2 if img_width < base_slice_size * 1.5 else base_slice_size
    slice_height = max(slice_height, 1)
    slice_width = max(slice_width, 1)

    # --- Tahap 1: Deteksi Wajah ---
    start_time = time.time()
    detection_result = get_sliced_prediction(
        image=test_image_path,
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
    detection_time = time.time() - start_time
    print(f"Deteksi selesai dalam {detection_time:.2f} detik. Ditemukan {len(detection_result.object_prediction_list)} wajah.")

    # --- Tahap 2: Cropping dan Visualisasi ---
    draw_detections(test_image_path, detection_result, visual_output_path)
    
    saved_crops = []
    if detection_result.object_prediction_list:
        saved_crops = save_face_crops(test_image_path, detection_result, crops_dir, prefix=clean_name)
        print(f"Menyimpan {len(saved_crops)} potongan wajah ke '{os.path.basename(crops_dir)}'.")
    else:
        print("Tidak ada wajah terdeteksi untuk dipotong.")

    create_detection_summary(
        result=detection_result, 
        image_path=test_image_path, 
        processing_time=detection_time, 
        output_path=detection_summary_path, 
        img_width=img_width, 
        img_height=img_height, 
        slice_width=slice_width, 
        slice_height=slice_height
    )
    
    # --- Tahap 3: Enhancement Wajah ---
    if saved_crops:
        print("Memulai enhancement wajah dengan Real-ESRGAN...")
        try:
            ENHANCEMENT_CONFIG = {
                'model_name': 'RealESRGAN_x4plus',
                'scale': 4,
                'tile': 400,
                'half': True
            }
            
            enhancer = FaceEnhancer(**ENHANCEMENT_CONFIG)
            
            enhancement_results = enhance_face_crops_batch(
                crops_dir=crops_dir,
                enhancer=enhancer,
                prefix=clean_name
            )
            
            create_enhancement_summary(enhancement_results, enhancement_summary_path)
            
            stats = enhancement_results['statistics']
            if stats['successful'] > 0:
                print(f"Enhancement selesai. {stats['successful']}/{stats['total_files']} wajah berhasil di-enhance dalam {stats['total_time']:.2f} detik.")
            else:
                print("Enhancement gagal untuk semua wajah yang terdeteksi.")

        except Exception as e:
            print(f"Terjadi error saat enhancement: {e}")
    else:
        print("Tidak ada potongan wajah untuk di-enhance.")

    print(f"\nPipeline selesai. Semua hasil ada di: '{result_dir}'")

if __name__ == "__main__":
    main()