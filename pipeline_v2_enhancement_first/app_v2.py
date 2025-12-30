import os
import re
import time
import math
import numpy as np
from PIL import Image
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sahi.predict import get_sliced_prediction
from sahi.annotation import BoundingBox
from utils.insightface_wrapper import InsightFaceDetectionModel
from utils.enhancer import FaceEnhancer
from utils.visualization import draw_detections_on_image


# --- helper pilih slice 3x3 atau 4x4 ---
def choose_slice_params(img_w: int, img_h: int, prefer="auto"):
    """
    Pilih slicing grid 3x3 (9 slice) atau 4x4 (16 slice).
    prefer="auto" -> pilih otomatis:
        - sisi terpanjang < 3000 px -> 3x3
        - sisi terpanjang >= 3000 px -> 4x4
    prefer="3x3" -> paksa 3x3
    prefer="4x4" -> paksa 4x4
    """
    long_side = max(img_w, img_h)

    if prefer == "3x3" or (prefer == "auto" and long_side < 3000):
        cols, rows = 3, 3
    else:
        cols, rows = 4, 4

    slice_w = math.ceil(img_w / cols)
    slice_h = math.ceil(img_h / rows)

    # stabilkan ke kelipatan 64
    def round64(x): return int(math.ceil(x / 64) * 64)
    slice_w = min(round64(slice_w), img_w)
    slice_h = min(round64(slice_h), img_h)

    # overlap tetap 0.2
    ov = 0.2
    return slice_h, slice_w, ov, ov


def main():
    # --- Konfigurasi Awal ---
    SCALE_FACTOR = 4
    ENHANCEMENT_CONFIG = {
        'model_name': 'RealESRGAN_x4plus',
        'scale': SCALE_FACTOR,
        'tile': 400,
        'half': True
    }
    
    detection_model = InsightFaceDetectionModel(
        confidence_threshold=0.4,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    face_enhancer = FaceEnhancer(**ENHANCEMENT_CONFIG)

    source_image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'input'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'output'))
    os.makedirs(output_dir, exist_ok=True)

    # --- Pilih Gambar ---
    test_image_name = "7_Cheering_Cheering_7_209.jpg"
    test_image_path = os.path.join(source_image_dir, test_image_name)

    if not os.path.exists(test_image_path):
        print(f"Error: Gambar tidak ditemukan di '{test_image_path}'")
        return

    print(f"Memulai pipeline V2 (Enhance-First) untuk: {test_image_name}")

    # --- Persiapan Direktori & Nama File Output ---
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', os.path.splitext(test_image_name)[0])
    result_dir = os.path.join(output_dir, f"result_{clean_name}_v2")
    os.makedirs(result_dir, exist_ok=True)
    visual_output_path = os.path.join(result_dir, f"visual_{clean_name}_v2.png")

    # --- Tahap 1: Enhancement Gambar ---
    print("Tahap 1: Melakukan enhancement pada seluruh gambar...")
    start_enhance = time.time()
    
    original_image = cv2.imread(test_image_path)
    
    # Fungsi enhancer.enhance mengembalikan gambar BGR (OpenCV)
    enhanced_image, success = face_enhancer.enhance_image(original_image)
    
    enhance_time = time.time() - start_enhance
    print(f"Enhancement selesai dalam {enhance_time:.2f} detik.")
    
    h_orig, w_orig, _ = original_image.shape
    h_enh, w_enh, _ = enhanced_image.shape
    print(f"Ukuran gambar: Asli=({w_orig}, {h_orig}), Enhanced=({w_enh}, {h_enh})")

    # --- Tahap 2: Deteksi Wajah pada Gambar Enhanced (selalu SAHI 3x3/4x4) ---
    print("\nTahap 2: Melakukan deteksi wajah pada gambar yang sudah di-enhance...")
    start_detect = time.time()
    
    # get_sliced_prediction memerlukan path file, jadi kita simpan sementara
    temp_enhanced_path = os.path.join(result_dir, "temp_enhanced.png")
    cv2.imwrite(temp_enhanced_path, enhanced_image)

    # pilih slice
    slice_h, slice_w, ov_h, ov_w = choose_slice_params(w_enh, h_enh, prefer="auto")
    print(f"SAHI params -> slice: {slice_w}x{slice_h}, overlap: {int(ov_w*100)}% (grid 3x3 atau 4x4)")

    detection_result = get_sliced_prediction(
        image=temp_enhanced_path,
        detection_model=detection_model,
        slice_height=slice_h,
        slice_width=slice_w,
        overlap_height_ratio=ov_h,
        overlap_width_ratio=ov_w,
    )

    detect_time = time.time() - start_detect
    print(f"Deteksi selesai dalam {detect_time:.2f} detik. Ditemukan {len(detection_result.object_prediction_list)} wajah.")
    
    # Hapus file sementara
    os.remove(temp_enhanced_path)

    # --- Tahap 3: Transformasi Koordinat dan Visualisasi ---
    print("\nTahap 3: Menyesuaikan koordinat dan menggambar hasil pada gambar asli...")
    
    # NOTE: bbox logic tetap sama, tidak diubah apapun
    if detection_result.object_prediction_list:
        print(f"Mengkonversi koordinat {len(detection_result.object_prediction_list)} deteksi dari gambar enhanced ke gambar asli...")
        
        for i, pred in enumerate(detection_result.object_prediction_list):
            bbox_enhanced = pred.bbox.to_xyxy()
            
            x1 = bbox_enhanced[0] / SCALE_FACTOR
            y1 = bbox_enhanced[1] / SCALE_FACTOR
            x2 = bbox_enhanced[2] / SCALE_FACTOR
            y2 = bbox_enhanced[3] / SCALE_FACTOR
            
            try:
                scaled_bbox = BoundingBox([x1, y1, x2, y2])
                pred.bbox = scaled_bbox
                print(f"Deteksi {i+1}: Koordinat berhasil dikonversi dari {bbox_enhanced} -> [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            except Exception as e:
                print(f"Error mengkonversi koordinat deteksi {i+1}: {e}")
                try:
                    scaled_bbox = BoundingBox.from_xyxy([x1, y1, x2, y2])
                    pred.bbox = scaled_bbox
                    print(f"Deteksi {i+1}: Koordinat berhasil dikonversi (fallback method)")
                except Exception as e2:
                    print(f"Fallback juga gagal untuk deteksi {i+1}: {e2}")
                    print(f"BoundingBox constructor methods: {[method for method in dir(BoundingBox) if 'from' in method.lower()]}")

    final_image = draw_detections_on_image(original_image, detection_result)
    
    cv2.imwrite(visual_output_path, final_image)
    print(f"Visualisasi disimpan di: {visual_output_path}")

    print(f"\nPipeline V2 selesai. Semua hasil ada di: '{result_dir}'")


if __name__ == "__main__":
    main()
