import os
import re
import time
from sahi.predict import get_sliced_prediction
from utils.insightface_wrapper import InsightFaceDetectionModel
from utils.visualization import draw_detections, save_face_crops, create_detection_summary
from PIL import Image

def main():
    # --- Konfigurasi Awal ---
    detection_model = InsightFaceDetectionModel(confidence_threshold=0.5)
    source_image_dir = "data/input"
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    test_image_name = "20_Family_Group_Family_Group_20_148.jpg"
    test_image_path = os.path.join(source_image_dir, test_image_name)

    if not os.path.exists(test_image_path):
        print(f"Error: Gambar tes tidak ditemukan di '{test_image_path}'")
        return

    print(f"Memulai deteksi wajah pada gambar: {test_image_path}")
    
    # --- Persiapan Nama & Direktori Output ---
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', os.path.splitext(test_image_name)[0])
    result_dir = os.path.join(output_dir, f"result_{clean_name}")
    os.makedirs(result_dir, exist_ok=True)
    
    crops_dir = os.path.join(result_dir, "face_crops")
    visual_output_path = os.path.join(result_dir, f"visual_{clean_name}.png")
    summary_output_path = os.path.join(result_dir, "summary.txt")

    # --- Logika Ukuran Slice Adaptif ---
    base_slice_size = 512
    with Image.open(test_image_path) as img:
        img_width, img_height = img.size

    slice_height = img_height // 2 if img_height < base_slice_size * 1.5 else base_slice_size
    slice_width = img_width // 2 if img_width < base_slice_size * 1.5 else base_slice_size
    slice_height = max(slice_height, 1)
    slice_width = max(slice_width, 1)

    # --- Proses Deteksi ---
    start_time = time.time()
    
    result = get_sliced_prediction(
        image=test_image_path,
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
    
    processing_time = time.time() - start_time
    
    print(f"\nDeteksi selesai dalam {processing_time:.2f} detik!")

    # --- Pembuatan Output & Visualisasi Baru ---
    
    # 1. Gambar hasil deteksi
    draw_detections(test_image_path, result, visual_output_path)
    print(f"-> Visualisasi utama disimpan di: '{visual_output_path}'")

    # 2. Simpan potongan wajah
    if result.object_prediction_list:
        saved_crops = save_face_crops(test_image_path, result, crops_dir, prefix=clean_name)
        print(f"-> {len(saved_crops)} potongan wajah disimpan di: '{crops_dir}'")
    else:
        print("-> Tidak ada wajah untuk dipotong.")

    # 3. Buat ringkasan teks
    create_detection_summary(
        result=result, 
        image_path=test_image_path, 
        processing_time=processing_time, 
        output_path=summary_output_path, 
        img_width=img_width, 
        img_height=img_height, 
        slice_width=slice_width, 
        slice_height=slice_height
    )
    print(f"-> Ringkasan deteksi disimpan di: '{summary_output_path}'")
    
    print(f"\nSemua hasil disimpan dalam direktori: '{result_dir}'")


if __name__ == "__main__":
    main()
