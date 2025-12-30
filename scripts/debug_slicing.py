import os
import re
import shutil
from PIL import Image

# Import fungsi-fungsi SAHI yang relevan
from sahi.slicing import slice_image
from sahi.predict import get_prediction, get_sliced_prediction
from utils.insightface_wrapper import InsightFaceDetectionModel

def main():
    print("--- Memulai Sesi Debug Slicing SAHI (Dengan Logika Adaptif V2) ---")

    # 1. Konfigurasi Awal
    # =================================
    confidence_threshold = 0.5
    test_image_name = "test_1.jpg"
    
    source_image_dir = "data/input"
    output_dir = "data/output"
    debug_dir = os.path.join(output_dir, "debug_slicing_output_adaptif_v2") # Folder output baru

    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)

    test_image_path = os.path.join(source_image_dir, test_image_name)
    if not os.path.exists(test_image_path):
        print(f"Error: Gambar tes '{test_image_path}' tidak ditemukan.")
        return

    print(f"Gambar Input: {test_image_path}")
    print(f"Direktori Output Debug: {debug_dir}")

    detection_model = InsightFaceDetectionModel(confidence_threshold=confidence_threshold)

    # 2. Logika Ukuran Slice Adaptif v2 (Lebih Efektif)
    # =================================
    print("\n--- Tahap 1: Menentukan ukuran slice dengan logika adaptif V2 ---")
    base_slice_size = 512
    
    with Image.open(test_image_path) as img:
        img_width, img_height = img.size

    # Logika baru untuk tinggi slice
    if img_height < base_slice_size * 1.5:
        slice_height = img_height // 2
        print(f"Tinggi gambar ({img_height}px) kecil, slice height diatur ke separuhnya: {slice_height}px")
    else:
        slice_height = base_slice_size
        print(f"Tinggi gambar ({img_height}px) besar, slice height diatur ke dasar: {slice_height}px")

    # Logika baru untuk lebar slice
    if img_width < base_slice_size * 1.5:
        slice_width = img_width // 2
        print(f"Lebar gambar ({img_width}px) kecil, slice width diatur ke separuhnya: {slice_width}px")
    else:
        slice_width = base_slice_size
        print(f"Lebar gambar ({img_width}px) besar, slice width diatur ke dasar: {slice_width}px")

    # Pastikan ukuran slice tidak nol untuk gambar yang sangat kecil
    slice_height = max(slice_height, 1)
    slice_width = max(slice_width, 1)

    print(f"\nUkuran gambar asli: {img_width}x{img_height}")
    print(f"Ukuran slice final yang digunakan: {slice_width}x{slice_height}")

    # 3. Tahap Slicing Gambar
    # =================================
    print("\n--- Tahap 2: Memotong gambar menjadi beberapa slice ---")
    overlap_height_ratio = 0.2
    overlap_width_ratio = 0.2

    slice_result = slice_image(
        image=test_image_path,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        output_dir=debug_dir,
        output_file_name="slice_"
    )

    num_slices = len(slice_result.images)
    print(f"Berhasil! Gambar dipotong menjadi {num_slices} slice dan disimpan di '{debug_dir}'")

    # 4. Tahap Deteksi & Visualisasi per-Slice
    # =================================
    print("\n--- Tahap 3: Melakukan deteksi pada setiap slice ---")
    for i, slice_img_np in enumerate(slice_result.images):
        slice_file_name = f"slice_{i+1}.png"
        print(f"  -> Memproses {slice_file_name}...")

        prediction_result_on_slice = get_prediction(
            image=slice_img_np,
            detection_model=detection_model
        )

        if prediction_result_on_slice.object_prediction_list:
            print(f"     Ditemukan {len(prediction_result_on_slice.object_prediction_list)} wajah di slice ini.")
            output_path = os.path.join(debug_dir, f"detected_on_{slice_file_name}")
            prediction_result_on_slice.export_visuals(export_dir=output_path, file_name="bbox")
            print(f"     Visualisasi bbox disimpan di '{output_path}'")
        else:
            print("     Tidak ada wajah yang terdeteksi di slice ini.")

    # 5. Tahap Akhir: Hasil Gabungan
    # =================================
    print("\n--- Tahap 4: Menjalankan pipeline lengkap (slice-detect-merge) ---")
    final_result = get_sliced_prediction(
        image=test_image_path,
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    final_output_path = os.path.join(debug_dir, "_FINAL_MERGED_RESULT.png")
    final_result.export_visuals(export_dir=os.path.dirname(final_output_path), file_name=os.path.basename(final_output_path))
    print(f"Hasil akhir gabungan disimpan di: '{final_output_path}'")

    print("\n--- Sesi Debug Selesai ---")
    print(f"Silakan periksa semua gambar di direktori '{debug_dir}' untuk melihat prosesnya.")

if __name__ == "__main__":
    main()
