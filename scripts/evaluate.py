import os
import time
from tqdm import tqdm
from utils.insightface_wrapper import InsightFaceDetectionModel
from app_retinaface import detect_faces_retinaface

def run_evaluation():
    """
    Menjalankan proses evaluasi pada WIDER FACE validation set dan menyimpan hasilnya.
    """
    print("Memulai proses evaluasi untuk baseline (RetinaFace)...")

    # --- Konfigurasi ---
    # Path ke direktori utama dataset WIDER FACE
    widerface_dir = os.path.abspath("data/widerface/widerface")
    val_images_dir = os.path.join(widerface_dir, "val", "images")
    ground_truth_file = os.path.join(widerface_dir, "val", "wider_val.txt")

    # Direktori untuk menyimpan hasil prediksi
    output_dir = "eval_results/retinaface_baseline"
    os.makedirs(output_dir, exist_ok=True)

    # Inisialisasi model
    # Confidence threshold diatur sangat rendah (mendekati 0) karena skrip evaluasi WIDER FACE
    # akan melakukan thresholding pada berbagai level. Ini memastikan kita tidak kehilangan deteksi.
    print("1. Menginisialisasi model RetinaFace...")
    detection_model = InsightFaceDetectionModel(confidence_threshold=0.02)
    print("   Model siap.")

    # --- Membaca Ground Truth untuk mendapatkan daftar gambar ---
    print(f"2. Membaca daftar gambar dari: {os.path.basename(ground_truth_file)}")
    with open(ground_truth_file, 'r') as f:
        lines = f.readlines()
    
    image_files = [line.strip() for line in lines if '.jpg' in line]
    print(f"   Ditemukan {len(image_files)} gambar untuk dievaluasi.")

    # --- Proses Deteksi dan Simpan Hasil ---
    print("\n3. Menjalankan deteksi pada validation set...")
    start_time = time.time()

    # Menggunakan tqdm untuk progress bar
    for image_name in tqdm(image_files, desc="Mengevaluasi Gambar"):
        image_path = os.path.join(val_images_dir, image_name)
        
        if not os.path.exists(image_path):
            # print(f"Warning: File gambar tidak ditemukan: {image_path}")
            continue

        # Lakukan deteksi
        detections = detect_faces_retinaface(image_path, detection_model)

        # --- Simpan hasil deteksi ke file teks dengan format WIDER FACE ---
        event_folder = os.path.dirname(image_name)
        prediction_folder = os.path.join(output_dir, event_folder)
        os.makedirs(prediction_folder, exist_ok=True)

        txt_filename = os.path.splitext(os.path.basename(image_name))[0] + ".txt"
        txt_filepath = os.path.join(prediction_folder, txt_filename)

        with open(txt_filepath, 'w') as f_out:
            f_out.write(f"{os.path.splitext(os.path.basename(image_name))[0]}\n")
            f_out.write(f"{len(detections)}\n")
            for bbox in detections:
                # Format: x1 y1 width height score
                x1, y1, x2, y2, score = bbox
                w = x2 - x1
                h = y2 - y1
                f_out.write(f"{x1:.2f} {y1:.2f} {w:.2f} {h:.2f} {score:.4f}\n")

    total_time = time.time() - start_time
    print(f"\nEvaluasi selesai dalam {total_time:.2f} detik.")
    print(f"Hasil prediksi disimpan di direktori: '{output_dir}'")
    print("\nLangkah selanjutnya: Jalankan skrip evaluasi mAP dari repositori evaluasi WIDER FACE dengan menunjuk ke direktori ini.")

if __name__ == '__main__':
    run_evaluation()
