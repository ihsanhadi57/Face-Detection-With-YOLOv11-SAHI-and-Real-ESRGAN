import os
import sys
import time
from tqdm import tqdm
import onnxruntime as ort
from insightface.app import FaceAnalysis
import cv2

# Tambahkan direktori induk ke sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def detect_faces_retinaface(image_path, detector):
    """
    Melakukan deteksi wajah langsung pada gambar menggunakan model RetinaFace.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Tidak dapat membaca gambar: {image_path}")
            return []
        
        faces = detector.get(img)
        
        bboxes = [[int(face.bbox[0]), int(face.bbox[1]), int(face.bbox[2]), int(face.bbox[3]), float(face.det_score)] for face in faces]
        return bboxes

    except Exception as e:
        print(f"Error detecting faces in {image_path}: {e}")
        return []

def evaluate_model(image_list_path, output_dir, detector):
    """
    Evaluasi model deteksi wajah pada daftar gambar dan simpan hasilnya.
    """
    print("Membaca daftar gambar dari:", image_list_path)
    
    # PERBAIKAN: Sesuaikan dengan struktur folder baru
    base_data_path = 'data/dataset/widerface/WIDER_val/images'
    
    # Cek apakah file list ada
    if not os.path.exists(image_list_path):
        print(f"Error: File {image_list_path} tidak ditemukan!")
        return
    
    # Cek apakah folder images ada
    if not os.path.exists(base_data_path):
        print(f"Error: Folder images {base_data_path} tidak ditemukan!")
        return
    
    with open(image_list_path, 'r') as f:
        # PERBAIKI: Hapus garis miring tambahan di awal path dan whitespace
        image_paths = [line.strip().lstrip('/') for line in f.readlines() if line.strip()]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Direktori output dibuat: {output_dir}")

    total_images = len(image_paths)
    print(f"Memulai evaluasi pada {total_images} gambar...")
    print(f"Base path untuk gambar: {base_data_path}")
    
    # Cek beberapa gambar pertama untuk memastikan path benar
    for i, image_path in enumerate(image_paths[:3]):
        full_path = os.path.join(base_data_path, image_path)
        exists = os.path.exists(full_path)
        print(f"Sample {i+1}: {image_path} -> exists: {exists}")
        if not exists:
            # Coba alternatif path jika ada .jpg extension missing
            alt_path = full_path + '.jpg'
            print(f"  Trying alternative: {alt_path} -> exists: {os.path.exists(alt_path)}")
    
    start_time_total = time.time()
    processed_count = 0
    error_count = 0
    
    for image_path in tqdm(image_paths, desc="Mengevaluasi Gambar"):
        full_image_path = os.path.join(base_data_path, image_path)
        
        # Cek apakah file gambar ada
        if not os.path.exists(full_image_path):
            # Coba dengan ekstensi .jpg jika belum ada
            if not full_image_path.endswith('.jpg'):
                full_image_path = full_image_path + '.jpg'
            
            if not os.path.exists(full_image_path):
                error_count += 1
                continue
        
        detected_faces = detect_faces_retinaface(full_image_path, detector)
        
        # PERBAIKAN: Buat nama output file yang konsisten
        output_filename = image_path.replace('/', '_')
        if not output_filename.endswith('.txt'):
            output_filename = output_filename + '.txt'
        
        output_path = os.path.join(output_dir, output_filename)
        
        # Pastikan direktori output ada
        output_sub_dir = os.path.dirname(output_path)
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)
        
        # Tulis hasil deteksi
        with open(output_path, 'w') as f:
            if detected_faces:
                for face in detected_faces:
                    x1, y1, x2, y2, score = face
                    width = x2 - x1
                    height = y2 - y1
                    f.write(f"{int(x1)} {int(y1)} {int(width)} {int(height)} {score:.4f}\n")
            # Jika tidak ada wajah yang terdeteksi, buat file kosong (sesuai format WIDERFace)
        
        processed_count += 1

    end_time_total = time.time()
    print("\n" + "="*50)
    print("EVALUASI SELESAI")
    print("="*50)
    print(f"Total gambar dalam list: {total_images}")
    print(f"Gambar berhasil diproses: {processed_count}")
    print(f"Gambar dengan error/tidak ditemukan: {error_count}")
    print(f"Total waktu evaluasi: {end_time_total - start_time_total:.2f} detik.")
    print(f"Rata-rata waktu per gambar: {(end_time_total - start_time_total)/processed_count:.3f} detik.")
    print(f"Hasil deteksi disimpan di direktori: {output_dir}")
    
    # Tampilkan beberapa contoh file output
    output_files = os.listdir(output_dir)
    print(f"\nContoh file output yang dibuat:")
    for i, filename in enumerate(output_files[:5]):
        file_path = os.path.join(output_dir, filename)
        file_size = os.path.getsize(file_path)
        print(f"  {i+1}. {filename} ({file_size} bytes)")

def verify_dataset_structure():
    """
    Verifikasi struktur dataset dan tampilkan informasi
    """
    print("Memeriksa struktur dataset...")
    
    base_paths = [
        'data/dataset/widerface/WIDER_val',
        'data/dataset/widerface/WIDER_val/images',
        'data/dataset/widerface/WIDER_val/wider_val.txt'
    ]
    
    for path in base_paths:
        exists = os.path.exists(path)
        print(f"  {path}: {'✓' if exists else '✗'}")
        
        if path.endswith('images') and exists:
            # Tampilkan beberapa subfolder dalam images
            subfolders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            print(f"    Subfolders found: {len(subfolders)}")
            print(f"    Sample subfolders: {subfolders[:5]}")
        
        if path.endswith('.txt') and exists:
            # Tampilkan beberapa baris pertama dari file txt
            with open(path, 'r') as f:
                lines = f.readlines()[:5]
            print(f"    Total lines: {len(open(path).readlines())}")
            print(f"    Sample lines:")
            for i, line in enumerate(lines):
                print(f"      {i+1}. {line.strip()}")

if __name__ == '__main__':
    print("="*60)
    print("EVALUASI MODEL RETINAFACE PADA DATASET WIDERFACE")
    print("="*60)
    
    # Verifikasi struktur dataset terlebih dahulu
    verify_dataset_structure()
    print()
    
    print("Menginisialisasi model deteksi...")
    
    providers = ['CPUExecutionProvider']
    ctx_id = -1
    try:
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0
            print("Info: Menggunakan CUDAExecutionProvider untuk akselerasi GPU.")
        else:
            print("Peringatan: CUDA tidak tersedia. Menggunakan CPUExecutionProvider.")
    except Exception as e:
        print(f"Peringatan: Gagal memeriksa CUDA. Menggunakan CPUExecutionProvider. Error: {e}")

    try:
        detector = FaceAnalysis(providers=providers)
        detector.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print("Model berhasil diinisialisasi.")
    except Exception as e:
        print(f"Error: Gagal menginisialisasi model. {e}")
        sys.exit(1)
    
    # PERBAIKAN: Path yang disesuaikan dengan struktur folder baru
    wider_val_list_path = 'data/dataset/widerface/WIDER_val/wider_val.txt'
    output_directory = 'data/eval_results/retinaface'
    
    # Cek sekali lagi sebelum menjalankan evaluasi
    if not os.path.exists(wider_val_list_path):
        print(f"Error: File list tidak ditemukan: {wider_val_list_path}")
        sys.exit(1)
    
    print(f"\nKonfigurasi evaluasi:")
    print(f"  Image list file: {wider_val_list_path}")
    print(f"  Output directory: {output_directory}")
    print(f"  Detection size: 640x640")
    print()
    
    # Jalankan evaluasi
    evaluate_model(wider_val_list_path, output_directory, detector)