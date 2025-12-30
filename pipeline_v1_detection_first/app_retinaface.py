import cv2
import time
from insightface.app import FaceAnalysis
import onnxruntime as ort

def detect_faces_retinaface(image_path, detector):
    """
    Melakukan deteksi wajah langsung pada gambar menggunakan model RetinaFace.

    Args:
        image_path (str): Path ke file gambar.
        detector (FaceAnalysis): Model deteksi yang sudah diinisialisasi.

    Returns:
        list: Daftar bounding box dan skor kepercayaan.
              Format: [[x1, y1, x2, y2, score], ...]
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Gagal membaca gambar di {image_path}")
            return []

        faces = detector.get(img)

        bboxes = [[int(face.bbox[0]), int(face.bbox[1]), int(face.bbox[2]), int(face.bbox[3]), float(face.det_score)] for face in faces]
        return bboxes

    except Exception as e:
        print(f"Error saat memproses {image_path}: {e}")
        return []

if __name__ == '__main__':
    # Contoh penggunaan
    print("Menginisialisasi model deteksi...")

    # Tentukan provider berdasarkan ketersediaan GPU
    providers = ['CPUExecutionProvider']
    ctx_id = -1
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ctx_id = 0
        print("Info: Menggunakan CUDAExecutionProvider untuk akselerasi GPU.")
    else:
        print("Peringatan: CUDA tidak tersedia. Menggunakan CPUExecutionProvider.")

    # Inisialisasi model FaceAnalysis dengan provider yang dipilih
    detector = FaceAnalysis(providers=providers)
    detector.prepare(ctx_id=ctx_id, det_size=(640, 640))

    print(f"Menggunakan ambang kepercayaan bawaan model InsightFace.")

    image_path = "data/input/test_1.jpg"
    print(f"Mendeteksi wajah pada contoh gambar: {image_path}")

    start_time = time.time()
    detected_faces = detect_faces_retinaface(image_path, detector)
    end_time = time.time()

    print(f"Deteksi selesai dalam {end_time - start_time:.4f} detik.")

    if detected_faces:
        print(f"Ditemukan {len(detected_faces)} wajah:")
        for i, face in enumerate(detected_faces):
            print(f"  - Wajah {i+1}: bbox={face[:4]}, score={face[4]:.4f}")
    else:
        print("Tidak ada wajah yang terdeteksi.")