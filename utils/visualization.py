import os
import cv2
import numpy as np

# Definisi 5 keypoint untuk face landmark (WIDERFACE format)
# 0: left_eye, 1: right_eye, 2: nose, 3: left_mouth, 4: right_mouth
FACE_KEYPOINT_NAMES = [
    'left_eye',
    'right_eye', 
    'nose',
    'left_mouth',
    'right_mouth'
]

# Koneksi untuk face landmark
FACE_SKELETON = [
    [0, 1],  # left_eye to right_eye
    [0, 2],  # left_eye to nose
    [1, 2],  # right_eye to nose
    [2, 3],  # nose to left_mouth
    [2, 4],  # nose to right_mouth
    [3, 4],  # left_mouth to right_mouth
]

# Warna untuk setiap keypoint (BGR format)
FACE_KEYPOINT_COLORS = [
    (255, 0, 0),    # left_eye - blue
    (0, 255, 0),    # right_eye - green
    (0, 0, 255),    # nose - red
    (255, 255, 0),  # left_mouth - cyan
    (255, 0, 255),  # right_mouth - magenta
]

# Warna untuk skeleton lines
SKELETON_COLOR = (0, 255, 255)  # yellow

def draw_keypoints_and_skeleton(image, keypoints, confidence_threshold=0.3, draw_skeleton=True):
    """
    Menggambar keypoints dan skeleton pada gambar wajah.
    
    Args:
        image: numpy array gambar
        keypoints: array keypoints dengan shape (5, 3) -> [x, y, confidence]
        confidence_threshold: threshold minimal confidence untuk menggambar keypoint
        draw_skeleton: apakah menggambar garis skeleton (default: True)
    """
    if keypoints is None or len(keypoints) == 0:
        return image
    
    # Gambar skeleton (garis antar keypoint) - OPSIONAL
    if draw_skeleton:
        for connection in FACE_SKELETON:
            start_idx, end_idx = connection[0], connection[1]
            
            # Cek apakah kedua keypoint memiliki confidence cukup
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx][2] > confidence_threshold and 
                keypoints[end_idx][2] > confidence_threshold):
                
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                
                # Gambar garis
                cv2.line(image, start_point, end_point, SKELETON_COLOR, 2, cv2.LINE_AA)
    
    # Gambar keypoints (titik)
    for idx, kpt in enumerate(keypoints):
        x, y, conf = kpt[0], kpt[1], kpt[2]
        
        if conf > confidence_threshold:
            # Gambar lingkaran untuk keypoint
            color = FACE_KEYPOINT_COLORS[idx] if idx < len(FACE_KEYPOINT_COLORS) else (255, 255, 255)
            cv2.circle(image, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
            cv2.circle(image, (int(x), int(y)), 3, (255, 255, 255), 1, cv2.LINE_AA)  # white border
    
    return image

def draw_detections(image_path, result, output_path, show_confidence=True, show_keypoints=True, 
                   box_color=(0, 255, 0), text_color=(0, 0, 0), kpt_conf_threshold=0.3,
                   draw_skeleton=False):  # ← TAMBAHKAN PARAMETER INI
    """
    Menggambar kotak deteksi dan keypoints pada gambar dan menyimpannya.
    
    Args:
        image_path: path ke gambar input
        result: hasil prediksi dari SAHI
        output_path: path untuk menyimpan hasil
        show_confidence: tampilkan skor confidence
        show_keypoints: tampilkan keypoints dan skeleton
        box_color: warna bounding box (BGR)
        text_color: warna teks (BGR)
        kpt_conf_threshold: threshold confidence untuk keypoints
        draw_skeleton: tampilkan garis skeleton (default: False)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Gagal membaca gambar dari {image_path}")
        return

    detection_list = result.object_prediction_list
    num_faces = len(detection_list)
    
    print(f"   📊 Jumlah deteksi untuk visualisasi = {num_faces}")

    keypoints_found = 0
    for idx, detection in enumerate(detection_list):
        bbox = [int(c) for c in detection.bbox.to_xyxy()]
        score = detection.score.value
        x1, y1, x2, y2 = bbox

        # Gambar bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

        # Tampilkan confidence score
        if show_confidence:
            label = f"Face: {score:.2f}"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), box_color, cv2.FILLED)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        
        # Gambar keypoints jika ada
        if show_keypoints:
            has_kpts = hasattr(detection, 'keypoints') and detection.keypoints is not None
            
            if has_kpts:
                keypoints_found += 1
                print(f"   ✓ Face #{idx+1}: Keypoints shape = {detection.keypoints.shape}")
                
                # ← PASS draw_skeleton PARAMETER
                image = draw_keypoints_and_skeleton(image, detection.keypoints, kpt_conf_threshold, draw_skeleton)
            else:
                print(f"   ⚠️  Face #{idx+1}: NO keypoints attached!")

    print(f"   📍 Total faces with keypoints: {keypoints_found}/{num_faces}")

    # Simpan hasil
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    success = cv2.imwrite(output_path, image)
    if success:
        print(f"   ✓ Hasil visualisasi disimpan ke: {output_path}")
    else:
        print(f"   ❌ Gagal menyimpan visualisasi ke: {output_path}")


def draw_detections_on_image(image, result, show_confidence=True, show_keypoints=True,
                            box_color=(0, 255, 0), text_color=(255, 255, 255), kpt_conf_threshold=0.3,
                            draw_skeleton=False):  # ← TAMBAHKAN PARAMETER INI
    """
    Menggambar kotak deteksi dan keypoints pada objek gambar (numpy array) dan mengembalikannya.
    """
    output_image = image.copy()
    detection_list = result.object_prediction_list

    for detection in detection_list:
        bbox = [int(c) for c in detection.bbox.to_xyxy()]
        score = detection.score.value
        x1, y1, x2, y2 = bbox

        # Gambar bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), box_color, 2)

        # Tampilkan confidence score
        if show_confidence:
            label = f"Face: {score:.2f}"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(output_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), box_color, cv2.FILLED)
            cv2.putText(output_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        
        # Gambar keypoints jika ada
        if show_keypoints and hasattr(detection, 'keypoints') and detection.keypoints is not None:
            # ← PASS draw_skeleton PARAMETER
            output_image = draw_keypoints_and_skeleton(output_image, detection.keypoints, kpt_conf_threshold, draw_skeleton)

    return output_image

def save_face_crops(image_path, result, output_dir, prefix="face_crop"):
    """
    Memotong dan menyimpan setiap wajah yang terdeteksi.
    
    Args:
        image_path: path ke gambar input
        result: hasil prediksi dari SAHI
        output_dir: direktori output untuk crop
        prefix: prefix nama file
    
    Returns:
        List path file yang disimpan
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Gagal membaca gambar dari {image_path}")
        return []
        
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, detection in enumerate(result.object_prediction_list):
        bbox = [int(c) for c in detection.bbox.to_xyxy()]
        score = detection.score.value
        x1, y1, x2, y2 = bbox
        
        # Pastikan koordinat valid
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        face_crop = image[y1:y2, x1:x2]

        if face_crop.size > 0:
            filename = f"{prefix}_{i+1}_conf_{score:.2f}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, face_crop)
            saved_paths.append(filepath)
    
    return saved_paths

def create_detection_summary(result, image_path, processing_time, output_path, 
                            img_width, img_height, slice_width, slice_height):
    """
    Membuat file teks berisi ringkasan hasil deteksi.
    
    Args:
        result: hasil prediksi dari SAHI
        image_path: path gambar input
        processing_time: waktu pemrosesan (detik)
        output_path: path untuk menyimpan summary
        img_width: lebar gambar
        img_height: tinggi gambar
        slice_width: lebar slice
        slice_height: tinggi slice
    """
    num_faces = len(result.object_prediction_list)
    scores = [p.score.value for p in result.object_prediction_list]
    
    avg_confidence = np.mean(scores) if scores else 0
    min_confidence = min(scores) if scores else 0
    max_confidence = max(scores) if scores else 0

    summary = f"""
=== Ringkasan Deteksi Wajah dengan Keypoints ===

--- Informasi Proses ---
Gambar Sumber: {os.path.basename(image_path)}
Ukuran Gambar Asli: {img_width}x{img_height} px
Ukuran Slice: {slice_width}x{slice_height} px
Waktu Proses Total: {processing_time:.2f} detik

--- Statistik Deteksi ---
Total Wajah Ditemukan: {num_faces}
Rata-rata Skor Kepercayaan: {avg_confidence:.3f}
Skor Kepercayaan Minimum: {min_confidence:.3f}
Skor Kepercayaan Maksimum: {max_confidence:.3f}

--- Detail Deteksi ---
"""
    if not result.object_prediction_list:
        summary += "Tidak ada wajah yang terdeteksi.\n"
    else:
        for i, det in enumerate(result.object_prediction_list):
            bbox = [int(c) for c in det.bbox.to_xyxy()]
            conf = det.score.value
            summary += f"\nWajah #{i+1}:\n"
            summary += f"  - Bounding Box: [x1: {bbox[0]}, y1: {bbox[1]}, x2: {bbox[2]}, y2: {bbox[3]}]\n"
            summary += f"  - Skor Kepercayaan: {conf:.3f}\n"
            
            # Tambahkan info keypoints jika ada
            if hasattr(det, 'keypoints') and det.keypoints is not None:
                summary += f"  - Keypoints:\n"
                for kpt_idx, kpt_name in enumerate(FACE_KEYPOINT_NAMES):
                    if kpt_idx < len(det.keypoints):
                        x, y, kpt_conf = det.keypoints[kpt_idx]
                        summary += f"      {kpt_name}: ({x:.1f}, {y:.1f}) [conf: {kpt_conf:.3f}]\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"✓ Summary disimpan ke: {output_path}")