import json
import os
import cv2
import datetime
from tqdm import tqdm

# ================= KONFIGURASI PATH WIDERFACE OFFICIAL =================
BASE_DIR = os.path.join("data", "dataset", "widerface")

# Path ground truth resmi WiderFace
WIDERFACE_GT_DIR = os.path.join(BASE_DIR, "wider_face_split")
VAL_IMAGES_DIR = os.path.join(BASE_DIR, "WIDER_val", "images")

# File ground truth
GT_FILE = os.path.join(WIDERFACE_GT_DIR, "wider_face_val_bbx_gt.txt")

# File output
OUTPUT_JSON_NAME = "val_ground_truth_coco_filtered.json" # Nama file saya ubah sedikit
OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_JSON_NAME)

def parse_widerface_gt_txt():
    """
    Parse format WiderFace ground truth .txt dengan FILTER INVALID
    
    Format file:
    x y w h blur expression illumination invalid occlusion pose
    """
    print(f"Membaca ground truth dari: {GT_FILE}")
    
    if not os.path.exists(GT_FILE):
        print(f"ERROR: File ground truth tidak ditemukan!")
        return None
    
    gt_dict = {}
    total_faces_found = 0
    total_faces_kept = 0
    total_faces_invalid = 0
    
    with open(GT_FILE, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines if any
        if not line:
            i += 1
            continue

        # Baca nama file gambar (biasanya diakhiri .jpg)
        if line.endswith('.jpg'):
            img_path = line
            i += 1
            
            if i >= len(lines): break
            
            # Baca jumlah wajah
            try:
                num_faces = int(lines[i].strip())
                i += 1
            except ValueError:
                # Handle kasus jika format baris tidak sesuai
                i += 1
                continue
            
            bboxes = []
            
            # Baca bbox sebanyak num_faces
            for _ in range(num_faces):
                if i >= len(lines): break
                
                parts = lines[i].strip().split()
                
                try:
                    # Parse: x y w h
                    x = float(parts[0])
                    y = float(parts[1])
                    w = float(parts[2])
                    h = float(parts[3])
                    
                    # --- MODIFIKASI DIMULAI DI SINI ---
                    # Ambil status invalid (index ke-7)
                    # Struktur: x y w h blur expr illum invalid occl pose
                    # Index:    0 1 2 3 4    5    6     7       8    9
                    
                    is_invalid = 0
                    # Cek panjang array untuk menghindari error index out of range
                    if len(parts) >= 8:
                        is_invalid = int(parts[7])
                    
                    total_faces_found += 1

                    # Logika Filter:
                    # 1. Dimensi harus positif (w>0, h>0)
                    # 2. Tidak boleh invalid (is_invalid == 0)
                    if w > 0 and h > 0:
                        if is_invalid == 0:
                            bboxes.append([x, y, w, h])
                            total_faces_kept += 1
                        else:
                            total_faces_invalid += 1 # Counter untuk statistik
                    
                    # --- MODIFIKASI SELESAI ---
                        
                except (ValueError, IndexError):
                    pass
                
                i += 1
            
            # Simpan hanya jika ada bbox valid
            if bboxes:
                gt_dict[img_path] = bboxes
        else:
            i += 1
    
    print(f"Berhasil parse {len(gt_dict)} gambar.")
    print(f"Statistik Wajah:")
    print(f"  - Total terdeteksi di txt : {total_faces_found}")
    print(f"  - Total VALID (disimpan)  : {total_faces_kept}")
    print(f"  - Total INVALID (dibuang) : {total_faces_invalid}")
    
    return gt_dict

def widerface_to_coco():
    print("="*60)
    print("KONVERSI WIDERFACE OFFICIAL GT → COCO FORMAT (WITH FILTER)")
    print("="*60)
    
    # 1. Parse Ground Truth
    gt_dict = parse_widerface_gt_txt()
    if gt_dict is None:
        return
    
    # 2. Inisialisasi Struktur COCO
    images = []
    annotations = []
    categories = [{"id": 0, "name": "face"}] 
    
    annotation_id = 1
    image_id = 1
    
    # 3. Loop semua gambar dari GT
    print("\nMemproses gambar...")
    
    skipped_images = []
    
    for img_relative_path, bboxes in tqdm(gt_dict.items()):
        # Konstruksi path lengkap
        img_full_path = os.path.join(VAL_IMAGES_DIR, img_relative_path)
        
        # Cek apakah file gambar ada
        if not os.path.exists(img_full_path):
            skipped_images.append(img_relative_path)
            continue
        
        # Baca dimensi gambar
        img = cv2.imread(img_full_path)
        if img is None:
            skipped_images.append(img_relative_path)
            continue
        
        height, width, _ = img.shape
        
        # Tambahkan info gambar
        images.append({
            "id": image_id,
            "file_name": img_relative_path,
            "height": height,
            "width": width
        })
        
        # Tambahkan annotations
        for bbox in bboxes:
            x, y, w, h = bbox
            
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 0, 
                "bbox": [x, y, w, h], 
                "area": w * h,
                "iscrowd": 0
            })
            annotation_id += 1
        
        image_id += 1
    
    # 4. Buat output COCO JSON
    coco_output = {
        "info": {
            "description": "WiderFace Validation Set - Official GT (Filtered Invalid)",
            "url": "http://shuoyang1213.me/WIDERFACE/",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "WiderFace",
            "date_created": datetime.datetime.now().strftime("%Y/%m/%d")
        },
        "licenses": [
            {"url": "http://creativecommons.org/licenses/by/2.0/", "id": 1, "name": "Attribution License"}
        ],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    # 5. Simpan ke file
    print(f"\nMenyimpan ke {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    # 6. Report
    print("\n" + "="*60)
    print("KONVERSI SELESAI!")
    print("="*60)
    print(f"Total gambar diproses    : {len(images)}")
    print(f"Total annotations        : {len(annotations)}")
    print(f"Gambar yang di-skip      : {len(skipped_images)}")
    print(f"\nFile output tersimpan di:\n{OUTPUT_PATH}")
    
    if skipped_images and len(skipped_images) <= 10:
        print(f"\nGambar yang di-skip:")
        for img in skipped_images[:10]:
            print(f"  - {img}")
    
    print("="*60)

if __name__ == "__main__":
    widerface_to_coco()