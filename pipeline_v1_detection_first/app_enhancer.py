import os
import re
import time
from sahi.predict import get_sliced_prediction
from utils.insightface_wrapper import InsightFaceDetectionModel
from utils.visualization import draw_detections, save_face_crops, create_detection_summary
from utils.enhancer import FaceEnhancer, enhance_face_crops_batch, create_enhancement_summary, get_available_models
from PIL import Image

def print_available_models():
    """Print informasi model yang tersedia"""
    models = get_available_models()
    print("\n=== Model Real-ESRGAN Tersedia ===")
    for model_name, info in models.items():
        print(f"• {model_name}")
        print(f"  - Deskripsi: {info['description']}")
        print(f"  - Scale: {info['scale']}x")
        print(f"  - Terbaik untuk: {info['best_for']}")
    print()

def main():
    # --- Konfigurasi Awal ---
    detection_model = InsightFaceDetectionModel(confidence_threshold=0.5)
    source_image_dir = "data/input"
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    test_image_name = "17_Ceremony_Ceremony_17_171.jpg"
    test_image_path = os.path.join(source_image_dir, test_image_name)

    if not os.path.exists(test_image_path):
        print(f"Error: Gambar tes tidak ditemukan di '{test_image_path}'")
        return

    print(f"Memulai pipeline deteksi dan enhancement wajah pada: {test_image_path}")
    
    # --- Persiapan Nama & Direktori Output ---
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', os.path.splitext(test_image_name)[0])
    result_dir = os.path.join(output_dir, f"result_{clean_name}")
    os.makedirs(result_dir, exist_ok=True)
    
    crops_dir = os.path.join(result_dir, "face_crops")
    visual_output_path = os.path.join(result_dir, f"visual_{clean_name}.png")
    summary_output_path = os.path.join(result_dir, "detection_summary.txt")
    enhancement_summary_path = os.path.join(result_dir, "enhancement_summary.txt")

    # --- Logika Ukuran Slice Adaptif ---
    base_slice_size = 512
    with Image.open(test_image_path) as img:
        img_width, img_height = img.size

    slice_height = img_height // 2 if img_height < base_slice_size * 1.5 else base_slice_size
    slice_width = img_width // 2 if img_width < base_slice_size * 1.5 else base_slice_size
    slice_height = max(slice_height, 1)
    slice_width = max(slice_width, 1)

    # --- TAHAP 1: DETEKSI WAJAH ---
    print("\n" + "="*50)
    print("TAHAP 1: DETEKSI WAJAH")
    print("="*50)
    
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
    print(f"✓ Deteksi selesai dalam {processing_time:.2f} detik!")
    print(f"✓ {len(result.object_prediction_list)} wajah terdeteksi")

    # --- TAHAP 2: PEMBUATAN OUTPUT & VISUALISASI ---
    print("\n" + "="*50)
    print("TAHAP 2: CROPPING & VISUALISASI")
    print("="*50)
    
    # 1. Gambar hasil deteksi
    draw_detections(test_image_path, result, visual_output_path)
    print(f"✓ Visualisasi deteksi disimpan: '{os.path.basename(visual_output_path)}'")

    # 2. Simpan potongan wajah
    saved_crops = []
    if result.object_prediction_list:
        saved_crops = save_face_crops(test_image_path, result, crops_dir, prefix=clean_name)
        print(f"✓ {len(saved_crops)} potongan wajah disimpan di: '{os.path.basename(crops_dir)}'")
    else:
        print("⚠ Tidak ada wajah untuk dipotong.")

    # 3. Buat ringkasan deteksi
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
    print(f"✓ Ringkasan deteksi disimpan: '{os.path.basename(summary_output_path)}'")
    
    # --- TAHAP 3: ENHANCEMENT DENGAN REAL-ESRGAN ---
    if saved_crops:
        print("\n" + "="*50)
        print("TAHAP 3: ENHANCEMENT DENGAN REAL-ESRGAN")
        print("="*50)
        
        # Tampilkan model yang tersedia
        print_available_models()
        
        try:
            # Konfigurasi Real-ESRGAN
            # Ubah parameter ini sesuai kebutuhan:
            ENHANCEMENT_CONFIG = {
                'model_name': 'RealESRGAN_x4plus',  # Pilih model sesuai kebutuhan
                'scale': 4,                         # Faktor skala enhancement
                'tile': 400,                        # Sesuaikan dengan VRAM GPU
                'half': True                        # Half precision untuk menghemat memori
            }
            
            print(f"Inisialisasi model enhancement:")
            print(f"• Model: {ENHANCEMENT_CONFIG['model_name']}")
            print(f"• Scale: {ENHANCEMENT_CONFIG['scale']}x")
            print(f"• Tile size: {ENHANCEMENT_CONFIG['tile']}")
            
            # Inisialisasi FaceEnhancer
            enhancer = FaceEnhancer(**ENHANCEMENT_CONFIG)
            
            # Print informasi model yang dimuat
            model_info = enhancer.get_model_info()
            print(f"✓ Model berhasil dimuat: {model_info['model_name']}")
            
            # Progress callback untuk tracking
            def progress_callback(current, total, filename):
                progress = (current / total) * 100
                print(f"  Progress: {progress:.1f}% - {filename}")
            
            # Enhancement batch semua face crops
            enhancement_results = enhance_face_crops_batch(
                crops_dir=crops_dir,
                enhancer=enhancer,
                prefix=clean_name,
                progress_callback=progress_callback
            )
            
            # Buat ringkasan enhancement
            create_enhancement_summary(enhancement_results, enhancement_summary_path)
            
            # Print hasil akhir
            stats = enhancement_results['statistics']
            if stats['successful'] > 0:
                print(f"\n🎉 Enhancement berhasil!")
                print(f"✓ {stats['successful']}/{stats['total_files']} wajah berhasil di-enhance")
                print(f"✓ Tingkat keberhasilan: {(stats['successful']/stats['total_files']*100):.1f}%")
                print(f"✓ Waktu total: {stats['total_time']:.2f} detik")
                
                # Print contoh hasil enhancement
                if enhancement_results['enhancement_info']:
                    first_info = enhancement_results['enhancement_info'][0]
                    orig_size = first_info['original_size']
                    enh_size = first_info['enhanced_size']
                    print(f"✓ Contoh enhancement: {orig_size[0]}x{orig_size[1]} → {enh_size[0]}x{enh_size[1]} pixels")
            else:
                print(f"\n⚠ Enhancement gagal untuk semua file")
                if enhancement_results['failed_files']:
                    print("File yang gagal:")
                    for failed in enhancement_results['failed_files']:
                        print(f"  • {os.path.basename(failed)}")
                
        except Exception as e:
            print(f"\n❌ Error saat enhancement: {e}")
            print("\nTroubleshooting:")
            print("• Pastikan Real-ESRGAN terinstall: pip install realesrgan")
            print("• Pastikan BasicSR terinstall: pip install basicsr")
            print("• Cek VRAM GPU jika menggunakan CUDA")
            print("• Coba kurangi tile size jika out of memory")
    
    else:
        print("\n⚠ Tidak ada face crops untuk di-enhance")
    
    # --- RINGKASAN AKHIR ---
    print("\n" + "="*50)
    print("RINGKASAN PIPELINE")
    print("="*50)
    print(f"📁 Semua hasil disimpan dalam: '{result_dir}'")
    print(f"📊 Total wajah terdeteksi: {len(result.object_prediction_list)}")
    if saved_crops:
        print(f"✂️ Face crops disimpan: {len(saved_crops)}")
        if 'enhancement_results' in locals():
            stats = enhancement_results['statistics']
            print(f"🔍 Enhanced crops: {stats['successful']}/{stats['total_files']}")
    
    print(f"\n📂 Struktur output:")
    print(f"   ├── {os.path.basename(visual_output_path)}")
    print(f"   ├── {os.path.basename(summary_output_path)}")
    if 'enhancement_results' in locals() and stats['successful'] > 0:
        print(f"   ├── {os.path.basename(enhancement_summary_path)}")
    print(f"   ├── face_crops/ ({len(saved_crops)} files)")
    if 'enhancement_results' in locals() and stats['successful'] > 0:
        enhanced_dir_name = f"{clean_name}_enhanced"
        print(f"   └── {enhanced_dir_name}/ ({stats['successful']} files)")

if __name__ == "__main__":
    main()