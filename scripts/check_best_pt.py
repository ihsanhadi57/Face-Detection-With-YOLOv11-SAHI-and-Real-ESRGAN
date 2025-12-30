import os
import pandas as pd
import yaml

# ================= KONFIGURASI =================
# Sesuaikan path ini dengan posisi terminal Anda saat menjalankan script
ROOT_DIR = "models"  

# ================= FUNGSI BANTUAN =================
def get_config_info(folder_path):
    """Membaca args.yaml untuk mendapatkan batch size dan imgsz"""
    yaml_path = os.path.join(folder_path, 'args.yaml')
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                args = yaml.safe_load(f)
                # Ambil nilai, jika tidak ada isi dengan '-'
                batch = args.get('batch', '-')
                imgsz = args.get('imgsz', '-')
                return batch, imgsz
        except Exception as e:
            print(f"⚠️ Error reading YAML in {folder_path}: {e}")
            return 'Err', 'Err'
    return '-', '-'

def process_results_box_only(csv_path):
    """Mencari Epoch Terbaik berdasarkan mAP Box (B)"""
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip() # Bersihkan spasi

        # Kunci utama pencarian: Box mAP 50-95
        target_metric = 'metrics/mAP50-95(B)'

        # Cek apakah kolom Box ada
        if target_metric not in df.columns:
            return None # Skip jika ini model pose murni tanpa box (jarang terjadi) or error

        # Cari index baris dengan mAP Box tertinggi
        best_idx = df[target_metric].idxmax()
        best_row = df.iloc[best_idx]
        
        return best_row
    except Exception as e:
        print(f"⚠️ Error reading CSV {csv_path}: {e}")
        return None

# ================= LOGIKA UTAMA =================
def scan_models(root_dir):
    data_summary = []
    
    if not os.path.exists(root_dir):
        print(f"❌ Folder '{root_dir}' tidak ditemukan!")
        return

    print(f"🔍 Scanning for BOX (B) metrics & YAML config in: {root_dir}...\n")

    # Ambil list folder model utama
    model_folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    
    for model_name in model_folders:
        model_path = os.path.join(root_dir, model_name)
        # Ambil list sub-folder training
        sub_folders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
        
        for sub_name in sub_folders:
            target_path = os.path.join(model_path, sub_name)
            csv_path = os.path.join(target_path, 'results.csv')

            if os.path.exists(csv_path):
                # 1. Ambil Metrics Terbaik (BOX ONLY)
                best_row = process_results_box_only(csv_path)
                
                # 2. Ambil Config dari args.yaml
                batch, imgsz = get_config_info(target_path)
                
                if best_row is not None:
                    # Masukkan data ke list sesuai kolom yang Anda minta
                    data_summary.append({
                        "Kode / Model": model_name,
                        # "Sub Folder": sub_name,
                        "Input Size": imgsz,       # Dari args.yaml
                        "Batch": batch,            # Dari args.yaml
                        "Epoch Terbaik": int(best_row['epoch']),
                        "mAP @0.5 (Val)": f"{best_row.get('metrics/mAP50(B)', 0):.5f}",
                        "mAP @0.5:0.95": f"{best_row.get('metrics/mAP50-95(B)', 0):.5f}",
                        "Precision": f"{best_row.get('metrics/precision(B)', 0):.5f}",
                        "Recall": f"{best_row.get('metrics/recall(B)', 0):.5f}"
                    })

    # ================= CETAK HASIL =================
    if data_summary:
        df_result = pd.DataFrame(data_summary)
        
        # Urutkan berdasarkan nama model
        df_result = df_result.sort_values(by="Kode / Model")

        print("="*140)
        print(f"📊 REKAPITULASI HASIL TRAINING (BOX METRICS ONLY)")
        print("="*140)
        print(df_result.to_string(index=False))
        print("="*140)
        
        # Simpan ke CSV agar mudah dicopy
        output_file = "summary_box_metrics.csv"
        df_result.to_csv(output_file, index=False)
        print(f"\n✅ File CSV tersimpan: {output_file}")
    else:
        print("❌ Tidak ditemukan data result.csv yang valid.")

if __name__ == "__main__":
    scan_models(ROOT_DIR)