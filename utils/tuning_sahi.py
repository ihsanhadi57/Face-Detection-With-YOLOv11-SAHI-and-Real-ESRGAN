import os
import json
import itertools
import io
from contextlib import redirect_stdout
from tqdm import tqdm

# Library SAHI & COCO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ================= KONFIGURASI PATH =================
BASE_DIR = os.path.join("data", "dataset", "widerface")
VAL_IMAGES_DIR = os.path.join(BASE_DIR, "WIDER_val", "images")
VAL_JSON_PATH = os.path.join(BASE_DIR, "val_ground_truth_coco.json")
MODEL_PATH = "models/yolo11s-pose-default/yolo11s_pose/weights/best.pt" 

# ================= GRID SEARCH LENGKAP =================
# Pilih salah satu: QUICK, BALANCED, atau COMPREHENSIVE

# QUICK - Testing cepat (~12 kombinasi, ~30-60 menit)
param_grid_quick = {
    'slice_size': [512, 640],
    'overlap_ratio': [0.2, 0.3],
    'postprocess_type': ['NMS'],
    'postprocess_match_metric': ['IOS'],  # Lebih baik untuk face detection
    'postprocess_match_threshold': [0.5]
}

# BALANCED - Recommended untuk tuning standar (~48 kombinasi, ~2-4 jam)
param_grid_balanced = {
    'slice_size': [320, 512, 640],
    'overlap_ratio': [0.2, 0.25, 0.3],
    'postprocess_type': ['NMS', 'GREEDYNMM'],
    'postprocess_match_metric': ['IOS', 'IOU'],
    'postprocess_match_threshold': [0.5]
}

# COMPREHENSIVE - Grid search menyeluruh (~240 kombinasi, ~8-12 jam)
param_grid_comprehensive = {
    'slice_size': [320, 512, 640, 800],
    'overlap_ratio': [0.1, 0.2, 0.25, 0.3, 0.4],
    'postprocess_type': ['NMS'],
    'postprocess_match_metric': ['IOS', 'IOU'],
    'postprocess_match_threshold': [0.3, 0.5, 0.7]
}

# PILIH GRID (ubah sesuai kebutuhan)
param_grid = param_grid_comprehensive  # ← GANTI INI

# ================= FUNGSI EVALUASI =================
def evaluate_sahi(model, params):
    """
    Evaluasi SAHI dengan parameter lengkap
    
    Args:
        model: SAHI AutoDetectionModel
        params: Dict dengan keys:
            - slice_size
            - overlap_ratio
            - postprocess_type
            - postprocess_match_metric
            - postprocess_match_threshold
    """
    from tqdm import tqdm
    
    # Load Ground Truth
    try:
        coco_gt = COCO(VAL_JSON_PATH)
    except Exception as e:
        tqdm.write(f"❌ ERROR: Gagal membaca {VAL_JSON_PATH}: {e}")
        return 0.0

    img_ids = coco_gt.getImgIds()
    results = []
    
    # Loop Inference
    skipped_count = 0
    pbar = tqdm(img_ids, 
                desc=f"S{params['slice_size']}|O{params['overlap_ratio']:.2f}|{params['postprocess_type'][:3]}|{params['postprocess_match_metric']}", 
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for img_id in pbar:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(VAL_IMAGES_DIR, img_info['file_name'])
        
        if not os.path.exists(img_path):
            skipped_count += 1
            continue
        
        try:
            # INFERENSI SAHI dengan parameter lengkap
            result = get_sliced_prediction(
                img_path,
                model,
                slice_height=params['slice_size'],
                slice_width=params['slice_size'],
                overlap_height_ratio=params['overlap_ratio'],
                overlap_width_ratio=params['overlap_ratio'],
                postprocess_type=params['postprocess_type'],
                postprocess_match_metric=params['postprocess_match_metric'],
                postprocess_match_threshold=params['postprocess_match_threshold'],
                postprocess_class_agnostic=True,  # Untuk single-class (face)
                verbose=0
            )
            
            # Konversi ke COCO format
            for object_prediction in result.object_prediction_list:
                bbox = object_prediction.bbox.to_xywh()
                score = object_prediction.score.value
                
                results.append({
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": bbox,
                    "score": score
                })
            
            # Update progress bar with detection count
            pbar.set_postfix({'detections': len(results), 'skipped': skipped_count}, refresh=False)
        except Exception as e:
            skipped_count += 1
            continue

    if skipped_count > 0:
        tqdm.write(f"⚠️  Skipped {skipped_count} images")

    if not results:
        tqdm.write("❌ No detections")
        return 0.0

    # Simpan hasil sementara
    res_file = f"temp_results_{params['slice_size']}_{params['overlap_ratio']}_{params['postprocess_type']}.json"
    with open(res_file, 'w') as f:
        json.dump(results, f)

    # Hitung mAP
    try:
        coco_dt = coco_gt.loadRes(res_file)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        
        with redirect_stdout(io.StringIO()):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        
        # Cleanup
        if os.path.exists(res_file):
            os.remove(res_file)
        
        # Extract metrics
        map_50_95 = coco_eval.stats[0]  # mAP@[0.5:0.95]
        map_50 = coco_eval.stats[1]     # mAP@0.5
        map_75 = coco_eval.stats[2]     # mAP@0.75
        
        tqdm.write(f"✅ mAP@50-95: {map_50_95:.4f} | mAP@50: {map_50:.4f} | mAP@75: {map_75:.4f}")
        
        return {
            'map_50_95': map_50_95,
            'map_50': map_50,
            'map_75': map_75
        }
        
    except Exception as e:
        tqdm.write(f"❌ Error: {e}")
        return 0.0

# ================= MAIN PROGRAM =================
def main():
    print("="*70)
    print("COMPREHENSIVE SAHI TUNING - WIDERFACE")
    print("="*70)
    
    # Validasi
    if not os.path.exists(VAL_JSON_PATH):
        print(f"\n❌ ERROR: {VAL_JSON_PATH} tidak ditemukan!")
        print("Jalankan: python convert_widerface_official_to_coco.py")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model tidak ditemukan: {MODEL_PATH}")
        return
    
    # Load Model
    print(f"\n📦 Loading model: {MODEL_PATH}")
    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=MODEL_PATH,
            confidence_threshold=0.25,  # Bisa di-tuning juga kalau mau
            device="cuda:0"
        )
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Buat kombinasi parameter
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n🔍 Total kombinasi: {len(combinations)}")
    print(f"📊 Estimasi waktu: ~{len(combinations) * 2} menit (tergantung hardware)")
    print("="*70)
    
    # Grid Search
    best_map = 0
    best_params = {}
    history = []

    for idx, params in enumerate(combinations, 1):
        tqdm.write(f"\n{'='*70}")
        tqdm.write(f"[{idx}/{len(combinations)}] Slice:{params['slice_size']} Over:{params['overlap_ratio']} "
                   f"Post:{params['postprocess_type']} Metric:{params['postprocess_match_metric']} Thresh:{params['postprocess_match_threshold']}")
        tqdm.write(f"{'='*70}")
        
        metrics = evaluate_sahi(detection_model, params)
        
        # Handle jika evaluate_sahi return 0.0 (error case)
        if isinstance(metrics, dict):
            current_map = metrics['map_50_95']
        else:
            current_map = 0.0
            metrics = {'map_50_95': 0.0, 'map_50': 0.0, 'map_75': 0.0}
        
        history.append({
            "params": params, 
            "metrics": metrics
        })

        if current_map > best_map:
            best_map = current_map
            best_params = params
            tqdm.write(f"🎯 NEW BEST! mAP@50-95: {best_map:.4f}\n")

    # Report Akhir
    print("\n" + "="*70)
    print("HASIL AKHIR - TOP 10 KONFIGURASI")
    print("="*70)
    
    # Sort by mAP
    history_sorted = sorted(history, key=lambda x: x['metrics']['map_50_95'], reverse=True)
    
    print(f"{'Rank':<5} {'Slice':<7} {'Over':<6} {'Post':<10} {'Metric':<6} {'Thresh':<7} {'mAP@50-95':<10}")
    print("-" * 70)
    
    for rank, h in enumerate(history_sorted[:10], 1):
        p = h['params']
        m = h['metrics']
        mark = "🏆" if rank == 1 else f"{rank}."
        print(f"{mark:<5} {p['slice_size']:<7} {p['overlap_ratio']:<6.2f} "
              f"{p['postprocess_type']:<10} {p['postprocess_match_metric']:<6} "
              f"{p['postprocess_match_threshold']:<7.2f} {m['map_50_95']:<10.4f}")
    
    print("-" * 70)
    
    # Best Config Detail
    if best_params:
        print(f"\n🏆 KONFIGURASI TERBAIK:")
        print(f"   Slice Size               : {best_params['slice_size']}")
        print(f"   Overlap Ratio            : {best_params['overlap_ratio']}")
        print(f"   Post-process Type        : {best_params['postprocess_type']}")
        print(f"   Post-process Metric      : {best_params['postprocess_match_metric']}")
        print(f"   Post-process Threshold   : {best_params['postprocess_match_threshold']}")
        print(f"   mAP@50-95                : {best_map:.4f}")
    
    print("="*70)
    
    # Simpan hasil
    output_file = "sahi_tuning_complete_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "best_params": best_params,
            "best_metrics": history_sorted[0]['metrics'] if history_sorted else {},
            "all_results": history
        }, f, indent=2)
    
    print(f"\n💾 Hasil lengkap: {output_file}")
    
    # Simpan config untuk dipakai di inference
    config_file = "best_sahi_config.json"
    with open(config_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"⚙️  Best config : {config_file}")

if __name__ == "__main__":
    main()