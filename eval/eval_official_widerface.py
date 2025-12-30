import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import cv2
import shutil
import time
import argparse

# --- Integrasi dari skrip evaluasi resmi WIDER Face ---
from scipy.io import loadmat
import sys

# Tambahkan path ke WiderFace-Evaluation untuk mengimpor utilitas bbox
# Sesuaikan path ini jika direktori WiderFace-Evaluation Anda ada di tempat lain
wider_face_eval_path = str(Path(__file__).resolve().parent.parent / 'WiderFace-Evaluation')
if wider_face_eval_path not in sys.path:
    sys.path.append(wider_face_eval_path)

try:
    from bbox import bbox_overlaps
except ImportError:
    print("="*80)
    print("ERROR: Gagal mengimpor 'bbox_overlaps'.")
    print(f"Pastikan direktori 'WiderFace-Evaluation' ada di: {wider_face_eval_path}")
    print("Dan pastikan Anda telah meng-compile Cython extension di dalamnya dengan menjalankan:")
    print(f"cd {wider_face_eval_path} && python setup.py build_ext --inplace")
    print("="*80)
    sys.exit(1)
# ---------------------------------------------------------

# --- Impor dari proyek Anda ---
sys.path.append(str(Path(__file__).parent.parent))
from sahi.predict import get_sliced_prediction
from utils.yolo_wrapper import YOLOv11PoseDetectionModel
from utils.enhancer import FaceEnhancer
# ---------------------------------------------------------


class OfficialWiderFaceEvaluator:
    def __init__(self, 
                 gt_path="data/dataset/widerface/wider_face_split",
                 images_path="data/dataset/widerface/WIDER_val/images",
                 model_path="models/yolo11s-pose-default/yolo11s_pose/weights/best.pt",
                 device='cuda:0',
                 use_sahi=True,
                 slicing_strategy='uniform',
                 use_enhancer=False,
                 bounded_enhancement=False,
                 face_size_threshold=50):
        
        self.gt_path = Path(gt_path)
        self.images_path = Path(images_path)
        
        self.settings = ['easy', 'medium', 'hard']
        self.iou_threshold = 0.5
        self.thresh_num = 1000 # Untuk kurva PR

        self.use_sahi = use_sahi
        self.use_enhancer = use_enhancer
        self.bounded_enhancement = bounded_enhancement
        self.face_size_threshold = face_size_threshold
        self.slicing_strategy = slicing_strategy
        
        self.inference_confidence = 0.01 # Confidence rendah untuk menangkap semua kemungkinan deteksi

        # Statistik enhancement
        self.enhancement_stats = defaultdict(lambda: {'enhanced': 0, 'skipped': 0, 'total': 0})
        
        self.temp_enh_dir = Path("output") / "eval_official_temp_enhanced"

        # --- Konfigurasi SAHI ---
        if self.slicing_strategy == 'uniform':
            self.sahi_config = {'slice_height': 640, 'slice_width': 640, 'overlap_ratio': 0.2}
        else: # adaptive
            self.sahi_config = {'overlap_ratio': 0.2}
        
        # --- Inisialisasi Model ---
        print("🔧 Memuat model...")
        self.detection_model = YOLOv11PoseDetectionModel(
            model_path=model_path,
            confidence_threshold=self.inference_confidence,
            device=device,
            load_at_init=True
        )
        
        self.face_enhancer = None
        if self.use_enhancer:
            try:
                self.face_enhancer = FaceEnhancer(model_name='RealESRGAN_x2plus')
                print(f"   ✓ Model Enhancer dimuat! (Skala: {self.face_enhancer.scale}x)")
            except Exception as e:
                print(f"   ❌ Gagal memuat model Enhancer: {e}. Enhancement dinonaktifkan.")
                self.use_enhancer = False
        
        self._build_mode_string()
        print(f"✓ Model dimuat! Mode: {self.mode_string}")

        # --- Memuat Ground Truth ---
        self._load_official_ground_truth()

    def _build_mode_string(self):
        mode_parts = []
        if self.use_enhancer:
            if self.bounded_enhancement:
                mode_parts.append(f"BOUNDED-ENHANCE (<{self.face_size_threshold}px)")
            else:
                mode_parts.append("FULL-ENHANCE")
        
        if self.use_sahi:
            mode_parts.append(f"SAHI ({self.slicing_strategy})")
        else:
            mode_parts.append("BASELINE")
        
        self.mode_string = " -> ".join(mode_parts)

    def _load_official_ground_truth(self):
        """Memuat ground truth dari file .mat resmi."""
        print("\n📖 Memuat ground truth dari file .mat...")
        try:
            gt_mat = loadmat(self.gt_path / 'wider_face_val.mat')
            self.facebox_list = gt_mat['face_bbx_list']
            self.event_list = gt_mat['event_list']
            self.file_list = gt_mat['file_list']

            hard_mat = loadmat(self.gt_path / 'wider_hard_val.mat')
            medium_mat = loadmat(self.gt_path / 'wider_medium_val.mat')
            easy_mat = loadmat(self.gt_path / 'wider_easy_val.mat')
            
            self.setting_gts = {
                'easy': easy_mat['gt_list'],
                'medium': medium_mat['gt_list'],
                'hard': hard_mat['gt_list']
            }
            print(f"✓ Ground truth untuk {len(self.event_list)} event berhasil dimuat.")
        except FileNotFoundError as e:
            print(f"❌ Error: File ground truth .mat tidak ditemukan di {self.gt_path}.")
            print(f"Detail: {e}")
            sys.exit(1)

    # --- Logika Pipeline Inferensi (diadaptasi dari skrip Anda) ---
    
    def _quick_face_analysis(self, img):
        if img is None: return False, "Image load failed", {}
        results = self.detection_model.model(img, conf=0.05, verbose=False)
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return True, "No faces detected", {}
        
        face_sizes = [max(box[2]-box[0], box[3]-box[1]) for box in results[0].boxes.xyxy.cpu().numpy()]
        small_face_ratio = sum(1 for s in face_sizes if s < self.face_size_threshold) / len(face_sizes)
        
        if small_face_ratio > 0.5 or np.mean(face_sizes) < self.face_size_threshold:
            return True, f"Small faces detected (ratio: {small_face_ratio:.2f})", {}
        return False, "Faces are large enough", {}

    def _get_slice_size_adaptive(self, w, h):
        max_dim = max(w, h)
        if max_dim > 2500: return 512
        if max_dim > 1500: return 416
        return 320

    def _run_single_inference(self, img_path):
        """Menjalankan inferensi (termasuk enhancement dan SAHI) pada satu gambar."""
        img = cv2.imread(img_path)
        if img is None: return np.array([])
        
        # Logika pipeline: Enhance -> SAHI -> Detect
        inference_img = img
        was_enhanced = False

        # --- 1. PHASE ENHANCEMENT ---
        if self.use_enhancer and self.face_enhancer:
            enhance_decision = False
            if self.bounded_enhancement:
                enhance_decision, _, _ = self._quick_face_analysis(img)
            else: # Full enhancement
                enhance_decision = True
            
            if enhance_decision:
                enhanced_image, success = self.face_enhancer.enhance_image(img)
                if success:
                    inference_img = enhanced_image
                    was_enhanced = True
        
        # --- 2. PHASE DETECTION ---
        pred_boxes = np.array([]) # Default kosong

        if self.use_sahi:
            # --- Mode SAHI ---
            h, w = inference_img.shape[:2]
            if self.slicing_strategy == 'uniform':
                slice_h, slice_w = self.sahi_config['slice_height'], self.sahi_config['slice_width']
            else: # adaptive
                slice_h = slice_w = self._get_slice_size_adaptive(w, h)

            result = get_sliced_prediction(
                inference_img, self.detection_model,
                slice_height=slice_h, slice_width=slice_w,
                overlap_height_ratio=self.sahi_config['overlap_ratio'],
                overlap_width_ratio=self.sahi_config['overlap_ratio'],
                postprocess_type='NMS', postprocess_match_threshold=0.5,
                postprocess_class_agnostic=True, verbose=0
            )
            pred_list = result.object_prediction_list
            
            if len(pred_list) > 0:
                pred_boxes = np.array([[*det.bbox.to_xywh(), det.score.value] for det in pred_list])
            else:
                pred_boxes = np.array([])

        else: 
            # ============================================================================
            # ✅ PERBAIKAN DI SINI - Mode Standard (YOLO Native)
            # ============================================================================
            results = self.detection_model.model(inference_img, conf=self.inference_confidence, verbose=False)
            boxes = results[0].boxes
            
            if len(boxes) > 0:
                # ❌ JANGAN GUNAKAN INI (FORMAT SALAH):
                # xywh = boxes.xywh.cpu().numpy()  # [x_center, y_center, w, h] ❌
                
                # ✅ GUNAKAN INI (FORMAT BENAR):
                # 1. Ambil koordinat xyxy (x1, y1, x2, y2)
                xyxy = boxes.xyxy.cpu().numpy()
                
                # 2. Hitung TopLeft-X, TopLeft-Y, Width, Height
                x1 = xyxy[:, 0]  # x_topleft
                y1 = xyxy[:, 1]  # y_topleft
                w = xyxy[:, 2] - x1  # width
                h = xyxy[:, 3] - y1  # height
                
                # 3. Gabungkan menjadi [x_topleft, y_topleft, width, height]
                xywh_topleft = np.column_stack((x1, y1, w, h))
                
                # 4. Tambahkan confidence score di kolom ke-5
                confs = boxes.conf.cpu().numpy()
                pred_boxes = np.column_stack((xywh_topleft, confs))
            else:
                pred_boxes = np.array([])

        # --- 3. PHASE POST-PROCESSING (Scaling) ---
        # Jika gambar di-enhance (di-upscale), kembalikan koordinat bbox ke ukuran asli
        if was_enhanced and self.face_enhancer.scale > 1 and len(pred_boxes) > 0:
            scale = self.face_enhancer.scale
            # Bagi x, y, w, h dengan scale factor
            pred_boxes[:, :4] /= scale
            
        return pred_boxes.astype('float')

    def _run_inference_on_all_images(self):
        """Menjalankan inferensi di semua gambar dan menyimpan hasilnya."""
        print(f"\n🚀 Menjalankan inferensi mode '{self.mode_string}' di seluruh dataset...")
        predictions = defaultdict(dict)
        
        pbar = tqdm(total=len(self.file_list), desc="Inferensi")
        for i, event in enumerate(self.event_list):
            event_name = event[0][0]
            img_list = self.file_list[i][0]
            
            for j, img_file in enumerate(img_list):
                img_name = img_file[0][0]
                img_path = str(self.images_path / event_name / f"{img_name}.jpg")
                
                if not os.path.exists(img_path):
                    pbar.update(1)
                    continue

                # Lakukan inferensi
                pred_boxes = self._run_single_inference(img_path)
                predictions[event_name][img_name] = pred_boxes
                pbar.update(1)
        pbar.close()
        print("✓ Inferensi selesai.")
        return predictions

    # --- Logika Evaluasi (diadaptasi dari skrip resmi) ---

    def _voc_ap(self, rec, prec):
        """
        Calculate VOC AP given precision and recall.
        Code adapted from the official WIDER Face evaluation script.
        """
        # Append sentinel values
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # To calculate area under PR curve, look for points where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # And sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def _image_eval(self, pred, gt, ignore):
        """
        Evaluate a single image's predictions.
        EXACT COPY from official WIDER Face evaluation script.
        
        Args:
            pred: Nx5 array [x, y, w, h, score]
            gt: Nx4 array [x, y, w, h]
            ignore: N array where:
                    ignore[i] == 0: this GT should be IGNORED (don't evaluate)
                    ignore[i] == 1: this GT should be EVALUATED
        
        Returns:
            pred_recall: cumulative number of GT matched at each prediction
            proposal_list: 1 = valid TP, 0 = FP, -1 = ignored
        """
        _pred = pred.copy()
        _gt = gt.copy()
        pred_recall = np.zeros(_pred.shape[0])
        recall_list = np.zeros(_gt.shape[0])
        proposal_list = np.ones(_pred.shape[0])

        # Convert from xywh to xyxy
        _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
        _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
        _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
        _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

        overlaps = bbox_overlaps(_pred[:, :4], _gt)

        for h in range(_pred.shape[0]):
            gt_overlap = overlaps[h]
            max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
            
            if max_overlap >= self.iou_threshold:
                if ignore[max_idx] == 0:
                    # This GT should be ignored, so mark both as ignored
                    recall_list[max_idx] = -1
                    proposal_list[h] = -1
                elif recall_list[max_idx] == 0:
                    # This is a valid match (True Positive)
                    recall_list[max_idx] = 1

            # Cumulative count of matched GT
            r_keep_index = np.where(recall_list == 1)[0]
            pred_recall[h] = len(r_keep_index)
        
        return pred_recall, proposal_list

    def _img_pr_info(self, pred_info, proposal_list, pred_recall):
        """
        Calculate PR curve info for a single image.
        EXACT COPY from official WIDER Face evaluation script.
        
        Returns:
            pr_info: Array of shape (thresh_num, 2) where:
                    pr_info[t, 0] = number of proposals (TP) at threshold t
                    pr_info[t, 1] = cumulative recall count at threshold t
        """
        pr_info = np.zeros((self.thresh_num, 2)).astype('float')
        
        for t in range(self.thresh_num):
            thresh = 1 - (t + 1) / self.thresh_num
            r_index = np.where(pred_info[:, 4] >= thresh)[0]
            
            if len(r_index) == 0:
                pr_info[t, 0] = 0
                pr_info[t, 1] = 0
            else:
                r_index = r_index[-1]
                # Count only valid proposals (proposal_list == 1 means TP)
                p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
                pr_info[t, 0] = len(p_index)
                pr_info[t, 1] = pred_recall[r_index]
        
        return pr_info

    def _dataset_pr_info(self, pr_curve, count_face):
        """
        Calculate final PR curve for the whole dataset.
        EXACT COPY from official WIDER Face evaluation script.
        """
        _pr_curve = np.zeros((self.thresh_num, 2))
        
        for i in range(self.thresh_num):
            # Precision = TP / total_proposals
            if pr_curve[i, 0] != 0:
                _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
            else:
                _pr_curve[i, 0] = 0
            
            # Recall = TP / total_GT
            _pr_curve[i, 1] = pr_curve[i, 1] / count_face
        
        return _pr_curve

    def _evaluate_setting(self, setting, all_predictions):
        """
        Evaluate one setting (easy, medium, hard).
        EXACT LOGIC from official WIDER Face evaluation script.
        """
        print(f"\n🔍 Mengevaluasi setting: {setting.upper()}...")
        gt_list = self.setting_gts[setting]
        count_face = 0
        pr_curve = np.zeros((self.thresh_num, 2), dtype=float)

        pbar = tqdm(range(len(self.event_list)), desc=f"Eval {setting.upper()}")
        for i in pbar:
            event_name = self.event_list[i][0][0]
            img_list = self.file_list[i][0]
            pred_list_event = all_predictions.get(event_name, {})
            sub_gt_list = gt_list[i][0]
            gt_bbx_list = self.facebox_list[i][0]

            for j in range(len(img_list)):
                img_name = img_list[j][0][0]
                pred_info = pred_list_event.get(img_name, np.array([]))
                
                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                
                # Count faces to evaluate
                count_face += len(keep_index)
                
                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue

                # ✓ CRITICAL: ignore array following official logic
                # ignore[i] = 0: don't evaluate this GT
                # ignore[i] = 1: DO evaluate this GT
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1
                
                # Evaluate image
                pred_recall, proposal_list = self._image_eval(
                    pred_info.copy(), gt_boxes.copy(), ignore)
                
                # Calculate PR info for this image
                _img_pr_info = self._img_pr_info(pred_info, proposal_list, pred_recall)
                pr_curve += _img_pr_info
        
        pbar.close()
        
        # Calculate final PR curve
        pr_curve = self._dataset_pr_info(pr_curve, count_face)

        propose = pr_curve[:, 0]  # Precision
        recall = pr_curve[:, 1]   # Recall

        ap = self._voc_ap(recall, propose)
        return ap, recall, propose
    
    def run(self):
        """Menjalankan seluruh alur inferensi dan evaluasi."""
        
        # --- 1. INFERENSI ---
        all_predictions = self._run_inference_on_all_images()

        # --- 2. EVALUASI ---
        results = {}
        fig, ax = plt.subplots(figsize=(10, 8))

        for setting in self.settings:
            ap, recall, propose = self._evaluate_setting(setting, all_predictions)
            results[setting] = ap
            ax.plot(recall, propose, lw=2, label=f'{setting.capitalize()} AP: {ap:.4f}')

        # --- 3. TAMPILKAN HASIL ---
        print("\n" + "="*50)
        print(f"📊 HASIL EVALUASI - {self.mode_string}")
        print("="*50)
        for setting, ap in results.items():
            print(f"  - {setting.capitalize():<7} AP: {ap:.4f}")
        print("="*50)

        # --- 4. VISUALISASI ---
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve\n{self.mode_string}')
        ax.grid(True)
        ax.legend(loc='lower left')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        filename_prefix = self.mode_string.lower().replace(" -> ", "_").replace(" ", "_")
        filename_prefix = "".join(c for c in filename_prefix if c.isalnum() or c in ('_', '-')).rstrip()
        
        plot_filename = output_dir / f"pr_curve_{filename_prefix}.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"\n✓ Kurva PR disimpan di: {plot_filename}")

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate YOLO on WIDER Face using official metrics.')
    
    parser.add_argument('--gt_path', type=str, default="data/dataset/widerface/wider_face_split", help='Path to directory with ground truth .mat files.')
    parser.add_argument('--images_path', type=str, default="data/dataset/widerface/WIDER_val/images", help='Path to WIDER FACE validation images directory.')
    parser.add_argument('--model_path', type=str, default="models/yolo11s-pose-default/yolo11s_pose/weights/best.pt", help='Path to the YOLO model file.')
    
    # Argumen untuk memilih pipeline
    parser.add_argument('--mode', 
                       choices=['standard', 'sahi-uniform', 'sahi-adaptive'], 
                       default='sahi-uniform',
                       help='Inference mode: standard (no SAHI), sahi-uniform, or sahi-adaptive.')
    parser.add_argument('--enhance', action='store_true', help='Enable enhancement pipeline (can be combined with SAHI).')
    parser.add_argument('--bounded', action='store_true', help='Enable Bounded Enhancement (selective, only if --enhance is used).')
    parser.add_argument('--threshold', type=int, default=50, help='Face size threshold for bounded enhancement (e.g., 50px).')

    parser.add_argument('--device', default='cuda:0', help='Device to run inference on (e.g., cuda:0 or cpu).')
    
    args = parser.parse_args()

    # Tentukan pipeline berdasarkan argumen --mode dan --enhance
    use_sahi_arg = args.mode.startswith('sahi')
    slicing_strategy_arg = args.mode.split('-')[1] if use_sahi_arg else 'uniform'
    
    use_enhancer_arg = args.enhance
    bounded_enhancement_arg = args.bounded if use_enhancer_arg else False

    # Buat evaluator
    evaluator = OfficialWiderFaceEvaluator(
        gt_path=args.gt_path,
        images_path=args.images_path,
        model_path=args.model_path,
        device=args.device,
        use_sahi=use_sahi_arg,
        slicing_strategy=slicing_strategy_arg,
        use_enhancer=use_enhancer_arg,
        bounded_enhancement=bounded_enhancement_arg,
        face_size_threshold=args.threshold,
    )
    
    # Jalankan evaluasi
    evaluator.run()
