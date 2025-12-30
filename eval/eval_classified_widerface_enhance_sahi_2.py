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

# Import SAHI dan wrapper/enhancer kita
import sys
sys.path.append(str(Path(__file__).parent.parent))
from sahi.predict import get_sliced_prediction
from utils.yolo_wrapper import YOLOv11PoseDetectionModel
from utils.enhancer import FaceEnhancer

class SAHIWiderFaceEvaluator:
    def __init__(self, 
                 base_path="data/dataset/widerface", 
                 model_path="models/yolo11s-pose-default/yolo11s_pose/weights/best.pt",
                 device='cuda:0',
                 use_sahi=True,
                 use_enhancer=False,
                 bounded_enhancement=False,  # NEW: Flag untuk bounded enhancement
                 face_size_threshold=50,      # NEW: Threshold ukuran face
                 slicing_strategy='uniform'):
        
        self.base_path = Path(base_path)
        self.classified_path = self.base_path / "classified_val"
        self.label_file = self.base_path / "wider_face_split" / "wider_face_val_bbx_gt.txt"
        
        self.categories = ['small_clear', 'small_degraded', 'medium_large']
        
        self.iou_threshold = 0.5
        self.global_confidence = 0.25
        self.inference_confidence = 0.01 if use_sahi else 0.5
        
        self.use_sahi = use_sahi
        self.use_enhancer = use_enhancer
        self.bounded_enhancement = bounded_enhancement  # NEW
        self.face_size_threshold = face_size_threshold  # NEW
        self.slicing_strategy = slicing_strategy
        
        # Statistik untuk bounded enhancement
        self.enhancement_stats = {
            'total_images': 0,
            'enhanced_images': 0,
            'skipped_images': 0,
            'enhancement_decisions': []
        }
        
        # Direktori temporer untuk gambar yang di-enhance
        self.temp_enh_dir = Path("output") / "eval_temp_enhanced"
        
        # ===== SAHI CONFIGURATION =====
        if slicing_strategy == 'uniform':
            self.sahi_config = {
                'slice_height': 640, 'slice_width': 640, 'overlap_ratio': 0.2,
                'confidence_threshold': 0.01, 'postprocess_match_threshold': 0.5,
                'postprocess_class_threshold': 0.25, 'postprocess_type': 'NMS',
                'description': 'Uniform 640x640 slicing'
            }
            print(f"\n📐 Slicing Strategy: UNIFORM (640x640, overlap=0.2)")
        elif slicing_strategy == 'adaptive':
            self.sahi_config = {
                'confidence_threshold': 0.01, 'postprocess_match_threshold': 0.5,
                'postprocess_class_threshold': 0.25, 'postprocess_type': 'NMS',
                'overlap_ratio': 0.2, 'description': 'Adaptive slicing'
            }
            print(f"\n📐 Slicing Strategy: ADAPTIVE (overlap=0.2)")
        
        # ===== INISIALISASI MODEL =====
        print(f"🔧 Loading models...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.detection_model = YOLOv11PoseDetectionModel(
            model_path=model_path,
            confidence_threshold=self.inference_confidence,
            device=device,
            load_at_init=True
        )
        
        self.face_enhancer = None
        if self.use_enhancer:
            print("   - Loading Real-ESRGAN model...")
            try:
                self.face_enhancer = FaceEnhancer(model_name='RealESRGAN_x2plus')
                print(f"   ✓ Enhancer model loaded! (Scale: {self.face_enhancer.scale}x)")
            except Exception as e:
                print(f"   ❌ Gagal memuat model Enhancer: {e}. Nonaktifkan enhancement.")
                self.use_enhancer = False

        mode_str = "SAHI" if use_sahi else "Standard"
        if self.use_enhancer:
            if self.bounded_enhancement:
                mode_str = f"BoundedEnhance({self.face_enhancer.scale}x, <{face_size_threshold}px) -> {mode_str}"
            else:
                mode_str = f"Enhance({self.face_enhancer.scale}x) -> {mode_str}"
            
        print(f"✓ Models loaded! Mode: {mode_str}")
        print(f"  - Inference Conf: {self.inference_confidence}")
        print(f"  - Eval Conf: {self.global_confidence}")
        if self.bounded_enhancement:
            print(f"  - Face Size Threshold: {self.face_size_threshold}px")
        
    def quick_face_analysis(self, img_path):
        """
        Analisis CEPAT untuk menentukan apakah gambar perlu di-enhance
        Returns: (should_enhance: bool, reason: str, face_info: dict)
        """
        img = cv2.imread(img_path)
        if img is None:
            return False, "Image load failed", {}
        
        img_height, img_width = img.shape[:2]
        
        # 1. Quick inference dengan confidence LEBIH TINGGI untuk akurasi
        results = self.detection_model.model(img, conf=0.05, verbose=False)  # Changed from 0.01 to 0.05
        
        # 2. Jika tidak detect face sama sekali
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return True, "No faces detected (might be too small)", {
                'num_faces': 0,
                'avg_size': 0,
                'small_ratio': 1.0,
                'resolution': (img_width, img_height)
            }
        
        # 3. Analisis ukuran face yang terdeteksi
        face_sizes = []
        face_boxes = []
        
        for i in range(len(results[0].boxes)):
            xyxy = results[0].boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            width = x2 - x1
            height = y2 - y1
            max_dim = max(width, height)
            face_sizes.append(max_dim)
            face_boxes.append([x1, y1, width, height])
        
        num_faces = len(face_sizes)
        avg_face_size = np.mean(face_sizes)
        min_face_size = np.min(face_sizes)
        max_face_size = np.max(face_sizes)
        
        # 4. Hitung rasio face kecil
        small_faces_count = sum(1 for size in face_sizes if size < self.face_size_threshold)
        small_face_ratio = small_faces_count / num_faces if num_faces > 0 else 0
        
        face_info = {
            'num_faces': num_faces,
            'avg_size': avg_face_size,
            'min_size': min_face_size,
            'max_size': max_face_size,
            'small_ratio': small_face_ratio,
            'small_count': small_faces_count,
            'resolution': (img_width, img_height)
        }
        
        # 5. Decision Logic
        # Rule 1: Jika >50% face kecil → ENHANCE
        if small_face_ratio > 0.5:
            return True, f"{small_face_ratio*100:.0f}% faces are small (<{self.face_size_threshold}px)", face_info
        
        # Rule 2: Jika rata-rata face size kecil → ENHANCE
        if avg_face_size < self.face_size_threshold:
            return True, f"Average face size ({avg_face_size:.1f}px) below threshold", face_info
        
        # Rule 3: Jika ada face sangat kecil (<30px) → ENHANCE
        if min_face_size < 30:
            return True, f"Found very small face ({min_face_size:.1f}px)", face_info
        
        # Rule 4: Jika semua face besar → SKIP
        return False, f"All faces are large enough (avg={avg_face_size:.1f}px)", face_info
    
    def parse_ground_truth(self):
        """Parse ground truth dari label file"""
        print("\n📖 Parsing ground truth...")
        with open(self.label_file, 'r') as f: 
            lines = f.readlines()
        
        gt_annotations = {}
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line: 
                i += 1
                continue
            
            if line.endswith('.jpg') or line.endswith('.png'):
                img_path = line
                i += 1
                if i >= len(lines): 
                    break
                
                next_line = lines[i].strip()
                parts = next_line.split()
                faces = []
                
                try:
                    if len(parts) == 1:
                        num_faces = int(parts[0])
                        i += 1
                        for j in range(num_faces):
                            if i >= len(lines): 
                                break
                            parts = lines[i].strip().split()
                            if len(parts) >= 4:
                                x, y, w, h = int(float(parts[0])), int(float(parts[1])), int(float(parts[2])), int(float(parts[3]))
                                if w > 0 and h > 0: 
                                    faces.append({'bbox': [x, y, w, h]})
                            i += 1
                    
                    if faces: 
                        gt_annotations[img_path] = faces
                        
                except ValueError: 
                    i += 1
                    continue
            else: 
                i += 1
        
        print(f"✓ Loaded ground truth for {len(gt_annotations)} images")
        return gt_annotations
    
    def get_slice_size_adaptive(self, img_width, img_height):
        """Adaptive slice - OPTIMIZED untuk face detection"""
        max_dim = max(img_width, img_height)
        if max_dim > 2500: 
            slice_size = 512     
        elif max_dim > 1500: 
            slice_size = 416
        elif max_dim > 800: 
            slice_size = 416
        elif max_dim > 400: 
            slice_size = 320
        else: 
            slice_size = 320
        return (slice_size // 32) * 32
    
    def run_sahi_inference(self, img_path, category):
        """Run SAHI inference dengan konfigurasi yang konsisten"""
        img = cv2.imread(img_path)
        if img is None: 
            return []
        
        img_height, img_width = img.shape[:2]
        
        if not self.use_sahi:
            results = self.detection_model.model(img, conf=self.inference_confidence, verbose=False)
            pred_boxes = []
            if len(results) > 0 and results[0].boxes is not None:
                for i in range(len(results[0].boxes)):
                    xyxy = results[0].boxes.xyxy[i].cpu().numpy()
                    conf = float(results[0].boxes.conf[i].cpu().numpy())
                    x1, y1, x2, y2 = xyxy
                    pred_boxes.append({
                        'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)], 
                        'confidence': conf
                    })
            return pred_boxes
        
        if self.slicing_strategy == 'uniform':
            slice_height, slice_width = self.sahi_config['slice_height'], self.sahi_config['slice_width']
        else:
            slice_size = self.get_slice_size_adaptive(img_width, img_height)
            slice_height, slice_width = slice_size, slice_size
        
        if img_width < slice_width * 0.8 and img_height < slice_height * 0.8:
            return self.run_sahi_inference(img_path, category)
        
        self.detection_model.confidence_threshold = self.sahi_config['confidence_threshold']
        result = get_sliced_prediction(
            img_path, self.detection_model,
            slice_height=slice_height, slice_width=slice_width,
            overlap_height_ratio=self.sahi_config['overlap_ratio'],
            overlap_width_ratio=self.sahi_config['overlap_ratio'],
            postprocess_type=self.sahi_config['postprocess_type'],
            postprocess_match_threshold=self.sahi_config['postprocess_match_threshold'],
            postprocess_class_agnostic=True, verbose=0
        )
        
        pred_boxes = []
        for detection in result.object_prediction_list:
            x1, y1, x2, y2 = detection.bbox.to_xyxy()
            pred_boxes.append({
                'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)], 
                'confidence': detection.score.value
            })
        
        return pred_boxes
    
    def run_inference_on_images(self, category):
        """
        Run inference pada semua gambar dalam kategori, 
        dengan BOUNDED ENHANCEMENT (selective enhancement)
        """
        mode_str = "SAHI" if self.use_sahi else "Standard"
        if self.use_enhancer:
            if self.bounded_enhancement:
                mode_str = f"BoundedEnhance -> {mode_str}"
            else:
                mode_str = f"Enhance -> {mode_str}"
        
        print(f"\n🚀 Running {mode_str} inference on {category}...")
        
        category_path = self.classified_path / category
        image_files = list(category_path.glob("*.jpg"))
        
        predictions = {}
        total_faces_detected = 0
        
        # Progress tracking
        category_stats = {
            'enhanced': 0,
            'skipped': 0,
            'total': len(image_files)
        }
        
        for img_file in tqdm(image_files, desc=f"{mode_str} {category}"):
            original_img_path = str(img_file)
            inference_path = original_img_path
            was_enhanced = False
            
            self.enhancement_stats['total_images'] += 1
            
            # --- BOUNDED ENHANCEMENT LOGIC ---
            if self.use_enhancer and self.face_enhancer and self.bounded_enhancement:
                # Quick analysis untuk tentukan enhance atau tidak
                should_enhance, reason, face_info = self.quick_face_analysis(original_img_path)
                
                # Simpan decision untuk statistik
                decision_record = {
                    'image': img_file.name,
                    'category': category,
                    'should_enhance': should_enhance,
                    'reason': reason,
                    'face_info': face_info
                }
                self.enhancement_stats['enhancement_decisions'].append(decision_record)
                
                if should_enhance:
                    # Enhance gambar
                    image = cv2.imread(original_img_path)
                    if image is not None:
                        enhanced_image, success = self.face_enhancer.enhance_image(image)
                        if success:
                            temp_img_name = f"enhanced_{img_file.name}"
                            enhanced_img_path = str(self.temp_enh_dir / temp_img_name)
                            cv2.imwrite(enhanced_img_path, enhanced_image)
                            inference_path = enhanced_img_path
                            was_enhanced = True
                            category_stats['enhanced'] += 1
                            self.enhancement_stats['enhanced_images'] += 1
                else:
                    category_stats['skipped'] += 1
                    self.enhancement_stats['skipped_images'] += 1
            
            # --- FULL ENHANCEMENT (Non-Bounded) ---
            elif self.use_enhancer and self.face_enhancer and not self.bounded_enhancement:
                image = cv2.imread(original_img_path)
                if image is not None:
                    enhanced_image, success = self.face_enhancer.enhance_image(image)
                    if success:
                        temp_img_name = f"enhanced_{img_file.name}"
                        enhanced_img_path = str(self.temp_enh_dir / temp_img_name)
                        cv2.imwrite(enhanced_img_path, enhanced_image)
                        inference_path = enhanced_img_path
                        was_enhanced = True
            
            # --- RUN INFERENCE ---
            pred_boxes = self.run_sahi_inference(inference_path, category)
            
            # --- SCALE DOWN BOUNDING BOX (jika di-enhance) ---
            if was_enhanced and self.face_enhancer:
                scale_factor = self.face_enhancer.scale
                scaled_pred_boxes = []
                for pred in pred_boxes:
                    bbox = pred['bbox']
                    scaled_bbox = [
                        round(bbox[0] / scale_factor),  # Gunakan round() untuk akurasi lebih baik
                        round(bbox[1] / scale_factor),
                        round(bbox[2] / scale_factor),
                        round(bbox[3] / scale_factor),
                    ]
                    scaled_pred_boxes.append({
                        'bbox': scaled_bbox,
                        'confidence': pred['confidence']
                    })
                pred_boxes = scaled_pred_boxes
            
            # Clean up temp file
            if was_enhanced and inference_path != original_img_path:
                os.remove(inference_path)
            
            total_faces_detected += len(pred_boxes)
            
            original_path_key = img_file.name.replace('_', '/', 1)
            predictions[original_path_key] = pred_boxes
        
        # Print category statistics
        if self.bounded_enhancement and self.use_enhancer:
            enhance_ratio = (category_stats['enhanced'] / category_stats['total'] * 100) if category_stats['total'] > 0 else 0
            print(f"   ✓ Enhancement Stats: {category_stats['enhanced']}/{category_stats['total']} images enhanced ({enhance_ratio:.1f}%)")
            print(f"   ✓ Total faces detected: {total_faces_detected}")
        else:
            print(f"   ✓ Total faces detected: {total_faces_detected}")
        
        return predictions
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x, y, w, h]"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        box1_x2, box1_y2 = x1 + w1, y1 + h1
        box2_x2, box2_y2 = x2 + w2, y2 + h2
        inter_x1, inter_y1 = max(x1, x2), max(y1, y2)
        inter_x2, inter_y2 = min(box1_x2, box2_x2), min(box1_y2, box2_y2)
        if inter_x2 < inter_x1 or inter_y2 < inter_y1: 
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = (w1 * h1) + (w2 * h2) - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def calculate_average_precision(self, all_detections, total_gt):
        """Calculate Average Precision (AP) using 11-point interpolation (VOC style)."""
        if total_gt == 0 or not all_detections: 
            return 0.0
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        tp_cumsum = np.cumsum([d['is_tp'] for d in all_detections])
        fp_cumsum = np.cumsum([not d['is_tp'] for d in all_detections])
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        ap = 0.0
        for t in np.arange(0., 1.1, 0.1):
            p = 0 if np.sum(recalls >= t) == 0 else np.max(precisions[recalls >= t])
            ap += p / 11.0
        return ap

    def evaluate_category(self, gt_annotations, predictions, category):
        """Evaluate detection for a single category and calculate AP."""
        print(f"\n🔍 Evaluating {category} for AP calculation...")
        category_path = self.classified_path / category
        image_files = [f.name for f in category_path.glob("*.jpg")]
        total_gt = 0
        all_detections = []
        
        for img_file in tqdm(image_files, desc=f"Matching {category}"):
            original_path = img_file.replace('_', '/', 1)
            if original_path not in gt_annotations: 
                continue
            
            gt_boxes = gt_annotations[original_path]
            pred_boxes = predictions.get(original_path, [])
            total_gt += len(gt_boxes)
            gt_matched_in_image = [False] * len(gt_boxes)
            pred_boxes.sort(key=lambda x: x['confidence'], reverse=True)
            
            for pred in pred_boxes:
                best_iou, best_gt_idx = 0, -1
                for j, gt in enumerate(gt_boxes):
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, j
                
                is_tp = False
                if best_iou >= self.iou_threshold and best_gt_idx != -1 and not gt_matched_in_image[best_gt_idx]:
                    is_tp = True
                    gt_matched_in_image[best_gt_idx] = True
                
                all_detections.append({'confidence': pred['confidence'], 'is_tp': is_tp})
        
        ap = self.calculate_average_precision(all_detections, total_gt)
        true_positives_at_thresh = sum(1 for d in all_detections if d['confidence'] >= self.global_confidence and d['is_tp'])
        total_pred_at_thresh = sum(1 for d in all_detections if d['confidence'] >= self.global_confidence)
        false_positives_at_thresh = total_pred_at_thresh - true_positives_at_thresh
        false_negatives_at_thresh = total_gt - true_positives_at_thresh
        precision = true_positives_at_thresh / total_pred_at_thresh if total_pred_at_thresh > 0 else 0
        recall = true_positives_at_thresh / total_gt if total_gt > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'category': category, 
            'total_gt': total_gt, 
            'total_pred': total_pred_at_thresh,
            'true_positives': true_positives_at_thresh, 
            'false_positives': false_positives_at_thresh,
            'false_negatives': false_negatives_at_thresh, 
            'precision': precision, 
            'recall': recall,
            'f1_score': f1_score, 
            'ap': ap, 
            'all_detections': all_detections
        }
    
    def visualize_results(self, summary_results, output_dir="output"):
        """Visualisasi hasil evaluasi"""
        print("\n📊 Creating visualizations...")
        all_results = summary_results['category_results']
        overall_map = summary_results['overall_map']
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Clean filename - remove invalid characters for Windows
        mode_str = summary_results['mode'].replace(" -> ", "_").replace(" ", "_")
        mode_str = mode_str.replace("(", "").replace(")", "").replace(",", "").replace("<", "lt")
        filename_prefix = f"{mode_str.lower()}_sahi_{self.slicing_strategy}" if self.use_sahi else f"{mode_str.lower()}_standard"
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        categories, tp_counts, fp_counts, fn_counts = [], [], [], []
        precisions, recalls, f1_scores, aps = [], [], [], []
        
        for result in all_results:
            categories.append(result['category'].replace('_', '\n').title())
            tp_counts.append(result['true_positives'])
            fp_counts.append(result['false_positives'])
            fn_counts.append(result['false_negatives'])
            precisions.append(result['precision'] * 100)
            recalls.append(result['recall'] * 100)
            f1_scores.append(result['f1_score'] * 100)
            aps.append(result['ap'] * 100)
        
        # Plot 1: Detection Success vs Failure
        ax1 = axes[0, 0]
        x = np.arange(len(categories))
        width = 0.35
        bars1 = ax1.bar(x - width/2, tp_counts, width, label=f'Detected (TP at {self.global_confidence} conf)', 
                       color='#27ae60', edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, fn_counts, width, label=f'Missed (FN at {self.global_confidence} conf)', 
                       color='#c0392b', edgecolor='black', linewidth=1.5)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', 
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax1.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Detection Success vs Failure', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, fontsize=10)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Performance Metrics
        ax2 = axes[0, 1]
        x = np.arange(len(categories))
        width = 0.2
        ax2.bar(x - width*1.5, precisions, width, label='Precision', color='#3498db')
        ax2.bar(x - width*0.5, recalls, width, label='Recall', color='#e67e22')
        ax2.bar(x + width*0.5, f1_scores, width, label='F1-Score', color='#9b59b6')
        ax2.bar(x + width*1.5, aps, width, label='AP', color='#1abc9c', edgecolor='black', linewidth=1.5)
        
        for i, ap in enumerate(aps):
            ax2.text(x[i] + width*1.5, ap + 1, f'{ap:.1f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
        
        ax2.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Performance Metrics', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, fontsize=10)
        ax2.legend(fontsize=10)
        ax2.set_ylim(0, 105)
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: AP per Category & mAP
        ax3 = axes[1, 0]
        x = np.arange(len(categories) + 1)
        width = 0.6
        ap_values = aps + [overall_map * 100]
        labels = categories + ['Overall\nmAP']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax3.bar(x, ap_values, width, color=colors, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, ap_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1, f'{val:.2f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax3.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Average Precision (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Average Precision (AP) per Category & mAP', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=10)
        ax3.set_ylim(0, max(ap_values) * 1.15)
        ax3.grid(axis='y', alpha=0.3)
        bars[-1].set_linewidth(2.5)
        
        # Plot 4: Summary Table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = [['Category', 'GT', 'Det', 'TP', 'FP', 'FN', 'Prec', 'Rec', 'F1', 'AP']]
        
        for result in all_results:
            table_data.append([
                result['category'].replace('_', ' ').title(),
                result['total_gt'],
                result['total_pred'],
                result['true_positives'],
                result['false_positives'],
                result['false_negatives'],
                f"{result['precision']*100:.1f}%",
                f"{result['recall']*100:.1f}%",
                f"{result['f1_score']*100:.1f}%",
                f"{result['ap']*100:.2f}%"
            ])
        
        total_gt_all = sum(r['total_gt'] for r in all_results)
        table_data.append(['---']*10)
        table_data.append(['**Overall**', total_gt_all, '-', '-', '-', '-', '-', '-', '**mAP**', f"**{overall_map*100:.2f}%**"])
        
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center', 
                         colWidths=[0.15, 0.07, 0.07, 0.07, 0.07, 0.07, 0.09, 0.09, 0.09, 0.09])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 2.2)
        
        # Header styling
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Row styling
        for i in range(1, len(table_data)):
            if i % 2 == 0 and i < len(all_results) + 1:
                for j in range(len(table_data[0])):
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        # mAP row styling
        map_row_idx = len(table_data) - 1
        for j in range(len(table_data[0])):
            table[(map_row_idx, j)].set_facecolor('#3498db')
            table[(map_row_idx, j)].set_text_props(weight='bold', color='white')
        
        # Configuration text
        config_text = f"Config: {summary_results['mode'].upper()} | IoU: {self.iou_threshold} | Conf Thresh: {self.global_confidence}"
        if self.use_sahi:
            if self.slicing_strategy == 'uniform':
                config_text += f" | Slice: {self.sahi_config['slice_width']}x{self.sahi_config['slice_height']}"
            config_text += f" | Overlap: {self.sahi_config['overlap_ratio']}"
        
        if self.bounded_enhancement:
            enhance_rate = (self.enhancement_stats['enhanced_images'] / self.enhancement_stats['total_images'] * 100) if self.enhancement_stats['total_images'] > 0 else 0
            config_text += f" | Enhanced: {self.enhancement_stats['enhanced_images']}/{self.enhancement_stats['total_images']} ({enhance_rate:.1f}%)"
        
        ax4.text(0.5, 0.05, config_text, transform=ax4.transAxes, ha='center', fontsize=9, 
                style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle(f'YOLO Evaluation - {summary_results["mode"].upper()} Mode (mAP: {overall_map*100:.2f}%)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        
        output_file = output_path / f"{filename_prefix}_eval_results.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def print_summary(self, summary_results):
        """Print summary statistics"""
        mode_str = summary_results['mode']
        all_results = summary_results['category_results']
        overall_map = summary_results['overall_map']
        
        print("\n" + "="*80)
        print(f"📈 EVALUATION SUMMARY - {mode_str} MODE")
        print(f"   Global Confidence Threshold: {self.global_confidence}")
        
        if self.use_sahi:
            if self.slicing_strategy == 'uniform':
                print(f"   Slice Size: {self.sahi_config['slice_width']}x{self.sahi_config['slice_height']}")
            else:
                print(f"   Slice Size: ADAPTIVE (by resolution)")
            print(f"   Overlap Ratio: {self.sahi_config['overlap_ratio']}")
        
        if self.bounded_enhancement:
            enhance_rate = (self.enhancement_stats['enhanced_images'] / self.enhancement_stats['total_images'] * 100) if self.enhancement_stats['total_images'] > 0 else 0
            print(f"\n   🎯 Bounded Enhancement Statistics:")
            print(f"   - Total Images: {self.enhancement_stats['total_images']}")
            print(f"   - Enhanced: {self.enhancement_stats['enhanced_images']} ({enhance_rate:.1f}%)")
            print(f"   - Skipped: {self.enhancement_stats['skipped_images']} ({100-enhance_rate:.1f}%)")
            print(f"   - Threshold: {self.face_size_threshold}px")
        
        print("="*80)
        
        for result in all_results:
            print(f"\n{result['category'].upper().replace('_', ' ')}")
            print("-"*80)
            print(f"  Ground Truth      : {result['total_gt']:,}")
            print(f"  Predicted         : {result['total_pred']:,}")
            print(f"  True Positives    : {result['true_positives']:,}")
            print(f"  False Positives   : {result['false_positives']:,}")
            print(f"  False Negatives   : {result['false_negatives']:,}")
            print(f"  Precision         : {result['precision']*100:.2f}%")
            print(f"  Recall            : {result['recall']*100:.2f}%")
            print(f"  F1-Score          : {result['f1_score']*100:.2f}%")
            print(f"  📊 AP (Avg Prec)  : {result['ap']*100:.2f}%")
        
        print("\n" + "="*80)
        print(f"🎯 OVERALL mAP (Mean Average Precision): {overall_map*100:.2f}%")
        print("="*80)
    
    def save_enhancement_analysis(self, output_dir="output"):
        """Simpan analisis detail tentang enhancement decisions"""
        if not self.bounded_enhancement or not self.enhancement_stats['enhancement_decisions']:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed JSON - clean filename
        filename_prefix = f"bounded_sahi_{self.slicing_strategy}" if self.use_sahi else "bounded_standard"
        filename_prefix = filename_prefix.replace("<", "lt").replace(">", "gt").replace(",", "")
        json_file = output_path / f"{filename_prefix}_enhancement_analysis.json"
        
        with open(json_file, 'w') as f:
            json.dump(self.enhancement_stats, f, indent=2)
        
        print(f"\n✓ Enhancement analysis saved to: {json_file}")
        
        # Print summary by category
        print("\n📊 Enhancement Breakdown by Category:")
        print("-"*80)
        
        by_category = defaultdict(lambda: {'enhanced': 0, 'skipped': 0, 'total': 0})
        
        for decision in self.enhancement_stats['enhancement_decisions']:
            cat = decision['category']
            by_category[cat]['total'] += 1
            if decision['should_enhance']:
                by_category[cat]['enhanced'] += 1
            else:
                by_category[cat]['skipped'] += 1
        
        for cat in sorted(by_category.keys()):
            stats = by_category[cat]
            enhance_rate = (stats['enhanced'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {cat.replace('_', ' ').title():20s}: "
                  f"{stats['enhanced']:4d}/{stats['total']:4d} enhanced ({enhance_rate:5.1f}%)")
        
        print("-"*80)
    
    def run(self):
        """Run complete evaluation"""
        mode_str = "SAHI" if self.use_sahi else "Standard"
        if self.use_enhancer and self.face_enhancer:
            if self.bounded_enhancement:
                mode_str = f"BoundedEnhance({self.face_enhancer.scale}x, <{self.face_size_threshold}px) -> {mode_str}"
            else:
                mode_str = f"Enhance({self.face_enhancer.scale}x) -> {mode_str}"
        
        print("="*80)
        print(f"🎯 YOLO EVALUATION - {mode_str} MODE")
        print("="*80)
        
        # Setup temp directory
        if self.use_enhancer:
            if self.temp_enh_dir.exists():
                shutil.rmtree(self.temp_enh_dir)
            self.temp_enh_dir.mkdir(parents=True)
            print(f"✓ Created temp directory for enhancement: {self.temp_enh_dir}")
        
        # Parse ground truth
        gt_annotations = self.parse_ground_truth()
        
        # Run inference and evaluation per category
        all_results = []
        start_time = time.time()
        
        for category in self.categories:
            predictions = self.run_inference_on_images(category)
            result = self.evaluate_category(gt_annotations, predictions, category)
            all_results.append(result)
        
        elapsed_time = time.time() - start_time
        
        # Calculate overall mAP
        overall_map = np.mean([result['ap'] for result in all_results])
        
        # Prepare summary
        summary_results = {
            'category_results': all_results,
            'overall_map': overall_map,
            'mode': mode_str,
            'elapsed_time': elapsed_time,
            'config': {
                'use_sahi': self.use_sahi,
                'use_enhancer': self.use_enhancer,
                'bounded_enhancement': self.bounded_enhancement,
                'face_size_threshold': self.face_size_threshold,
                'slicing_strategy': self.slicing_strategy,
                'global_confidence': self.global_confidence,
                'inference_confidence': self.inference_confidence,
                'iou_threshold': self.iou_threshold
            }
        }
        
        # Print and visualize results
        self.print_summary(summary_results)
        self.visualize_results(summary_results)
        
        # Save enhancement analysis (if bounded)
        if self.bounded_enhancement:
            self.save_enhancement_analysis()
        
        # Save JSON results
        # Clean filename for cross-platform compatibility
        filename_prefix = f"sahi_{self.slicing_strategy}" if self.use_sahi else "standard"
        if self.use_enhancer:
            if self.bounded_enhancement:
                filename_prefix = f"bounded_enhanced_{filename_prefix}"
            else:
                filename_prefix = f"enhanced_{filename_prefix}"
        
        # Remove any invalid characters
        filename_prefix = filename_prefix.replace("<", "lt").replace(">", "gt").replace(",", "")
        
        output_file = Path("output") / f"{filename_prefix}_results.json"
        output_file.parent.mkdir(exist_ok=True)
        
        json_results = {
            'mode': mode_str,
            'overall_map': float(overall_map),
            'elapsed_time': elapsed_time,
            'config': summary_results['config'],
            'category_results': []
        }
        
        for result in all_results:
            json_result = result.copy()
            json_result.pop('all_detections', None)
            json_results['category_results'].append(json_result)
        
        # Add enhancement stats if bounded
        if self.bounded_enhancement:
            json_results['enhancement_stats'] = {
                'total_images': self.enhancement_stats['total_images'],
                'enhanced_images': self.enhancement_stats['enhanced_images'],
                'skipped_images': self.enhancement_stats['skipped_images'],
                'enhancement_rate': (self.enhancement_stats['enhanced_images'] / self.enhancement_stats['total_images'] * 100) if self.enhancement_stats['total_images'] > 0 else 0
            }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        # Cleanup temp directory
        if self.use_enhancer and self.temp_enh_dir.exists():
            shutil.rmtree(self.temp_enh_dir)
            print(f"✓ Cleaned up temp directory: {self.temp_enh_dir}")
        
        print(f"\n🎉 Evaluation completed in {elapsed_time:.1f}s!")
        print(f"   Final mAP: {overall_map*100:.2f}%")
        
        return summary_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YOLO on WIDER Face with Bounded Enhancement and SAHI')
    parser.add_argument('--mode', 
                       choices=['standard', 'sahi-uniform', 'sahi-adaptive'], 
                       default='sahi-uniform',
                       help='Inference mode: standard (no SAHI), sahi-uniform, or sahi-adaptive')
    parser.add_argument('--enhance', 
                       action='store_true', 
                       help='Enable Enhancement pipeline')
    parser.add_argument('--bounded', 
                       action='store_true', 
                       help='Enable Bounded Enhancement (selective enhancement based on face size)')
    parser.add_argument('--threshold', 
                       type=int, 
                       default=50,
                       help='Face size threshold for bounded enhancement (default: 50px)')
    parser.add_argument('--device', 
                       default='cuda:0', 
                       help='Device: cuda:0 or cpu')
    
    args = parser.parse_args()
    
    # Parse mode
    use_sahi = args.mode.startswith('sahi')
    slicing_strategy = args.mode.split('-')[1] if use_sahi else 'uniform'
    
    # Enhancement logic
    use_enhancer = args.enhance
    bounded_enhancement = args.bounded if args.enhance else False
    
    # Build mode string for display
    mode_parts = []
    if use_enhancer:
        if bounded_enhancement:
            mode_parts.append(f"BOUNDED-ENHANCE (<{args.threshold}px)")
        else:
            mode_parts.append("FULL-ENHANCE")
    mode_parts.append(args.mode.upper())
    mode_str = " -> ".join(mode_parts)
    
    print(f"\n{'='*80}")
    print(f"🚀 YOLO Face Detection Evaluation")
    print(f"{'='*80}")
    print(f"Mode: {mode_str}")
    print(f"Device: {args.device}")
    if use_enhancer:
        print(f"Enhancement: {'Bounded (Selective)' if bounded_enhancement else 'Full (All Images)'}")
        if bounded_enhancement:
            print(f"Face Size Threshold: {args.threshold}px")
    print(f"{'='*80}\n")
    
    # Create evaluator
    evaluator = SAHIWiderFaceEvaluator(
        base_path="data/dataset/widerface",
        model_path="models/yolo11s-pose-default/yolo11s_pose/weights/best.pt",
        device=args.device,
        use_sahi=use_sahi,
        use_enhancer=use_enhancer,
        bounded_enhancement=bounded_enhancement,
        face_size_threshold=args.threshold,
        slicing_strategy=slicing_strategy
    )
    
    # Run evaluation
    results = evaluator.run()
    
    # Print final comparison
    print(f"\n{'='*80}")
    print(f"📊 FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Mode: {mode_str}")
    print(f"Overall mAP: {results['overall_map']*100:.2f}%")
    print(f"Inference Time: {results['elapsed_time']:.1f}s")
    
    if bounded_enhancement:
        enhance_rate = (evaluator.enhancement_stats['enhanced_images'] / evaluator.enhancement_stats['total_images'] * 100)
        print(f"\nBounded Enhancement Stats:")
        print(f"  Enhanced: {evaluator.enhancement_stats['enhanced_images']}/{evaluator.enhancement_stats['total_images']} images ({enhance_rate:.1f}%)")
        print(f"  Time Saved: ~{evaluator.enhancement_stats['skipped_images'] * 0.5:.1f}s (estimated)")
    
    print(f"{'='*80}\n")

     