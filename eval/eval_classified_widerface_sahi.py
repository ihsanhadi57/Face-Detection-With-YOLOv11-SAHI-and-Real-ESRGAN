import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import cv2

# Import SAHI dan wrapper YOLO kita
from sahi.predict import get_sliced_prediction
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.yolo_wrapper import YOLOv11PoseDetectionModel

class SAHIWiderFaceEvaluator:
    def __init__(self, 
                 base_path="data/dataset/widerface", 
                 model_path="models/yolo11s-pose-max/yolo11s_pose/weights/best.pt",
                 device='cuda:0',
                 use_sahi=True,
                 slicing_strategy='uniform'):
        self.base_path = Path(base_path)
        self.classified_path = self.base_path / "classified_val"
        self.label_file = self.base_path / "wider_face_split" / "wider_face_val_bbx_gt.txt"
        
        self.categories = ['small_clear', 'small_degraded', 'medium_large']
        
        # IOU threshold untuk match detection
        self.iou_threshold = 0.5
        
        # GLOBAL CONFIDENCE THRESHOLD untuk fair comparison
        # PENTING: Untuk SAHI, gunakan confidence lebih rendah saat inference
        # karena NMS akan filter duplikat
        self.global_confidence = 0.25  # Untuk evaluasi metrics
        self.inference_confidence = 0.01 if use_sahi else 0.5  # Untuk detection
        
        # Flag untuk enable/disable SAHI
        self.use_sahi = use_sahi
        
        # Slicing strategy: 'uniform' atau 'adaptive'
        self.slicing_strategy = slicing_strategy
        
        # ===== SAHI CONFIGURATION =====
        if slicing_strategy == 'uniform':
            # Strategy 1: UNIFORM - OPTIMAL untuk WIDER Face
            self.sahi_config = {
                'slice_height': 640,
                'slice_width': 640,
                'overlap_ratio': 0.2,  # TURUNKAN dari 0.3 -> mengurangi duplikat
                'confidence_threshold': 0.01,  # RENDAH untuk capture semua deteksi
                'postprocess_match_threshold': 0.5,  # IOU untuk NMS antar slice
                'postprocess_class_threshold': 0.25,  # Filter akhir per confidence
                'postprocess_type': 'NMS',
                'description': 'Uniform 640x640 slicing - OPTIMAL config'
            }
            print(f"\n📐 Slicing Strategy: UNIFORM (640x640, overlap=0.2)")
            
        elif slicing_strategy == 'adaptive':
            # Strategy 2: ADAPTIVE - dengan config optimal
            self.sahi_config = {
                'confidence_threshold': 0.01,  # RENDAH untuk capture semua
                'postprocess_match_threshold': 0.5,  # NMS threshold
                'postprocess_class_threshold': 0.25,  # Filter akhir
                'postprocess_type': 'NMS',
                'overlap_ratio': 0.2,  # TURUNKAN untuk kurangi duplikat
                'description': 'Adaptive slicing - OPTIMAL config'
            }
            print(f"\n📐 Slicing Strategy: ADAPTIVE (overlap=0.2)")
        
        # Load YOLO model with wrapper
        print(f"🔧 Loading YOLO model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.detection_model = YOLOv11PoseDetectionModel(
            model_path=model_path,
            confidence_threshold=self.inference_confidence,  # Gunakan inference_confidence
            device=device,
            load_at_init=True
        )
        print(f"✓ Model loaded! Mode: {'SAHI' if use_sahi else 'Standard'}")
        print(f"  - Inference Conf: {self.inference_confidence}")
        print(f"  - Eval Conf: {self.global_confidence}")
        
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
                                
                                # Skip invalid boxes
                                if w > 0 and h > 0:
                                    faces.append({
                                        'bbox': [x, y, w, h],
                                        'x': x, 'y': y, 'w': w, 'h': h
                                    })
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
        """
        Adaptive slice - OPTIMIZED untuk face detection
        """
        max_dim = max(img_width, img_height)
        
        # REVISED: Slice lebih kecil untuk better small face detection
        if max_dim > 2500:       # Ultra High Resolution (4K+)
                # Turunkan ke 512 untuk meningkatkan jumlah slice dan konteks objek kecil
                slice_size = 512     
        elif max_dim > 1500:     # High Resolution (2K/HD)
                slice_size = 416
        elif max_dim > 800:      # Medium Large
                slice_size = 416
        elif max_dim > 400:      # Medium Small
                slice_size = 320
        else:                    # Smallest images
                # Langsung inferensi jika gambar sangat kecil
                slice_size = 320
        
        return (slice_size // 32) * 32
    
    def run_sahi_inference(self, img_path, category):
        """
        Run SAHI inference dengan konfigurasi yang konsisten
        """
        # Load image untuk get size
        img = cv2.imread(img_path)
        if img is None:
            return []
        
        img_height, img_width = img.shape[:2]
        
        # ===== MODE 1: STANDARD (No SAHI) =====
        if not self.use_sahi:
            results = self.detection_model.model(
                img, 
                conf=self.inference_confidence,  # Gunakan inference_confidence
                verbose=False
            )
            
            pred_boxes = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    x1, y1, x2, y2 = xyxy
                    pred_boxes.append({
                        'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                        'confidence': conf
                    })
            return pred_boxes
        
        # ===== MODE 2: SAHI =====
        
        # Determine slice size
        if self.slicing_strategy == 'uniform':
            slice_height = self.sahi_config['slice_height']
            slice_width = self.sahi_config['slice_width']
        else:  # adaptive
            slice_size = self.get_slice_size_adaptive(img_width, img_height)
            slice_height = slice_size
            slice_width = slice_size
        
        # Skip slicing jika gambar terlalu kecil
        if img_width < slice_width * 0.8 and img_height < slice_height * 0.8:
            # Direct inference untuk gambar kecil
            results = self.detection_model.model(
                img, 
                conf=self.inference_confidence,  # Gunakan inference_confidence
                verbose=False
            )
            
            pred_boxes = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    x1, y1, x2, y2 = xyxy
                    pred_boxes.append({
                        'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                        'confidence': conf
                    })
            return pred_boxes
        
        # Update model confidence threshold untuk SAHI
        self.detection_model.confidence_threshold = self.sahi_config['confidence_threshold']
        
        # Run SAHI prediction dengan class agnostic postprocessing
        result = get_sliced_prediction(
            img_path,
            self.detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=self.sahi_config['overlap_ratio'],
            overlap_width_ratio=self.sahi_config['overlap_ratio'],
            postprocess_type=self.sahi_config['postprocess_type'],
            postprocess_match_threshold=self.sahi_config['postprocess_match_threshold'],
            postprocess_class_agnostic=True,  # PENTING: Hindari aggressive NMS per class
            verbose=0
        )
        
        # Convert to our format (NO KEYPOINTS!)
        pred_boxes = []
        for detection in result.object_prediction_list:
            bbox = detection.bbox.to_xyxy()
            x1, y1, x2, y2 = bbox
            
            pred_boxes.append({
                'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                'confidence': detection.score.value
            })
        
        return pred_boxes
    
    def run_inference_on_images(self, category):
        """Run inference pada semua gambar dalam kategori"""
        mode_str = "SAHI" if self.use_sahi else "Standard"
        
        print(f"\n🚀 Running {mode_str} inference on {category}...")
        
        if self.use_sahi:
            if self.slicing_strategy == 'uniform':
                print(f"   Strategy: UNIFORM")
                print(f"   - Slice: {self.sahi_config['slice_width']}x{self.sahi_config['slice_height']}")
                print(f"   - Overlap: {self.sahi_config['overlap_ratio']}")
            else:
                print(f"   Strategy: ADAPTIVE (by resolution)")
                print(f"   - Overlap: {self.sahi_config['overlap_ratio']}")
            print(f"   - Inference Confidence: {self.sahi_config['confidence_threshold']}")
            print(f"   - Eval Confidence: {self.global_confidence}")
            print(f"   - NMS Threshold: {self.sahi_config['postprocess_match_threshold']}")
        else:
            print(f"   - Direct inference (no slicing)")
            print(f"   - Confidence: {self.inference_confidence}")
        
        category_path = self.classified_path / category
        image_files = list(category_path.glob("*.jpg"))
        
        predictions = {}
        total_faces_detected = 0
        
        # Track slice sizes untuk adaptive mode
        slice_size_counts = defaultdict(int)
        
        for img_file in tqdm(image_files, desc=f"{mode_str} {category}"):
            img_path = str(img_file)
            
            # Track slice size jika adaptive
            if self.use_sahi and self.slicing_strategy == 'adaptive':
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    slice_size = self.get_slice_size_adaptive(w, h)
                    slice_size_counts[slice_size] += 1
            
            # Run inference
            pred_boxes = self.run_sahi_inference(img_path, category)
            
            # Count stats
            total_faces_detected += len(pred_boxes)
            
            # Restore original path format
            original_path = img_file.name.replace('_', '/', 1)
            predictions[original_path] = pred_boxes
        
        print(f"   ✓ Total faces detected: {total_faces_detected}")
        
        if self.use_sahi and self.slicing_strategy == 'adaptive':
            print(f"   📊 Slice size distribution:")
            for size in sorted(slice_size_counts.keys(), reverse=True):
                count = slice_size_counts[size]
                pct = count / len(image_files) * 100
                print(f"      {size}x{size}: {count} images ({pct:.1f}%)")
        
        return predictions
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x, y, w, h]"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to [x1, y1, x2, y2]
        box1_x2 = x1 + w1
        box1_y2 = y1 + h1
        box2_x2 = x2 + w2
        box2_y2 = y2 + h2
        
        # Intersection area
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def calculate_average_precision(self, all_detections, total_gt):
        """Calculate Average Precision (AP) using 11-point interpolation (VOC style)."""
        if total_gt == 0 or not all_detections:
            return 0.0

        # Sort detections by confidence
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)

        tp_cumsum = np.cumsum([d['is_tp'] for d in all_detections])
        fp_cumsum = np.cumsum([not d['is_tp'] for d in all_detections])
        
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # 11-point interpolated AP
        ap = 0.0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        
        return ap

    def evaluate_category(self, gt_annotations, predictions, category):
        """Evaluate detection for a single category and calculate AP."""
        print(f"\n🔍 Evaluating {category} for AP calculation...")
        
        category_path = self.classified_path / category
        image_files = [f.name for f in category_path.glob("*.jpg")]
        
        total_gt = 0
        all_detections = []  # List to store {confidence, is_tp}

        for img_file in tqdm(image_files, desc=f"Matching {category}"):
            original_path = img_file.replace('_', '/', 1)
            
            if original_path not in gt_annotations:
                continue
            
            gt_boxes = gt_annotations[original_path]
            pred_boxes = predictions.get(original_path, [])
            
            total_gt += len(gt_boxes)
            
            gt_matched_in_image = [False] * len(gt_boxes)
            
            # Sort predictions by confidence for this image to prioritize high-confidence matches
            pred_boxes.sort(key=lambda x: x['confidence'], reverse=True)

            for pred in pred_boxes:
                pred_bbox = pred['bbox']
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt in enumerate(gt_boxes):
                    iou = self.calculate_iou(pred_bbox, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                is_tp = False
                if best_iou >= self.iou_threshold:
                    if not gt_matched_in_image[best_gt_idx]:
                        is_tp = True
                        gt_matched_in_image[best_gt_idx] = True

                all_detections.append({
                    'confidence': pred['confidence'],
                    'is_tp': is_tp
                })

        # Calculate Average Precision for the category
        ap = self.calculate_average_precision(all_detections, total_gt)

        # Calculate legacy metrics at the global confidence threshold for comparison
        true_positives_at_thresh = 0
        total_pred_at_thresh = 0
        for det in all_detections:
            if det['confidence'] >= self.global_confidence:
                total_pred_at_thresh += 1
                if det['is_tp']:
                    true_positives_at_thresh += 1
        
        false_positives_at_thresh = total_pred_at_thresh - true_positives_at_thresh
        
        # FN is total GT minus TPs found *above the threshold*
        matched_gt_above_thresh = true_positives_at_thresh
        false_negatives_at_thresh = total_gt - matched_gt_above_thresh

        precision = true_positives_at_thresh / total_pred_at_thresh if total_pred_at_thresh > 0 else 0
        recall = true_positives_at_thresh / total_gt if total_gt > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results = {
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
        
        return results
    
    def visualize_results(self, summary_results, output_dir="output"):
        """Visualisasi hasil evaluasi"""
        print("\n📊 Creating visualizations...")
        
        all_results = summary_results['category_results']
        overall_map = summary_results['overall_map']

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        mode_str = f"SAHI-{self.slicing_strategy}" if self.use_sahi else "Standard"
        
        # Set style
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        categories = []
        tp_counts = []
        fp_counts = []
        fn_counts = []
        precisions = []
        recalls = []
        f1_scores = []
        aps = []
        
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
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=10)
        
        ax1.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Detection Success vs Failure', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, fontsize=10)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Metrics Bar Chart (DENGAN AP)
        ax2 = axes[0, 1]
        x = np.arange(len(categories))
        width = 0.2
        
        ax2.bar(x - width*1.5, precisions, width, label='Precision', color='#3498db')
        ax2.bar(x - width*0.5, recalls, width, label='Recall', color='#e67e22')
        ax2.bar(x + width*0.5, f1_scores, width, label='F1-Score', color='#9b59b6')
        ax2.bar(x + width*1.5, aps, width, label='AP', color='#1abc9c', edgecolor='black', linewidth=1.5)

        # Tambahkan nilai di atas bar untuk AP
        for i, ap in enumerate(aps):
            ax2.text(x[i] + width*1.5, ap + 1, f'{ap:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

        ax2.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Performance Metrics', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, fontsize=10)
        ax2.legend(fontsize=10)
        ax2.set_ylim(0, 105)
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: AP per Kategori + mAP
        ax3 = axes[1, 0]
        x = np.arange(len(categories) + 1)
        width = 0.6
        
        ap_values = aps + [overall_map * 100]
        labels = categories + ['Overall\nmAP']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        bars = ax3.bar(x, ap_values, width, color=colors, edgecolor='black', linewidth=1.5)
        
        # Tambahkan nilai di atas bar
        for bar, val in zip(bars, ap_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.2f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
        
        ax3.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Average Precision (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Average Precision (AP) per Category & mAP', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=10)
        ax3.set_ylim(0, max(ap_values) * 1.15)
        ax3.grid(axis='y', alpha=0.3)
        
        # Highlight mAP bar
        bars[-1].set_linewidth(2.5)
        
        # Plot 4: Summary Table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        table_data.append(['Category', 'GT', 'Det', 'TP', 'FP', 'FN', 'Prec', 'Rec', 'F1', 'AP'])
        
        for result in all_results:
            row = [
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
            ]
            table_data.append(row)
        
        # Add overall mAP row
        total_gt_all = sum(r['total_gt'] for r in all_results)
        table_data.append(['---', '---', '---', '---', '---', '---', '---', '---', '---', '---'])
        table_data.append(['**Overall**', total_gt_all, '-', '-', '-', '-', '-', '-', '**mAP**', f"**{overall_map*100:.2f}%**"])

        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.15, 0.07, 0.07, 0.07, 0.07, 0.07, 0.09, 0.09, 0.09, 0.09])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 2.2)
        
        # Style header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style data rows
        for i in range(1, len(table_data)):
            if i % 2 == 0 and i < len(all_results) + 1:
                for j in range(len(table_data[0])):
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        # Style mAP row
        map_row_idx = len(table_data) - 1
        for j in range(len(table_data[0])):
            table[(map_row_idx, j)].set_facecolor('#3498db')
            table[(map_row_idx, j)].set_text_props(weight='bold', color='white')

        # Add config info as text
        config_text = f"Config: {mode_str.upper()} | IoU: {self.iou_threshold} | Conf Thresh: {self.global_confidence}"
        if self.use_sahi:
            if self.slicing_strategy == 'uniform':
                config_text += f" | Slice: {self.sahi_config['slice_width']}x{self.sahi_config['slice_height']}"
            config_text += f" | Overlap: {self.sahi_config['overlap_ratio']}"
        
        ax4.text(0.5, 0.05, config_text, transform=ax4.transAxes,
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle(f'YOLO Evaluation - {mode_str.upper()} Mode (mAP: {overall_map*100:.2f}%)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        
        filename = f"{mode_str.lower().replace('-', '_')}_eval_results.png"
        output_file = output_path / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def print_summary(self, summary_results):
            """Print summary statistics"""
            mode_str = f"SAHI ({self.slicing_strategy.upper()})" if self.use_sahi else "STANDARD"
            
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
            print("="*80)
            
            for result in all_results:
                print(f"\n{result['category'].upper().replace('_', ' ')}")
                print("-" * 80)
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
    
    def run(self):
        """Run complete evaluation"""
        mode_str = f"SAHI ({self.slicing_strategy.upper()})" if self.use_sahi else "STANDARD"
        
        print("="*80)
        print(f"🎯 YOLO EVALUATION - {mode_str} MODE")
        print("="*80)
        
        # 1. Load ground truth
        gt_annotations = self.parse_ground_truth()
        
        # 2. Run inference dan evaluate
        all_results = []
        
        for category in self.categories:
            predictions = self.run_inference_on_images(category)
            result = self.evaluate_category(gt_annotations, predictions, category)
            all_results.append(result)
        
        # 3. Calculate overall mAP (mean of category APs)
        overall_map = np.mean([result['ap'] for result in all_results])
        
        # 4. Create summary results with mAP
        summary_results = {
            'category_results': all_results,
            'overall_map': overall_map,
            'mode': mode_str,
            'config': {
                'use_sahi': self.use_sahi,
                'slicing_strategy': self.slicing_strategy,
                'global_confidence': self.global_confidence,
                'inference_confidence': self.inference_confidence,
                'iou_threshold': self.iou_threshold
            }
        }
        
        # 5. Print summary
        self.print_summary(summary_results)
        
        # 6. Visualize
        self.visualize_results(summary_results)
        
        # 7. Save JSON
        filename_prefix = f"sahi_{self.slicing_strategy}" if self.use_sahi else "standard"
        output_file = Path("output") / f"{filename_prefix}_results.json"
        output_file.parent.mkdir(exist_ok=True)
        
        # Prepare data for JSON (remove non-serializable parts)
        json_results = {
            'mode': mode_str,
            'overall_map': float(overall_map),
            'config': summary_results['config'],
            'category_results': []
        }
        
        for result in all_results:
            json_result = result.copy()
            # Remove all_detections dari JSON karena terlalu besar
            json_result.pop('all_detections', None)
            json_results['category_results'].append(json_result)
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
        
        print(f"\n🎉 Evaluation completed!")
        print(f"   Final mAP: {overall_map*100:.2f}%")
        
        return summary_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YOLO on WIDER Face with OPTIMAL SAHI Config')
    parser.add_argument('--mode', choices=['standard', 'sahi-uniform', 'sahi-adaptive'], 
                       default='sahi-adaptive',
                       help='Inference mode: standard (no SAHI), sahi-uniform (640x640 RECOMMENDED), or sahi-adaptive (by resolution)')
    parser.add_argument('--device', default='cuda:0', help='Device: cuda:0 or cpu')
    
    args = parser.parse_args()
    
    # Parse mode
    use_sahi = args.mode.startswith('sahi')
    slicing_strategy = args.mode.split('-')[1] if use_sahi else 'uniform'
    
    print(f"\n🚀 Running evaluation with mode: {args.mode}")
    print("\n" + "="*80)
    print("📋 OPTIMAL SAHI CONFIGURATION FOR WIDER FACE:")
    print("="*80)
    if use_sahi:
        print("""
✅ KEY IMPROVEMENTS:
1. Inference Confidence: 0.01 (capture semua deteksi, filter nanti)
2. Eval Confidence: 0.25 (untuk perhitungan metrics)
3. Overlap Ratio: 0.2 (turun dari 0.3-0.35 -> kurangi duplikat)
4. NMS Threshold: 0.5 (balance antara merge dan preserve)
5. Class Agnostic: True (hindari aggressive NMS)

🎯 EXPECTED RESULTS:
- SAHI seharusnya LEBIH BAIK atau SETARA dengan Standard
- Improvement terbesar di small_clear & small_degraded
- Medium_large mungkin sedikit turun (acceptable trade-off)

⚠️ TROUBLESHOOTING:
Jika SAHI masih lebih rendah:
- Cek apakah ada terlalu banyak False Positives (turunkan inf_conf ke 0.05)
- Cek apakah banyak duplikat (turunkan overlap ke 0.1)
- Coba tanpa NMS: postprocess_type='GREEDYNMM'
        """)
    print("="*80 + "\n")
    
    # Run evaluation
    evaluator = SAHIWiderFaceEvaluator(
        base_path="data/dataset/widerface",
        model_path="models/yolo11s-pose-max/yolo11s_pose/weights/best.pt",
        device=args.device,
        use_sahi=use_sahi,
        slicing_strategy=slicing_strategy
    )
    
    results = evaluator.run()