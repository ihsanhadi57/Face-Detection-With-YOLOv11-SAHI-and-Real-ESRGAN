import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import cv2
from ultralytics import YOLO

class StandardWiderFaceEvaluator:
    """
    Menjalankan evaluasi YOLO standar (gambar penuh) pada WIDER Face.
    - Confidence Threshold: 0.5
    - IOU Evaluation Threshold: 0.5
    """
    def __init__(self, base_path="data/dataset/widerface", model_path="models/yolo11s-pose-max/yolo11s_pose/weights/best.pt"):
        self.base_path = Path(base_path)
        self.classified_path = self.base_path / "classified_val"
        self.label_file = self.base_path / "wider_face_split" / "wider_face_val_bbx_gt.txt"
        
        self.categories = ['small_clear', 'small_degraded', 'medium_large']
        
        # --- KONFIGURASI ---
        self.iou_threshold = 0.5
        self.confidence_threshold = 0.5 
        
        
        print(f"\n🔧 Loading YOLO model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = YOLO(model_path)
        print(f"✓ Model loaded! Mode: STANDARD, Conf: {self.confidence_threshold}, IOU Eval: {self.iou_threshold}")
        
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

    def run_inference_on_images(self, category):
        """Run YOLO inference pada semua gambar dalam kategori"""
        print(f"\n🚀 Running STANDARD inference (Conf={self.confidence_threshold}) on {category} images...")
        
        category_path = self.classified_path / category
        image_files = list(category_path.glob("*.jpg"))
        
        predictions = {}
        
        for img_file in tqdm(image_files, desc=f"Inference {category}"):
            img_path = str(img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # MODIFIKASI: Menambahkan conf=self.confidence_threshold
            results = self.model(img, conf=self.confidence_threshold, verbose=False)
            
            pred_boxes = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    x1, y1, x2, y2 = xyxy
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    
                    pred_boxes.append({
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'confidence': conf
                    })
            
            original_path = img_file.name.replace('_', '/', 1)
            predictions[original_path] = pred_boxes
        
        return predictions

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x, y, w, h]"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        box1_x2 = x1 + w1; box1_y2 = y1 + h1
        box2_x2 = x2 + w2; box2_y2 = y2 + h2
        inter_x1 = max(x1, x2); inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2); inter_y2 = min(box1_y2, box2_y2)
        if inter_x2 < inter_x1 or inter_y2 < inter_y1: return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box1_area = w1 * h1; box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    
    def evaluate_category(self, gt_annotations, predictions, category):
        """Evaluate detection untuk satu kategori"""
        print(f"\n🔍 Evaluating category: {category} (IOU >= {self.iou_threshold})")
        
        category_path = self.classified_path / category
        image_files = [f.name for f in category_path.glob("*.jpg")]
        
        total_gt = 0
        total_pred = 0
        true_positives = 0
        false_positives = 0
        
        # Variabel ini tidak lagi digunakan di metrik akhir, tapi ada di kode asli
        matched_pairs = []
        unmatched_gt = []
        unmatched_pred = []

        for img_file in tqdm(image_files, desc=f"Evaluating {category}"):
            original_path = img_file.replace('_', '/', 1)
            
            if original_path not in gt_annotations:
                continue
            
            gt_boxes = gt_annotations[original_path]
            pred_boxes = predictions.get(original_path, [])
            
            total_gt += len(gt_boxes)
            total_pred += len(pred_boxes)
            
            matched_gt = set()
            matched_pred = set()
            
            for i, pred in enumerate(pred_boxes):
                pred_bbox = pred['bbox']
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    
                    gt_bbox = gt['bbox']
                    iou = self.calculate_iou(pred_bbox, gt_bbox)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= self.iou_threshold:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(i)
                    matched_pairs.append({
                        'image': img_file, 'iou': best_iou, 'confidence': pred.get('confidence', 0)
                    })
                else:
                    false_positives += 1
                    unmatched_pred.append({
                        'image': img_file, 'bbox': pred_bbox
                    })

            # Hitung False Negatives dari GT yang tidak ter-match
            for j, gt in enumerate(gt_boxes):
                if j not in matched_gt:
                    unmatched_gt.append({
                        'image': img_file, 'bbox': gt['bbox']
                    })

        # Hitung FN berdasarkan total GT dikurangi TP
        false_negatives = total_gt - true_positives
        
        precision = true_positives / total_pred if total_pred > 0 else 0
        recall = true_positives / total_gt if total_gt > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'category': category,
            'total_gt': total_gt,
            'total_pred': total_pred,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            # Mengembalikan list ini agar konsisten dengan kode asli Anda
            'matched_pairs': matched_pairs,
            'unmatched_gt': unmatched_gt,
            'unmatched_pred': unmatched_pred
        }
        
        return results
    
    def visualize_results(self, all_results, output_dir="output"):
        """Visualisasi hasil evaluasi (Dikembalikan seperti asli)"""
        print("\n📊 Creating visualizations...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Plot 2x2 Grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        categories, tp_counts, fp_counts, fn_counts = [], [], [], []
        precisions, recalls, f1_scores = [], [], []
        
        for result in all_results:
            categories.append(result['category'].replace('_', ' ').title())
            tp_counts.append(result['true_positives'])
            fp_counts.append(result['false_positives'])
            fn_counts.append(result['false_negatives'])
            precisions.append(result['precision'] * 100)
            recalls.append(result['recall'] * 100)
            f1_scores.append(result['f1_score'] * 100)
        
        # Plot 1: Detection Counts (Stacked Bar)
        ax1 = axes[0, 0]
        x = np.arange(len(categories))
        width = 0.6
        ax1.bar(x, tp_counts, width, label='True Positive (Detected)', color='#2ecc71')
        ax1.bar(x, fp_counts, width, bottom=tp_counts, label='False Positive', color='#e74c3c')
        bottom_fn = np.array(tp_counts) + np.array(fp_counts)
        ax1.bar(x, fn_counts, width, bottom=bottom_fn, label='False Negative (Missed)', color='#95a5a6')
        ax1.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Detection Results per Category (Standard Mode)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=15, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Success Rate (Text) - DIKEMBALIKAN
        ax2 = axes[0, 1]
        ax2.axis('off')
        
        for idx, result in enumerate(all_results):
            detected = result['true_positives']
            missed = result['false_negatives']
            total = detected + missed # Total GT = TP + FN
            
            if total > 0:
                rate = (detected / total) * 100
                color = '#2ecc71' if rate >= 70 else '#f39c12' if rate >= 50 else '#e74c3c'
                
                ax2.text(0.5, 0.9 - idx*0.3, f"{categories[idx]}", 
                         ha='center', fontsize=11, fontweight='bold')
                ax2.text(0.5, 0.85 - idx*0.3, 
                         f"Detected: {detected}/{total} ({rate:.1f}%)",
                         ha='center', fontsize=10, color=color)
        
        ax2.set_title('Detection Success Rate (TP vs Total GT)', fontsize=14, fontweight='bold', pad=20)
        
        # Plot 3: Precision, Recall, F1-Score
        ax3 = axes[1, 0]
        x = np.arange(len(categories))
        width = 0.25
        ax3.bar(x - width, precisions, width, label='Precision', color='#3498db')
        ax3.bar(x, recalls, width, label='Recall', color='#e67e22')
        ax3.bar(x + width, f1_scores, width, label='F1-Score', color='#9b59b6')
        ax3.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Precision, Recall & F1-Score', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories, rotation=15, ha='right')
        ax3.legend()
        ax3.set_ylim(0, 105)
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Comparison Table
        ax4 = axes[1, 1]
        ax4.axis('tight'); ax4.axis('off')
        table_data = [['Category', 'GT', 'Det.', 'TP', 'FP', 'FN', 'Prec', 'Rec', 'F1']]
        for result in all_results:
            row = [
                result['category'].replace('_', '\n'),
                result['total_gt'], result['total_pred'],
                result['true_positives'], result['false_positives'], result['false_negatives'],
                f"{result['precision']*100:.1f}%", f"{result['recall']*100:.1f}%", f"{result['f1_score']*100:.1f}%"
            ]
            table_data.append(row)
        
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                          colWidths=[0.15, 0.08, 0.08, 0.08, 0.08, 0.08, 0.12, 0.12, 0.12])
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 2)
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                if i % 2 == 0: table[(i, j)].set_facecolor('#ecf0f1')
        ax4.set_title('Detailed Metrics Summary', fontsize=14, fontweight='bold', pad=20)
        
        # Tambahkan judul utama dengan info konfigurasi
        plt.suptitle(f'Standard Mode Evaluation (Conf={self.confidence_threshold}, IOU={self.iou_threshold})',
                     fontsize=16, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
        output_file = output_path / 'evaluation_results_standard_0_5.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
        
        # 2. Individual category comparison plot (DIKEMBALIKAN)
        self._plot_category_comparison(all_results, output_path)

    def _plot_category_comparison(self, all_results, output_path):
        """Plot perbandingan antar kategori (Dikembalikan seperti asli)"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        categories = [r['category'].replace('_', ' ').title() for r in all_results]
        detected = [r['true_positives'] for r in all_results]
        missed = [r['false_negatives'] for r in all_results]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, detected, width, label='Successfully Detected (TP)', 
                       color='#27ae60', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, missed, width, label='Failed to Detect (FN)', 
                       color='#c0392b', edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Face Category', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Faces', fontsize=13, fontweight='bold')
        ax.set_title(f'YOLO Face Detection Success vs Failure (Standard Mode, C={self.confidence_threshold}, IOU={self.iou_threshold})', 
                     fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_file = output_path / 'detection_success_vs_failure_standard_0_5.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()

    def print_summary(self, all_results):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("📈 YOLO MODEL EVALUATION SUMMARY (STANDARD MODE)")
        print(f"Confidence Threshold: {self.confidence_threshold} | IOU Evaluation: {self.iou_threshold}")
        print("="*80)
        
        for result in all_results:
            print(f"\n{result['category'].upper().replace('_', ' ')}")
            print("-" * 80)
            print(f"  Ground Truth Faces    : {result['total_gt']:,}")
            print(f"  Predicted Faces       : {result['total_pred']:,}")
            print(f"  True Positives (TP)   : {result['true_positives']:,}")
            print(f"  False Positives (FP)  : {result['false_positives']:,}")
            print(f"  False Negatives (FN)  : {result['false_negatives']:,}")
            print(f"  Precision             : {result['precision']*100:.2f}%")
            print(f"  Recall                : {result['recall']*100:.2f}%")
            print(f"  F1-Score              : {result['f1_score']*100:.2f}%")
        
        print("\n" + "="*80)

    def run(self):
        """Run complete evaluation"""
        print("="*80)
        print("🎯 YOLO MODEL EVALUATION ON WIDER FACE (STANDARD MODE)")
        print(f"🎯 CONF = {self.confidence_threshold}, IOU = {self.iou_threshold}")
        print("="*80)
        
        gt_annotations = self.parse_ground_truth()
        
        all_results = []
        all_predictions = {}
        
        for category in self.categories:
            predictions = self.run_inference_on_images(category)
            all_predictions.update(predictions)
            
            result = self.evaluate_category(gt_annotations, predictions, category)
            all_results.append(result)
        
        self.print_summary(all_results)
        
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        # Panggil visualisasi DENGAN output_dir
        self.visualize_results(all_results, output_dir)
        
        output_file = output_dir / "predictions_standard_0_5.json"
        with open(output_file, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        print(f"\n✓ Predictions saved to: {output_file}")
        
        output_file = output_dir / "evaluation_results_standard_0_5.json"
        with open(output_file, 'w') as f:
            # Hapus list besar untuk JSON yang lebih bersih
            clean_results = []
            for r in all_results:
                clean_r = {k: v for k, v in r.items() 
                           if k not in ['matched_pairs', 'unmatched_gt', 'unmatched_pred']}
                clean_results.append(clean_r)
            json.dump(clean_results, f, indent=2)
        print(f"✓ Results saved to: {output_file}")
        
        print("\n🎉 Evaluation completed!")
        return all_results


if __name__ == "__main__":
    evaluator = StandardWiderFaceEvaluator(
        base_path="data/dataset/widerface",
        model_path="models/yolo11s-pose-max/yolo11s_pose/weights/best.pt"
    )
    results = evaluator.run()