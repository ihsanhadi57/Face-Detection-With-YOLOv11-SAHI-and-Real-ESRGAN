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

# Import SAHI dan wrapper/enhancer
import sys
sys.path.append(str(Path(__file__).parent.parent))
from sahi.predict import get_sliced_prediction
from utils.yolo_wrapper import YOLOv11PoseDetectionModel
from utils.enhancer import FaceEnhancer


class DualWiderFaceEvaluator:
    """
    Dual Evaluator untuk WIDER Face:
    1. Standard WIDER Face (Easy/Medium/Hard) - untuk publikasi
    2. 6 Sub-Categories - untuk analisis mendalam
    
    Compatible dengan 4 pipeline:
    1. Baseline (YOLO only)
    2. YOLO + SAHI
    3. Enhance + YOLO
    4. Full Pipeline (Enhance + SAHI + YOLO)
    """
    
    def __init__(self, 
                 base_path="data/dataset/widerface",
                 subcategory_file="data/dataset/widerface/subcategory_annotations/subcategory_gt.json",
                 model_path="models/yolo11s-pose-default/yolo11s_pose/weights/best.pt",
                 device='cuda:0',
                 use_sahi=False,
                 use_enhancer=False,
                 bounded_enhancement=False,
                 face_size_threshold=50,
                 slicing_strategy='uniform',
                 sahi_match_thresholds=[0.5], 
                 sahi_match_metric='IOS'): 
        
        self.base_path = Path(base_path)
        self.images_path = self.base_path / "WIDER_val" / "images"
        
        # Load sub-category annotations
        print(f"\n📖 Loading sub-category annotations from: {subcategory_file}")
        with open(subcategory_file, 'r') as f:
            self.subcategory_gt = json.load(f)
        print(f"✓ Loaded annotations for {len(self.subcategory_gt)} images")
        
        # 6 Sub-categories
        self.subcategories = [
            'large_clear', 'large_degraded',
            'medium_clear', 'medium_degraded',
            'small_clear', 'small_degraded'
        ]
        
        # Standard WIDER Face difficulties
        self.difficulties = ['easy', 'medium', 'hard']
        
        # Evaluation parameters
        self.iou_threshold = 0.5
        self.global_confidence = 0.25
        self.inference_confidence = 0.01 if use_sahi else 0.5
        
        # Pipeline configuration
        self.use_sahi = use_sahi
        self.use_enhancer = use_enhancer
        self.bounded_enhancement = bounded_enhancement
        self.face_size_threshold = face_size_threshold
        self.slicing_strategy = slicing_strategy
        
        # Enhancement statistics
        self.enhancement_stats = {
            'total_images': 0,
            'enhanced_images': 0,
            'skipped_images': 0
        }
        
        # Cache for predictions (avoid re-inference)
        self.prediction_cache = {}
        
        # Temp directory
        self.temp_enh_dir = Path("output") / "eval_dual_temp_enhanced"
        
        # SAHI configuration
        if slicing_strategy == 'uniform':
            self.sahi_config = {
                'slice_height': 640, 'slice_width': 640, 'overlap_ratio': 0.25,
                'confidence_threshold': 0.01,
                'postprocess_match_thresholds': sahi_match_thresholds, # Use the list
                'postprocess_match_metric': sahi_match_metric, # New metric
                'postprocess_class_threshold': 0.25,
                'postprocess_type': 'NMS'
            }
        else:
            self.sahi_config = {
                'confidence_threshold': 0.01,
                'postprocess_match_thresholds': sahi_match_thresholds, # Use the list
                'postprocess_match_metric': sahi_match_metric, # New metric
                'postprocess_class_threshold': 0.25,
                'postprocess_type': 'NMS',
                'overlap_ratio': 0.2
            }
        
        # Initialize models
        print(f"\n🔧 Loading models...")
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
                print(f"   ✓ Enhancer loaded! (Scale: {self.face_enhancer.scale}x)")
            except Exception as e:
                print(f"   ❌ Failed to load enhancer: {e}")
                self.use_enhancer = False
        
        self._build_mode_string()
        print(f"✓ Models loaded! Mode: {self.mode_string}")
    
    def _build_mode_string(self):
        """Build mode string for display"""
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
        
        self.mode_string = " + ".join(mode_parts) if mode_parts else "BASELINE"
    
    def quick_face_analysis(self, img):
        """Quick analysis untuk bounded enhancement"""
        if img is None:
            return False, "Image load failed", {}
        
        results = self.detection_model.model(img, conf=0.05, verbose=False)
        
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return True, "No faces detected", {}
        
        face_sizes = []
        for i in range(len(results[0].boxes)):
            xyxy = results[0].boxes.xyxy[i].cpu().numpy()
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]
            max_dim = max(width, height)
            face_sizes.append(max_dim)
        
        avg_face_size = np.mean(face_sizes)
        small_faces_count = sum(1 for size in face_sizes if size < self.face_size_threshold)
        small_face_ratio = small_faces_count / len(face_sizes)
        
        if small_face_ratio > 0.5 or avg_face_size < self.face_size_threshold:
            return True, f"Small faces detected", {}
        
        return False, "Faces are large enough", {}
    
    def get_slice_size_adaptive(self, w, h):
        """Adaptive slice size"""
        max_dim = max(w, h)
        if max_dim > 2500:
            return 512
        elif max_dim > 1500:
            return 416
        else:
            return 320
    
    def run_inference(self, img_path):
        """Run inference with caching"""
        # Check cache
        if img_path in self.prediction_cache:
            return self.prediction_cache[img_path]
        
        img = cv2.imread(img_path)
        if img is None:
            return []
        
        inference_img = img
        was_enhanced = False
        
        # Enhancement phase
        if self.use_enhancer and self.face_enhancer:
            enhance_decision = False
            
            if self.bounded_enhancement:
                enhance_decision, _, _ = self.quick_face_analysis(img)
            else:
                enhance_decision = True
            
            if enhance_decision:
                enhanced_image, success = self.face_enhancer.enhance_image(img)
                if success:
                    inference_img = enhanced_image
                    was_enhanced = True
                    self.enhancement_stats['enhanced_images'] += 1
            else:
                self.enhancement_stats['skipped_images'] += 1
        
        # Detection phase
        pred_boxes = []
        
        if self.use_sahi:
            h, w = inference_img.shape[:2]
            if self.slicing_strategy == 'uniform':
                slice_h = self.sahi_config['slice_height']
                slice_w = self.sahi_config['slice_width']
            else:
                slice_h = slice_w = self.get_slice_size_adaptive(w, h)
            
            result = get_sliced_prediction(
                inference_img, self.detection_model,
                slice_height=slice_h, slice_width=slice_w,
                overlap_height_ratio=self.sahi_config['overlap_ratio'],
                overlap_width_ratio=self.sahi_config['overlap_ratio'],
                postprocess_type=self.sahi_config['postprocess_type'],
                postprocess_match_metric=self.sahi_config['postprocess_match_metric'],  # ✅ Tambahkan
                postprocess_match_threshold=self.sahi_config['postprocess_match_thresholds'][0],  # ✅ Ambil index [0]
                postprocess_class_agnostic=True, verbose=0
            )
            
            for det in result.object_prediction_list:
                x, y, w, h = det.bbox.to_xywh()
                pred_boxes.append({
                    'bbox': [x, y, w, h],
                    'confidence': det.score.value
                })
        else:
            results = self.detection_model.model(inference_img, conf=self.inference_confidence, verbose=False)
            boxes = results[0].boxes
            
            if len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = xyxy[i]
                    w, h = x2 - x1, y2 - y1
                    
                    pred_boxes.append({
                        'bbox': [x1, y1, w, h],
                        'confidence': confs[i]
                    })
        
        # Scale down if enhanced
        if was_enhanced and self.face_enhancer.scale > 1:
            scale = self.face_enhancer.scale
            for pred in pred_boxes:
                pred['bbox'] = [coord / scale for coord in pred['bbox']]
        
        # Cache predictions
        self.prediction_cache[img_path] = pred_boxes
        
        return pred_boxes
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        box1_x2, box1_y2 = x1 + w1, y1 + h1
        box2_x2, box2_y2 = x2 + w2, y2 + h2
        
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = (w1 * h1) + (w2 * h2) - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def calculate_average_precision(self, all_detections, total_gt):
        """Calculate AP using 11-point interpolation"""
        if total_gt == 0 or not all_detections:
            return 0.0
        
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        tp_cumsum = np.cumsum([d['is_tp'] for d in all_detections])
        fp_cumsum = np.cumsum([not d['is_tp'] for d in all_detections])
        
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # 11-point interpolation
        ap = 0.0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        
        return ap
    
    def map_subcategory_to_difficulty(self, category):
        """Map sub-category to WIDER Face difficulty levels"""
        difficulties = []
        
        # Easy: only large_clear
        if category == 'large_clear':
            difficulties.append('easy')
        
        # Medium: large_clear + large_degraded + medium_clear
        if category in ['large_clear', 'large_degraded', 'medium_clear']:
            difficulties.append('medium')
        
        # Hard: all categories
        difficulties.append('hard')
        
        return difficulties
    
    def evaluate_single_set(self, category_type, category_name, valid_categories):
        """
        Generic evaluation function
        category_type: 'subcategory' or 'difficulty'
        category_name: e.g., 'large_clear' or 'easy'
        valid_categories: list of subcategories to include
        """
        total_gt = 0
        all_detections = []
        false_negatives = 0
        
        # Setup temp directory
        if self.use_enhancer and not self.temp_enh_dir.exists():
            self.temp_enh_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path, gt_data in self.subcategory_gt.items():
            # Get valid face indices
            valid_indices = []
            for cat in valid_categories:
                valid_indices.extend(gt_data[cat])
            
            valid_indices = list(set(valid_indices))  # Remove duplicates
            
            if not valid_indices:
                continue
            
            # Get GT faces
            all_faces = gt_data['all_faces']
            gt_faces = [all_faces[i] for i in valid_indices]
            total_gt += len(gt_faces)
            
            # Get ignored faces
            ignored_indices = [i for i in range(len(all_faces)) if i not in valid_indices]
            ignored_faces = [all_faces[i] for i in ignored_indices]
            
            # Run inference (cached)
            full_img_path = str(self.images_path / img_path)
            if img_path not in self.prediction_cache:
                self.enhancement_stats['total_images'] += 1
            pred_boxes = self.run_inference(full_img_path)
            
            # Match predictions
            gt_matched = [False] * len(gt_faces)
            
            for pred in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                is_ignored = False
                
                # Check valid GT
                for gt_idx, gt_face in enumerate(gt_faces):
                    iou = self.calculate_iou(pred['bbox'], gt_face['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Determine TP/FP
                if best_iou >= self.iou_threshold and best_gt_idx != -1 and not gt_matched[best_gt_idx]:
                    gt_matched[best_gt_idx] = True
                    is_tp = True
                else:
                    # Check ignored faces
                    for ignored_face in ignored_faces:
                        iou = self.calculate_iou(pred['bbox'], ignored_face['bbox'])
                        if iou >= self.iou_threshold:
                            is_ignored = True
                            break
                    is_tp = False
                
                if not is_ignored:
                    all_detections.append({
                        'confidence': pred['confidence'],
                        'is_tp': is_tp
                    })
            
            false_negatives += sum(1 for matched in gt_matched if not matched)
        
        # Calculate metrics
        ap = self.calculate_average_precision(all_detections, total_gt)
        
        filtered_detections = [d for d in all_detections if d['confidence'] >= self.global_confidence]
        true_positives = sum(1 for d in filtered_detections if d['is_tp'])
        false_positives = len(filtered_detections) - true_positives
        
        precision = true_positives / len(filtered_detections) if filtered_detections else 0
        recall = true_positives / total_gt if total_gt > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'category': category_name,
            'total_gt': total_gt,
            'total_pred': len(filtered_detections),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'ap': ap
        }
    
    def run(self):
        """Run complete dual evaluation"""
        print("="*80)
        print(f"🎯 DUAL EVALUATION - {self.mode_string}")
        print("="*80)
        
        start_time = time.time()
        
        # === PART 1: Sub-Category Evaluation ===
        print("\n" + "="*80)
        print("📊 PART 1: SUB-CATEGORY EVALUATION (6 Categories)")
        print("="*80)
        
        subcategory_results = []
        for category in tqdm(self.subcategories, desc="Sub-categories"):
            result = self.evaluate_single_set('subcategory', category, [category])
            subcategory_results.append(result)
        
        # === PART 2: Standard WIDER Face Evaluation ===
        print("\n" + "="*80)
        print("📊 PART 2: STANDARD WIDER FACE EVALUATION (Easy/Medium/Hard)")
        print("="*80)
        
        difficulty_mapping = {
            'easy': ['large_clear'],
            'medium': ['large_clear', 'large_degraded', 'medium_clear'],
            'hard': self.subcategories  # All categories
        }
        
        difficulty_results = []
        for difficulty in tqdm(self.difficulties, desc="Difficulties"):
            result = self.evaluate_single_set('difficulty', difficulty, 
                                             difficulty_mapping[difficulty])
            difficulty_results.append(result)
        
        elapsed_time = time.time() - start_time
        
        # Calculate summary metrics
        summary = self.calculate_summary_metrics(subcategory_results, difficulty_results)
        
        # Print results
        self.print_results(subcategory_results, difficulty_results, summary, elapsed_time)
        
        # Save results
        self.save_results(subcategory_results, difficulty_results, summary, elapsed_time)
        
        # Generate visualizations
        self.generate_visualizations(subcategory_results, difficulty_results, summary)
        
        # Cleanup
        if self.temp_enh_dir.exists():
            shutil.rmtree(self.temp_enh_dir)
        
        print(f"\n🎉 Dual evaluation completed in {elapsed_time:.1f}s!")
        
        return subcategory_results, difficulty_results, summary
    
    def calculate_summary_metrics(self, subcategory_results, difficulty_results):
        """Calculate summary metrics"""
        summary = {}
        
        # Sub-category metrics
        summary['subcategory_map'] = np.mean([r['ap'] for r in subcategory_results])
        summary['large_map'] = np.mean([r['ap'] for r in subcategory_results if 'large' in r['category']])
        summary['medium_map'] = np.mean([r['ap'] for r in subcategory_results if 'medium' in r['category']])
        summary['small_map'] = np.mean([r['ap'] for r in subcategory_results if 'small' in r['category']])
        summary['clear_map'] = np.mean([r['ap'] for r in subcategory_results if 'clear' in r['category']])
        summary['degraded_map'] = np.mean([r['ap'] for r in subcategory_results if 'degraded' in r['category']])
        
        # Standard WIDER Face metrics
        summary['standard_map'] = np.mean([r['ap'] for r in difficulty_results])
        summary['easy_ap'] = next(r['ap'] for r in difficulty_results if r['category'] == 'easy')
        summary['medium_ap'] = next(r['ap'] for r in difficulty_results if r['category'] == 'medium')
        summary['hard_ap'] = next(r['ap'] for r in difficulty_results if r['category'] == 'hard')
        
        return summary
    
    def print_results(self, subcategory_results, difficulty_results, summary, elapsed_time):
        """Print comprehensive results"""
        print("\n" + "="*80)
        print(f"📈 EVALUATION RESULTS - {self.mode_string}")
        print("="*80)
        
        # Part 1: Standard WIDER Face
        print("\n🎯 STANDARD WIDER FACE EVALUATION (for Publication)")
        print("-"*80)
        for result in difficulty_results:
            print(f"\n{result['category'].upper()}")
            print(f"  AP: {result['ap']*100:.2f}% | Precision: {result['precision']*100:.2f}% | "
                  f"Recall: {result['recall']*100:.2f}% | GT: {result['total_gt']:,}")
        
        print(f"\n  📊 Standard mAP: {summary['standard_map']*100:.2f}%")
        
        # Part 2: Sub-Category Analysis
        print("\n\n🔍 SUB-CATEGORY ANALYSIS (for Deep Insights)")
        print("-"*80)
        for result in subcategory_results:
            print(f"\n{result['category'].upper().replace('_', ' ')}")
            print(f"  AP: {result['ap']*100:.2f}% | Precision: {result['precision']*100:.2f}% | "
                  f"Recall: {result['recall']*100:.2f}% | GT: {result['total_gt']:,}")
        
        print(f"\n  📊 Sub-Category mAP: {summary['subcategory_map']*100:.2f}%")
        
        # Grouped Analysis
        print("\n\n📊 GROUPED ANALYSIS")
        print("-"*80)
        print(f"By Size:")
        print(f"  Large:  {summary['large_map']*100:.2f}%")
        print(f"  Medium: {summary['medium_map']*100:.2f}%")
        print(f"  Small:  {summary['small_map']*100:.2f}%")
        
        print(f"\nBy Condition:")
        print(f"  Clear:    {summary['clear_map']*100:.2f}%")
        print(f"  Degraded: {summary['degraded_map']*100:.2f}%")
        
        # Summary
        print("\n" + "="*80)
        print(f"⏱️  Time: {elapsed_time:.1f}s")
        if self.bounded_enhancement:
            enhance_rate = (self.enhancement_stats['enhanced_images'] / 
                          self.enhancement_stats['total_images'] * 100)
            print(f"🔧 Enhanced: {self.enhancement_stats['enhanced_images']}/"
                  f"{self.enhancement_stats['total_images']} ({enhance_rate:.1f}%)")
        print("="*80)
    
    def save_results(self, subcategory_results, difficulty_results, summary, elapsed_time):
        """Save results to JSON"""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        mode_str = self.mode_string.lower().replace(" + ", "_").replace(" ", "_")
        mode_str = "".join(c for c in mode_str if c.isalnum() or c in ('_', '-'))
        
        output_file = output_dir / f"dual_{mode_str}_results.json"
        
        json_results = {
            'mode': self.mode_string,
            'standard_widerface': {
                'map': float(summary['standard_map']),
                'easy_ap': float(summary['easy_ap']),
                'medium_ap': float(summary['medium_ap']),
                'hard_ap': float(summary['hard_ap']),
                'results': difficulty_results
            },
            'subcategory_analysis': {
                'map': float(summary['subcategory_map']),
                'grouped': {
                    'large': float(summary['large_map']),
                    'medium': float(summary['medium_map']),
                    'small': float(summary['small_map']),
                    'clear': float(summary['clear_map']),
                    'degraded': float(summary['degraded_map'])
                },
                'results': subcategory_results
            },
            'elapsed_time': elapsed_time,
            'config': {
                'use_sahi': self.use_sahi,
                'use_enhancer': self.use_enhancer,
                'bounded_enhancement': self.bounded_enhancement,
                'face_size_threshold': self.face_size_threshold,
                'slicing_strategy': self.slicing_strategy,
                'global_confidence': self.global_confidence,
                'iou_threshold': self.iou_threshold
            }
        }
        
        if self.bounded_enhancement:
            json_results['enhancement_stats'] = self.enhancement_stats
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    def generate_visualizations(self, subcategory_results, difficulty_results, summary):
        """Generate simplified visualizations (3 charts only)"""
        output_dir = Path("output")
        mode_str = self.mode_string.lower().replace(" + ", "_").replace(" ", "_")
        mode_str = "".join(c for c in mode_str if c.isalnum() or c in ('_', '-'))
        
        # Set style
        sns.set_style("whitegrid")
        
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        
        # 1. Standard WIDER Face AP (Left)
        ax1 = fig.add_subplot(gs[0, 0])
        difficulties = ['Easy', 'Medium', 'Hard']
        standard_aps = [summary['easy_ap']*100, summary['medium_ap']*100, summary['hard_ap']*100]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        bars1 = ax1.bar(difficulties, standard_aps, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.axhline(summary['standard_map']*100, color='blue', linestyle='--', linewidth=2.5, 
                   label=f'mAP: {summary["standard_map"]*100:.2f}%')
        ax1.set_ylabel('AP (%)', fontsize=13, fontweight='bold')
        ax1.set_title('Standard WIDER Face AP', fontsize=14, fontweight='bold', pad=15)
        ax1.legend(fontsize=11, loc='lower left')
        ax1.set_ylim([0, 100])
        ax1.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars1, standard_aps):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 2. Sub-Category AP Analysis (Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        categories = [r['category'].replace('_', '\n').title() for r in subcategory_results]
        subcategory_aps = [r['ap']*100 for r in subcategory_results]
        colors_sub = ['#27ae60', '#16a085', '#3498db', '#2980b9', '#f39c12', '#e67e22']
        bars2 = ax2.bar(categories, subcategory_aps, color=colors_sub, alpha=0.8, edgecolor='black', linewidth=2)
        ax2.axhline(summary['subcategory_map']*100, color='red', linestyle='--', linewidth=2.5, 
                   label=f'mAP: {summary["subcategory_map"]*100:.2f}%')
        ax2.set_ylabel('AP (%)', fontsize=13, fontweight='bold')
        ax2.set_title('Sub-Category AP Analysis', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(fontsize=11, loc='lower left')
        ax2.set_ylim([0, 100])
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=0, labelsize=10)
        for bar, val in zip(bars2, subcategory_aps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 3. Overall mAP Comparison (Right)
        ax3 = fig.add_subplot(gs[0, 2])
        overall_labels = ['Standard\nmAP', 'Sub-Category\nmAP']
        overall_aps = [summary['standard_map']*100, summary['subcategory_map']*100]
        colors_overall = ['#3498db', '#9b59b6']
        bars3 = ax3.bar(overall_labels, overall_aps, color=colors_overall, alpha=0.8, 
                       edgecolor='black', linewidth=2, width=0.6)
        ax3.set_ylabel('mAP (%)', fontsize=13, fontweight='bold')
        ax3.set_title('Overall mAP Comparison', fontsize=14, fontweight='bold', pad=15)
        ax3.set_ylim([0, 100])
        ax3.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars3, overall_aps):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
        
       
        # Main title
        fig.suptitle(f'WIDER Face Dual Evaluation Results - {self.mode_string}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save
        output_file = output_dir / f"dual_{mode_str}_visualization.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Visualization saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dual Evaluation: Sub-Category + Standard WIDER Face')
    parser.add_argument('--mode', choices=['baseline', 'sahi', 'enhance', 'full'], 
                       default='baseline',
                       help='Pipeline: baseline, sahi, enhance, or full')
    parser.add_argument('--bounded', action='store_true',
                       help='Use bounded enhancement')
    parser.add_argument('--threshold', type=int, default=50,
                       help='Face size threshold for bounded enhancement')
    parser.add_argument('--device', default='cuda:0')
    
    args = parser.parse_args()
    
    # Parse mode
    use_sahi = args.mode in ['sahi', 'full']
    use_enhancer = args.mode in ['enhance', 'full']
    
    print("\n" + "="*80)
    print("🚀 DUAL EVALUATION PIPELINE")
    print("="*80)
    print(f"Mode: {args.mode.upper()}")
    print(f"  - YOLO: ✓")
    print(f"  - SAHI: {'✓' if use_sahi else '✗'}")
    print(f"  - Enhancer: {'✓' if use_enhancer else '✗'}")
    if use_enhancer and args.bounded:
        print(f"  - Bounded Enhancement: ✓ (threshold: {args.threshold}px)")
    print("="*80)
    
    evaluator = DualWiderFaceEvaluator(
        base_path="data/dataset/widerface",
        subcategory_file="data/dataset/widerface/subcategory_annotations/subcategory_gt.json",
        model_path="models/yolo11s-pose-default/yolo11s_pose/weights/best.pt",
        device=args.device,
        use_sahi=use_sahi,
        use_enhancer=use_enhancer,
        bounded_enhancement=args.bounded,
        face_size_threshold=args.threshold,
        slicing_strategy='uniform'
    )
    
    evaluator.run()