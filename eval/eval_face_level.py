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


class FaceLevelWiderFaceEvaluator:
    """
    Evaluator untuk WIDER Face dengan face-level classification
    Compatible dengan official WIDER Face evaluation methodology
    """
    
    def __init__(self, 
                 base_path="data/dataset/widerface",
                 annotations_file="data/dataset/widerface/face_level_annotations/face_level_gt.json",
                 model_path="models/yolo11s-pose-default/yolo11s_pose/weights/best.pt",
                 device='cuda:0',
                 use_sahi=True,
                 use_enhancer=False,
                 bounded_enhancement=False,
                 face_size_threshold=50,
                 slicing_strategy='uniform'):
        
        self.base_path = Path(base_path)
        self.images_path = self.base_path / "WIDER_val" / "images"
        
        # Load face-level annotations
        print(f"\n📖 Loading face-level annotations from: {annotations_file}")
        with open(annotations_file, 'r') as f:
            self.face_level_gt = json.load(f)
        print(f"✓ Loaded annotations for {len(self.face_level_gt)} images")
        
        # Difficulty levels
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
        
        # Temp directory for enhanced images
        self.temp_enh_dir = Path("output") / "eval_facelevel_temp_enhanced"
        
        # SAHI configuration
        if slicing_strategy == 'uniform':
            self.sahi_config = {
                'slice_height': 640, 'slice_width': 640, 'overlap_ratio': 0.2,
                'confidence_threshold': 0.01, 'postprocess_match_threshold': 0.5,
                'postprocess_class_threshold': 0.25, 'postprocess_type': 'NMS'
            }
        else:  # adaptive
            self.sahi_config = {
                'confidence_threshold': 0.01, 'postprocess_match_threshold': 0.5,
                'postprocess_class_threshold': 0.25, 'postprocess_type': 'NMS',
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
        
        self.mode_string = " -> ".join(mode_parts) if mode_parts else "BASELINE"
    
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
            return True, f"Small faces detected (ratio: {small_face_ratio:.2f})", {}
        
        return False, "Faces are large enough", {}
    
    def get_slice_size_adaptive(self, w, h):
        """Adaptive slice size based on image dimensions"""
        max_dim = max(w, h)
        if max_dim > 2500:
            return 512
        elif max_dim > 1500:
            return 416
        else:
            return 320
    
    def run_inference(self, img_path):
        """
        Run inference on single image
        Returns: list of predictions [x, y, w, h, confidence]
        """
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
            # SAHI mode
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
                postprocess_match_threshold=self.sahi_config['postprocess_match_threshold'],
                postprocess_class_agnostic=True, verbose=0
            )
            
            for det in result.object_prediction_list:
                x, y, w, h = det.bbox.to_xywh()
                pred_boxes.append({
                    'bbox': [x, y, w, h],
                    'confidence': det.score.value
                })
        
        else:
            # Standard mode
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
        
        return pred_boxes
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x, y, w, h]"""
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
    
    def evaluate_difficulty(self, difficulty='easy'):
        """
        Evaluate one difficulty level with ignore mechanism
        """
        print(f"\n🔍 Evaluating {difficulty.upper()} set...")
        
        total_gt = 0
        all_detections = []
        false_negatives = 0
        
        # Setup temp directory if using enhancer
        if self.use_enhancer and not self.temp_enh_dir.exists():
            self.temp_enh_dir.mkdir(parents=True, exist_ok=True)
        
        # Iterate through all images
        for img_path, gt_data in tqdm(self.face_level_gt.items(), desc=f"{difficulty.upper()}"):
            # Get valid GT indices for this difficulty
            if difficulty == 'easy':
                valid_indices = gt_data['easy_indices']
            elif difficulty == 'medium':
                valid_indices = gt_data['medium_indices']
            else:  # hard
                valid_indices = gt_data['hard_indices']
            
            # Skip if no valid faces for this difficulty
            if not valid_indices:
                continue
            
            # Get GT faces to evaluate
            all_faces = gt_data['all_faces']
            gt_faces = [all_faces[i] for i in valid_indices]
            total_gt += len(gt_faces)
            
            # Get ignored faces (for FP checking)
            ignored_indices = [i for i in range(len(all_faces)) if i not in valid_indices]
            ignored_faces = [all_faces[i] for i in ignored_indices]
            
            # Run inference
            full_img_path = str(self.images_path / img_path)
            self.enhancement_stats['total_images'] += 1
            pred_boxes = self.run_inference(full_img_path)
            
            # Track which GT faces have been matched
            gt_matched = [False] * len(gt_faces)
            
            # Evaluate predictions
            for pred in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                is_ignored = False
                
                # Check match with valid GT
                for gt_idx, gt_face in enumerate(gt_faces):
                    iou = self.calculate_iou(pred['bbox'], gt_face['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Determine if TP or FP
                if best_iou >= self.iou_threshold and best_gt_idx != -1 and not gt_matched[best_gt_idx]:
                    # True Positive
                    gt_matched[best_gt_idx] = True
                    is_tp = True
                else:
                    # Check if matches ignored face
                    for ignored_face in ignored_faces:
                        iou = self.calculate_iou(pred['bbox'], ignored_face['bbox'])
                        if iou >= self.iou_threshold:
                            is_ignored = True
                            break
                    
                    # False Positive only if not matching ignored face
                    is_tp = False
                
                # Only add to detections if not ignored
                if not is_ignored:
                    all_detections.append({
                        'confidence': pred['confidence'],
                        'is_tp': is_tp
                    })
            
            # Count false negatives (unmatched GT)
            false_negatives += sum(1 for matched in gt_matched if not matched)
        
        # Calculate AP
        ap = self.calculate_average_precision(all_detections, total_gt)
        
        # Calculate metrics at global confidence threshold
        filtered_detections = [d for d in all_detections if d['confidence'] >= self.global_confidence]
        true_positives = sum(1 for d in filtered_detections if d['is_tp'])
        false_positives = len(filtered_detections) - true_positives
        
        precision = true_positives / len(filtered_detections) if filtered_detections else 0
        recall = true_positives / total_gt if total_gt > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'difficulty': difficulty,
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
        """Run complete evaluation"""
        print("="*80)
        print(f"🎯 YOLO EVALUATION - {self.mode_string}")
        print("="*80)
        
        start_time = time.time()
        
        # Evaluate each difficulty
        results = []
        for difficulty in self.difficulties:
            result = self.evaluate_difficulty(difficulty)
            results.append(result)
        
        elapsed_time = time.time() - start_time
        
        # Calculate overall mAP
        overall_map = np.mean([r['ap'] for r in results])
        
        # Print summary
        self.print_summary(results, overall_map, elapsed_time)
        
        # Save results
        self.save_results(results, overall_map, elapsed_time)
        
        # Cleanup
        if self.temp_enh_dir.exists():
            shutil.rmtree(self.temp_enh_dir)
        
        print(f"\n🎉 Evaluation completed in {elapsed_time:.1f}s!")
        print(f"   Final mAP: {overall_map*100:.2f}%")
        
        return results, overall_map
    
    def print_summary(self, results, overall_map, elapsed_time):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print(f"📈 EVALUATION SUMMARY - {self.mode_string}")
        print("="*80)
        
        for result in results:
            print(f"\n{result['difficulty'].upper()}")
            print("-"*80)
            print(f"  Ground Truth      : {result['total_gt']:,}")
            print(f"  Predicted         : {result['total_pred']:,}")
            print(f"  True Positives    : {result['true_positives']:,}")
            print(f"  False Positives   : {result['false_positives']:,}")
            print(f"  False Negatives   : {result['false_negatives']:,}")
            print(f"  Precision         : {result['precision']*100:.2f}%")
            print(f"  Recall            : {result['recall']*100:.2f}%")
            print(f"  F1-Score          : {result['f1_score']*100:.2f}%")
            print(f"  📊 AP             : {result['ap']*100:.2f}%")
        
        print("\n" + "="*80)
        print(f"🎯 OVERALL mAP: {overall_map*100:.2f}%")
        print(f"⏱️  Time: {elapsed_time:.1f}s")
        
        if self.bounded_enhancement:
            enhance_rate = (self.enhancement_stats['enhanced_images'] / 
                          self.enhancement_stats['total_images'] * 100)
            print(f"🔧 Enhanced: {self.enhancement_stats['enhanced_images']}/"
                  f"{self.enhancement_stats['total_images']} ({enhance_rate:.1f}%)")
        
        print("="*80)
    
    def save_results(self, results, overall_map, elapsed_time):
        """Save results to JSON"""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Create clean filename
        mode_str = self.mode_string.lower().replace(" -> ", "_").replace(" ", "_")
        mode_str = "".join(c for c in mode_str if c.isalnum() or c in ('_', '-'))
        
        output_file = output_dir / f"facelevel_{mode_str}_results.json"
        
        json_results = {
            'mode': self.mode_string,
            'overall_map': float(overall_map),
            'elapsed_time': elapsed_time,
            'config': {
                'use_sahi': self.use_sahi,
                'use_enhancer': self.use_enhancer,
                'bounded_enhancement': self.bounded_enhancement,
                'face_size_threshold': self.face_size_threshold,
                'slicing_strategy': self.slicing_strategy,
                'global_confidence': self.global_confidence,
                'iou_threshold': self.iou_threshold
            },
            'results': results
        }
        
        if self.bounded_enhancement:
            json_results['enhancement_stats'] = self.enhancement_stats
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YOLO with Face-Level Classification')
    parser.add_argument('--mode', choices=['standard', 'sahi-uniform', 'sahi-adaptive'], 
                       default='sahi-uniform')
    parser.add_argument('--enhance', action='store_true')
    parser.add_argument('--bounded', action='store_true')
    parser.add_argument('--threshold', type=int, default=50)
    parser.add_argument('--device', default='cuda:0')
    
    args = parser.parse_args()
    
    use_sahi = args.mode.startswith('sahi')
    slicing_strategy = args.mode.split('-')[1] if use_sahi else 'uniform'
    
    evaluator = FaceLevelWiderFaceEvaluator(
        base_path="data/dataset/widerface",
        annotations_file="data/dataset/widerface/face_level_annotations/face_level_gt.json",
        model_path="models/yolo11s-pose-default/yolo11s_pose/weights/best.pt",
        device=args.device,
        use_sahi=use_sahi,
        use_enhancer=args.enhance,
        bounded_enhancement=args.bounded,
        face_size_threshold=args.threshold,
        slicing_strategy=slicing_strategy
    )
    
    evaluator.run()