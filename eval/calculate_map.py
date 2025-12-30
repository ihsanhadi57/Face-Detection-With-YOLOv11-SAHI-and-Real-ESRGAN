import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_gt_file(gt_path):
    """Parses ground truth file. GT format: x, y, w, h"""
    with open(gt_path, 'r') as f:
        lines = f.readlines()

    gt_data = {}
    i = 0
    while i < len(lines):
        image_path = lines[i].strip()
        if not image_path.endswith('.jpg'):
            i += 1
            continue
        
        num_faces = int(lines[i+1].strip())
        bboxes = []
        for j in range(num_faces):
            box_line = lines[i+2+j].strip().split()
            x1, y1, w, h = [int(v) for v in box_line[:4]]
            if w > 0 and h > 0 and x1 >= 0 and y1 >= 0:
                # Convert (x, y, w, h) -> (x1, y1, x2, y2)
                bboxes.append([x1, y1, x1 + w, y1 + h])
        
        normalized_path = os.path.normpath(image_path)
        gt_data[normalized_path] = np.array(bboxes, dtype=np.float32) if bboxes else np.empty((0, 4), dtype=np.float32)
        
        i += 2 + num_faces
        
    return gt_data

def parse_prediction_files(pred_dir):
    """Parses prediction files. Format: face score x1 y1 x2 y2"""
    pred_data = {}
    
    file_count = sum(1 for _ in glob(os.path.join(pred_dir, '**', '*.txt'), recursive=True))
    
    processed = 0
    for subdir, _, files in os.walk(pred_dir):
        for file in files:
            if file.endswith('.txt'):
                processed += 1
                if processed % 100 == 0:
                    print(f"  Processed {processed}/{file_count} files...")
                
                relative_subdir = os.path.relpath(subdir, pred_dir)
                image_name_without_ext = os.path.splitext(file)[0]
                
                if relative_subdir != '.':
                    image_key = os.path.join(relative_subdir, image_name_without_ext + '.jpg')
                else:
                    image_key = image_name_without_ext + '.jpg'
                
                image_key = os.path.normpath(image_key)

                with open(os.path.join(subdir, file), 'r') as f:
                    lines = f.readlines()
                
                boxes = []
                scores = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 6 and parts[0] == 'face':
                        try:
                            score = float(parts[1])
                            x1, y1, x2, y2 = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                            
                            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and score > 0:
                                scores.append(score)
                                boxes.append([x1, y1, x2, y2])
                        except ValueError:
                            continue
                
                pred_data[image_key] = {
                    'boxes': np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32),
                    'scores': np.array(scores, dtype=np.float32) if scores else np.empty(0, dtype=np.float32)
                }

    return pred_data

def voc_ap(rec, prec):
    """Calculate AP using VOC method"""
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def plot_pr_curve(rec, prec, ap, output_path, method_name="Model"):
    """Plot Precision-Recall curve"""
    plt.figure(figsize=(10, 7))
    plt.plot(rec, prec, linewidth=2, label=f'{method_name} (AP = {ap:.4f})')
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def calculate_metrics(gt_data, pred_data, iou_threshold=0.5):
    """Calculate mAP and other metrics"""
    image_ids = []
    confidence = []
    boxes = []
    
    for img_key, preds in pred_data.items():
        if preds['boxes'].shape[0] > 0:
            for i in range(preds['boxes'].shape[0]):
                image_ids.append(img_key)
                confidence.append(preds['scores'][i])
                boxes.append(preds['boxes'][i])

    if len(confidence) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, None, None

    confidence = np.array(confidence)
    boxes = np.array(boxes)
    sorted_ind = np.argsort(-confidence)
    boxes = boxes[sorted_ind]
    image_ids = [image_ids[i] for i in sorted_ind]
    confidence = confidence[sorted_ind]

    num_gt_boxes = sum(len(v) for v in gt_data.values())
    if num_gt_boxes == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, None, None

    tp = np.zeros(len(image_ids))
    fp = np.zeros(len(image_ids))
    detected_gt = {img_key: np.zeros(gt_boxes.shape[0], dtype=bool) 
                   for img_key, gt_boxes in gt_data.items()}

    for i in tqdm(range(len(image_ids)), desc="Calculating metrics"):
        img_key = image_ids[i]
        bb = boxes[i]
        gt_boxes = gt_data.get(img_key, np.empty((0, 4), dtype=np.float32))
        
        if gt_boxes.shape[0] == 0:
            fp[i] = 1
            continue

        ixmin = np.maximum(gt_boxes[:, 0], bb[0])
        iymin = np.maximum(gt_boxes[:, 1], bb[1])
        ixmax = np.minimum(gt_boxes[:, 2], bb[2])
        iymax = np.minimum(gt_boxes[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        inters = iw * ih
        
        union = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                 (gt_boxes[:, 2] - gt_boxes[:, 0]) *
                 (gt_boxes[:, 3] - gt_boxes[:, 1]) - inters)
        
        ious = inters / (union + 1e-10)
        jmax = np.argmax(ious)
        iou_max = ious[jmax]

        if iou_max >= iou_threshold:
            if not detected_gt[img_key][jmax]:
                tp[i] = 1
                detected_gt[img_key][jmax] = True
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    fp_cumsum = np.cumsum(fp)
    tp_cumsum = np.cumsum(tp)
    recall = tp_cumsum / float(num_gt_boxes)
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
    ap = voc_ap(recall, precision)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1)
    
    return (ap, f1[best_f1_idx], precision[best_f1_idx], 
            recall[best_f1_idx], confidence[best_f1_idx], recall, precision)

if __name__ == '__main__':
    from glob import glob
    
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    GT_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'dataset', 'UFDD', 
                           'UFDD-annotationfile', 'UFDD_split', 'UFDD_val_bbx_gt.txt')
    
    # GANTI INI untuk evaluasi baseline atau SAHI
    PRED_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'eval_results', 'esrgan_sahi_ufdd')
    PLOT_PATH = os.path.join(PRED_DIR, 'precision_recall_curve.png')

    print("="*70)
    print("EVALUATION - mAP Calculation")
    print("="*70)
    print(f"\nGT Path  : {GT_PATH}")
    print(f"Pred Dir : {PRED_DIR}\n")
    print("-"*70)

    gt_data = parse_gt_file(GT_PATH)
    print(f"✓ GT: {len(gt_data)} images, {sum(len(v) for v in gt_data.values())} boxes")
    
    print("\nParsing predictions...")
    pred_data = parse_prediction_files(PRED_DIR)
    print(f"✓ Predictions: {len(pred_data)} images, {sum(len(v['boxes']) for v in pred_data.values())} boxes")

    gt_filtered = {k: v for k, v in gt_data.items() if k in pred_data}
    print(f"✓ Matched: {len(gt_filtered)} images")

    print("\nCalculating metrics...")
    ap, best_f1, best_prec, best_rec, best_thresh, rec, prec = calculate_metrics(gt_filtered, pred_data)

    if rec is not None:
        plot_pr_curve(rec, prec, ap, PLOT_PATH, os.path.basename(PRED_DIR))

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"AP @ IoU 0.5    : {ap:.4f}")
    print(f"Best F1-Score   : {best_f1:.4f}")
    print(f"  Precision     : {best_prec:.4f}")
    print(f"  Recall        : {best_rec:.4f}")
    print(f"  Threshold     : {best_thresh:.4f}")
    print("="*70)