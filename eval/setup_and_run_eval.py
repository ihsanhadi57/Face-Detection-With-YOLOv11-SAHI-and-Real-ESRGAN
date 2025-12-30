import os
import shutil
import sys

def setup_evaluation_files():
    """
    Setup file-file yang diperlukan untuk evaluasi
    """
    print("Setting up evaluation files...")
    
    # Buat direktori yang diperlukan
    os.makedirs('ground_truth', exist_ok=True)
    os.makedirs('pred', exist_ok=True)
    
    # Copy ground truth files
    gt_files = [
        ('../data/dataset/widerface/wider_face_split/wider_face_val.mat', 'ground_truth/wider_face_val.mat'),
        ('../data/dataset/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 'ground_truth/wider_face_val_bbx_gt.txt')
    ]
    
    print("Copying ground truth files...")
    for src, dst in gt_files:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✓ Copied: {src} -> {dst}")
        else:
            print(f"✗ Not found: {src}")
    
    # Copy prediction files
    pred_src = '../data/eval_results/retinaface'
    if os.path.exists(pred_src):
        pred_files = [f for f in os.listdir(pred_src) if f.endswith('.txt')]
        print(f"\nCopying {len(pred_files)} prediction files...")
        
        for i, filename in enumerate(pred_files):
            src_path = os.path.join(pred_src, filename)
            dst_path = os.path.join('pred', filename)
            shutil.copy2(src_path, dst_path)
            
            if (i + 1) % 100 == 0:
                print(f"  Copied {i + 1}/{len(pred_files)} files...")
        
        print(f"✓ All prediction files copied to pred/")
    else:
        print(f"✗ Prediction directory not found: {pred_src}")
        return False
    
    return True

def run_evaluation():
    """
    Jalankan evaluasi WiderFace
    """
    try:
        print("\n" + "=" * 60)
        print("RUNNING WIDERFACE EVALUATION")
        print("=" * 60)
        
        # Import evaluation module
        sys.path.insert(0, '.')
        from evaluation import evaluation
        
        print("Starting evaluation...")
        print("This will take several minutes...")
        
        # Run evaluation
        evaluation('./pred', './ground_truth')
        
        print("\n✅ Evaluation completed successfully!")
        
        # Check output files
        output_files = [
            'wider_pr_info_Val.txt',
            'Val_PR_curve.png'
        ]
        
        print("\nOutput files:")
        for file in output_files:
            if os.path.exists(file):
                print(f"✓ {file}")
            else:
                print(f"✗ {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def parse_results():
    """
    Parse dan tampilkan hasil evaluasi
    """
    result_file = 'wider_pr_info_Val.txt'
    if not os.path.exists(result_file):
        print("Result file not found")
        return
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    with open(result_file, 'r') as f:
        content = f.read()
        print(content)

def main():
    """
    Main function - jalankan evaluasi lengkap
    """
    print("WiderFace Evaluation Setup and Run")
    print("=" * 40)
    
    # Step 1: Setup files
    if not setup_evaluation_files():
        print("❌ Setup failed!")
        return
    
    # Step 2: Run evaluation
    if not run_evaluation():
        print("❌ Evaluation failed!")
        return
    
    # Step 3: Parse results
    parse_results()
    
    print("\n" + "=" * 60)
    print("🎉 EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Check the following files for detailed results:")
    print("- wider_pr_info_Val.txt: mAP values")
    print("- Val_PR_curve.png: Precision-Recall curves")

if __name__ == "__main__":
    main()