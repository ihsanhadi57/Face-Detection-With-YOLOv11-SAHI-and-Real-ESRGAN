"""
Script untuk membandingkan Standard YOLO vs SAHI dengan konfigurasi yang sama
"""
import subprocess
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def run_evaluation(mode, conf_threshold=0.25):
    """
    Run evaluasi dengan mode tertentu
    
    Args:
        mode: 'old-standard', 'new-standard', 'sahi-uniform', 'sahi-adaptive'
        conf_threshold: confidence threshold (default 0.25 untuk apple-to-apple)
    """
    if mode == 'old-standard':
        print("\n" + "="*80)
        print("🔧 Running OLD Standard Evaluation (eval_classified_widerface.py)")
        print("="*80)
        cmd = ["python", "eval/eval_classified_widerface.py"]
        
    elif mode == 'new-standard':
        print("\n" + "="*80)
        print(f"🔧 Running NEW Standard Evaluation (conf={conf_threshold})")
        print("="*80)
        # Modify code baru untuk gunakan conf yang sama
        cmd = ["python", "eval/eval_classified_widerface_sahi.py", 
               "--mode", "standard", 
               f"--conf", str(conf_threshold)]
        
    elif mode.startswith('sahi'):
        print("\n" + "="*80)
        print(f"🔧 Running SAHI Evaluation - {mode.upper()} (conf={conf_threshold})")
        print("="*80)
        cmd = ["python", "eval/eval_classified_widerface_sahi.py", 
               "--mode", mode,
               f"--conf", str(conf_threshold)]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("❌ Error:", result.stderr)
        return None
    
    return result.returncode == 0

def load_results(filename):
    """Load hasil evaluasi dari JSON"""
    filepath = Path("output") / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def compare_results():
    """Bandingkan hasil dari semua mode"""
    print("\n" + "="*80)
    print("📊 COMPARING ALL RESULTS")
    print("="*80)
    
    # Load all results
    results_map = {
        'Old Standard (default conf)': load_results('evaluation_results.json'),
        'New Standard (conf=0.25)': load_results('standard_025_results.json'),
        'New Standard (conf=0.5)': load_results('standard_results.json'),
        'SAHI Uniform (conf=0.25)': load_results('sahi_uniform_025_results.json'),
        'SAHI Uniform (conf=0.5)': load_results('sahi_uniform_results.json'),
    }
    
    # Filter valid results
    valid_results = {k: v for k, v in results_map.items() if v is not None}
    
    if not valid_results:
        print("❌ No results found!")
        return
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    categories = ['small_clear', 'small_degraded', 'medium_large']
    category_names = ['Small Clear', 'Small Degraded', 'Medium Large']
    
    for cat_idx, (category, cat_name) in enumerate(zip(categories, category_names)):
        ax = axes[cat_idx]
        
        modes = []
        tp_counts = []
        fn_counts = []
        recalls = []
        
        for mode_name, results in valid_results.items():
            for result in results:
                if result['category'] == category:
                    modes.append(mode_name.replace(' ', '\n'))
                    tp_counts.append(result['true_positives'])
                    fn_counts.append(result['false_negatives'])
                    recalls.append(result['recall'] * 100)
        
        x = np.arange(len(modes))
        width = 0.35
        
        # Plot TP vs FN
        bars1 = ax.bar(x - width/2, tp_counts, width, label='Detected (TP)', 
                       color='#27ae60', edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, fn_counts, width, label='Missed (FN)', 
                       color='#c0392b', edgecolor='black', linewidth=1)
        
        # Add labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=8)
        
        ax.set_xlabel('Mode', fontsize=10, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax.set_title(f'{cat_name}\nRecall: {" | ".join([f"{r:.1f}%" for r in recalls])}', 
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(modes, fontsize=7, rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Comparison: Standard YOLO vs SAHI (Different Configurations)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = Path("output") / "comparison_all_modes.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison saved to: {output_file}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE - SMALL DEGRADED (Most Critical)")
    print("="*80)
    print(f"{'Mode':<35} {'GT':<8} {'TP':<8} {'FN':<8} {'Recall':<10}")
    print("-" * 80)
    
    for mode_name, results in valid_results.items():
        for result in results:
            if result['category'] == 'small_degraded':
                print(f"{mode_name:<35} {result['total_gt']:<8} "
                      f"{result['true_positives']:<8} {result['false_negatives']:<8} "
                      f"{result['recall']*100:<10.2f}%")
    print("="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Standard vs SAHI evaluation')
    parser.add_argument('--run-all', action='store_true', 
                       help='Run all evaluations (old-standard, new-standard, sahi)')
    parser.add_argument('--compare-only', action='store_true',
                       help='Only generate comparison from existing results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for new evaluations (default: 0.25)')
    
    args = parser.parse_args()
    
    if args.run_all:
        # Run semua evaluasi
        print("🚀 Running ALL evaluations for comparison...")
        
        # 1. Old standard (baseline)
        run_evaluation('old-standard')
        
        # 2. New standard dengan conf yang sama
        run_evaluation('new-standard', conf_threshold=args.conf)
        
        # 3. New standard dengan conf=0.5
        if args.conf != 0.5:
            run_evaluation('new-standard', conf_threshold=0.5)
        
        # 4. SAHI uniform dengan conf yang sama
        run_evaluation('sahi-uniform', conf_threshold=args.conf)
        
        # 5. SAHI uniform dengan conf=0.5
        if args.conf != 0.5:
            run_evaluation('sahi-uniform', conf_threshold=0.5)
        
        # Generate comparison
        compare_results()
        
    elif args.compare_only:
        # Hanya generate comparison
        compare_results()
        
    else:
        print("Usage:")
        print("  python scripts/compare_standard_sahi.py --run-all          # Run all and compare")
        print("  python scripts/compare_standard_sahi.py --compare-only     # Only compare existing")
        print("  python scripts/compare_standard_sahi.py --run-all --conf 0.3  # Custom confidence")