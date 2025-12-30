from pathlib import Path
from ultralytics.utils.plotting import plot_results

run_dir = Path(r"D:/collage/Skripsi/Code/another/models/yolo11m-pose-custom/yolo11m_pose_p2p3p4")
csv_path = run_dir / "results.csv"

plot_results(file=str(csv_path))
# atau kalau kamu ingin eksplisit:
# plot_results(file=str(csv_path), dir=str(run_dir))
 