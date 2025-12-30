from ultralytics import YOLO

def main():
    # Load the YOLOv11-pose model
    model = YOLO('models/yolo11s-pose-custom/yolo11s_pose_p2p3p4/weights/best.pt')

    # Evaluasi (non-multiprocessing di Windows)
    metrics = model.val(
        data='data/face_pose_yolo/face_pose.yaml',
        device='0',
        workers=0,        # <- penting di Windows
        batch=4          # opsional: atur sesuai VRAM (RTX 3050 4GB mungkin 8–16)
    )

    print("Evaluation metrics:")
    print(metrics)

if __name__ == "__main__":
    # Alternatif opsional: paksa spawn (tidak wajib jika workers=0)
    # import torch.multiprocessing as mp
    # mp.set_start_method("spawn", force=True)

    main()
