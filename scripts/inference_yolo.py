from ultralytics import YOLO
import cv2
import torch

# Load the YOLOv11-pose model
model = YOLO('models/yolo11s-pose-default/yolo11s_pose/weights/best.pt')


# Hapus cache GPU dulu
torch.cuda.empty_cache()

# Reset counter
torch.cuda.reset_peak_memory_stats()

# Input image
img_path = 'data/dataset/widerface/wider_val/images/0--parade/0_Parade_Parade_0_628.jpg'
img = cv2.imread(img_path)

# Run inference
results = model(img, device=0, imgsz=960)

# Save the output image with results
output_path = 'data/output/result_test_1_yolov11pose.jpg'
results[0].save(filename=output_path)

# Catat penggunaan memori maksimum
mem_used = torch.cuda.max_memory_allocated() / 1e6  # MB
print(f"Peak GPU memory used: {mem_used:.2f} MB")

print(f"Inference complete. Results saved to {output_path}")
