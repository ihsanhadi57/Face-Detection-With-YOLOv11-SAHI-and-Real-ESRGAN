"""
Visualization utilities untuk face detection results
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple
import os

class FaceVisualizer:
    """Class untuk visualisasi hasil deteksi wajah"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
    
    def draw_detections(
    self, 
    image: np.ndarray, 
    detections: List, 
    save_path: str = None,
    show_confidence: bool = True,
    box_color: str = 'red',
    text_color: str = 'white'
    ) -> np.ndarray:

        fig, ax = plt.subplots(1, figsize=self.figsize)
        ax.imshow(image)
        ax.axis('off')

        detection_list = detections.object_prediction_list if hasattr(detections, 'object_prediction_list') else detections
        num_faces = len(detection_list)

        for detection in detection_list:
            # bbox
            if hasattr(detection, 'bbox'):
                bbox = detection.bbox.to_xyxy() if hasattr(detection.bbox, "to_xyxy") else detection.bbox
                # CAST score -> float
                confidence = float(getattr(detection.score, "value", detection.score or 0.0))
            else:
                bbox = detection[:4]
                confidence = float(detection[4]) if len(detection) > 4 else 1.0

            x1, y1, x2, y2 = bbox
            width, height = x2 - x1, y2 - y1

            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=box_color, facecolor='none')
            ax.add_patch(rect)

            if show_confidence:
                ax.text(
                    x1, y1-10, f'{confidence:.2f}',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color, alpha=0.7),
                    fontsize=10, color=text_color, weight='bold'
                )

        plt.title(f'Face Detection Results - {num_faces} faces detected', fontsize=14, weight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Visualization saved to: {save_path}")

        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array

    
    def save_face_crops(
    self, 
    image: np.ndarray, 
    detections: List, 
    output_dir: str,
    prefix: str = "face_crop"
    ) -> List[str]:

        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        detection_list = detections.object_prediction_list if hasattr(detections, 'object_prediction_list') else detections

        for i, detection in enumerate(detection_list):
            if hasattr(detection, 'bbox'):
                bbox = detection.bbox.to_xyxy() if hasattr(detection.bbox, "to_xyxy") else detection.bbox
                confidence = float(getattr(detection.score, "value", detection.score or 0.0))
            else:
                bbox = detection[:4]
                confidence = float(detection[4]) if len(detection) > 4 else 1.0

            x1, y1, x2, y2 = [int(c) for c in bbox]
            face_crop = image[y1:y2, x1:x2]

            if face_crop.size > 0:
                face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                filename = f"{prefix}_{i+1}_conf_{confidence:.2f}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, face_crop_bgr)
                saved_paths.append(filepath)

        print(f"Saved {len(saved_paths)} face crops to {output_dir}")
        return saved_paths

    
    def create_detection_summary(
        self, 
        results: dict, 
        save_path: str = None
    ) -> str:
        """
        Create text summary of detection results
        
        Args:
            results: Dictionary dengan detection statistics
            save_path: Path untuk save summary
            
        Returns:
            Summary text
        """
        summary = f"""
=== Face Detection Summary ===
Image: {results.get('image_path', 'Unknown')}
Total Faces Detected: {results.get('num_faces', 0)}
Processing Time: {results.get('processing_time', 0):.2f} seconds
Average Confidence: {results.get('avg_confidence', 0):.2f}
Min Confidence: {results.get('min_confidence', 0):.2f}
Max Confidence: {results.get('max_confidence', 0):.2f}

Detection Details:
"""
        
        for i, det in enumerate(results.get('detections', [])):
            bbox = det.get('bbox', [0, 0, 0, 0])
            conf = det.get('confidence', 0)
            summary += f"Face {i+1}: BBox({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}), Conf: {conf:.3f}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(summary)
            print(f"Summary saved to: {save_path}")
        
        return summary