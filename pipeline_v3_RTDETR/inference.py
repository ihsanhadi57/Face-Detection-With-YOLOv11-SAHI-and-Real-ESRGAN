"""
RT-DETR Inference untuk Face Detection
"""

import os
import cv2
import torch
from ultralytics import RTDETR
from pathlib import Path
import numpy as np

class FaceDetector:
    def __init__(self, weights_path='output/rtdetr_widerface/weights/best.pt', conf=0.25):
        """
        Initialize face detector
        Args:
            weights_path: Path to trained weights
            conf: Confidence threshold
        """
        self.model = RTDETR(weights_path)
        self.conf = conf
        print(f"Model loaded: {weights_path}")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    def detect_image(self, image_path, save_path=None, show=False):
        """
        Detect faces in single image
        Args:
            image_path: Path to input image
            save_path: Path to save result (optional)
            show: Show result window
        Returns:
            results: Detection results
        """
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf,
            save=save_path is not None,
            show=show
        )
        
        # Get image and draw boxes
        img = cv2.imread(image_path)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Draw box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw confidence
                label = f'Face: {conf:.2f}'
                cv2.putText(img, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save result
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img)
            print(f"Result saved to: {save_path}")
        
        # Show result
        if show:
            cv2.imshow('Face Detection', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results
    
    def detect_folder(self, folder_path, output_folder='output/predictions'):
        """
        Detect faces in all images in folder
        Args:
            folder_path: Path to input folder
            output_folder: Path to save results
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f'*{ext}'))
            image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images")
        
        # Process each image
        for img_path in image_files:
            save_path = os.path.join(output_folder, img_path.name)
            print(f"Processing: {img_path.name}")
            self.detect_image(str(img_path), save_path=save_path)
        
        print(f"\nAll results saved to: {output_folder}")
    
    def detect_video(self, video_path, output_path='output/video_result.mp4'):
        """
        Detect faces in video
        Args:
            video_path: Path to input video
            output_path: Path to save result video
        """
        # Run inference on video
        results = self.model.predict(
            source=video_path,
            conf=self.conf,
            save=True,
            stream=True
        )
        
        print(f"Processing video: {video_path}")
        for r in results:
            pass  # Process frames
        
        print(f"Video result saved")
    
    def detect_webcam(self, camera_id=0):
        """
        Real-time face detection from webcam
        Args:
            camera_id: Camera device ID (default 0)
        """
        cap = cv2.VideoCapture(camera_id)
        
        print("Starting webcam... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = self.model.predict(
                source=frame,
                conf=self.conf,
                verbose=False
            )
            
            # Draw results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f'Face: {conf:.2f}'
                    cv2.putText(frame, label, (int(x1), int(y1)-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Face Detection - Press q to quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function for testing"""
    
    # Initialize detector
    detector = FaceDetector(
        weights_path='output/rtdetr_widerface/weights/best.pt',
        conf=0.3
    )
    
    print("\n" + "="*50)
    print("RT-DETR Face Detection - Inference")
    print("="*50)
    print("\nOptions:")
    print("1. Detect single image")
    print("2. Detect folder of images")
    print("3. Detect video")
    print("4. Real-time webcam detection")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == '1':
        img_path = input("Enter image path: ")
        detector.detect_image(img_path, save_path='output/result.jpg', show=True)
    
    elif choice == '2':
        folder_path = input("Enter folder path: ")
        detector.detect_folder(folder_path)
    
    elif choice == '3':
        video_path = input("Enter video path: ")
        detector.detect_video(video_path)
    
    elif choice == '4':
        detector.detect_webcam()
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()