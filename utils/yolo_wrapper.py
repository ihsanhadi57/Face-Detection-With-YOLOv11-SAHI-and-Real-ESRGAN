import numpy as np
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from ultralytics import YOLO
from typing import Optional, List

class YOLOv11PoseDetectionModel(DetectionModel):
    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.3,
        device: str = 'cpu',
        image_size: int = 1024,
        **kwargs,
    ):
        """
        Inisialisasi wrapper model YOLOv11-Pose untuk SAHI.
        
        Args:
            model_path: Path ke model weights
            confidence_threshold: Threshold confidence untuk deteksi
            device: Device untuk inference ('cpu' atau 'cuda:0')
            image_size: Ukuran image untuk inference (harus sama dengan training)
        """
        # Save parameters sebelum parent init
        self._model_path = model_path
        self._device = device
        self._image_size = image_size
        self._confidence_threshold = confidence_threshold
        
        # Dictionary untuk menyimpan keypoints
        self.keypoints_cache = {}
        
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
            **kwargs,
        )
        
        # Restore attributes setelah parent init
        self.model_path = self._model_path
        self.device = self._device
        self.image_size = self._image_size
        self.confidence_threshold = self._confidence_threshold

    def load_model(self):
        """Memuat model YOLOv11-pose"""
        if not self.model_path:
            raise ValueError("model_path harus ditentukan")
        
        print(f"Loading model from: {self.model_path}")
        print(f"Device: {self.device}, Image size: {self.image_size}")
        
        self.model = YOLO(self.model_path)
        self.category_mapping = {'0': 'face'}

    def unload_model(self):
        """Mengosongkan model dari memori"""
        self.model = None
        self.keypoints_cache = {}

    def perform_inference(self, image: np.ndarray):
        """
        Melakukan inferensi pada gambar.
        
        Args:
            image: Numpy array image (RGB format)
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Gunakan .predict() dengan explicit parameters
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            device=self.device,
            imgsz=self.image_size,
            verbose=False
        )
        
        self._original_predictions = results

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        Mengonversi hasil prediksi YOLO menjadi format SAHI ObjectPrediction.
        Keypoints disimpan ke cache untuk di-attach nanti.
        """
        original_predictions = self._original_predictions
        
        if not original_predictions or len(original_predictions[0].boxes) == 0:
            self._object_prediction_list_per_image = [[]]
            return

        # Parse shift_amount dan full_shape
        if shift_amount_list is None:
            shift_amount = [0, 0]
        elif isinstance(shift_amount_list, list):
            if len(shift_amount_list) > 0 and isinstance(shift_amount_list[0], list):
                shift_amount = shift_amount_list[0]
            else:
                shift_amount = shift_amount_list if len(shift_amount_list) == 2 else [0, 0]
        else:
            shift_amount = [0, 0]
        
        if full_shape_list is None:
            full_shape = None
        elif isinstance(full_shape_list, list):
            if len(full_shape_list) > 0 and isinstance(full_shape_list[0], list):
                full_shape = full_shape_list[0]
            else:
                full_shape = full_shape_list
        else:
            full_shape = None

        object_prediction_list = []
        result = original_predictions[0]
        boxes = result.boxes
        
        # Extract keypoints jika ada
        has_keypoints = hasattr(result, 'keypoints') and result.keypoints is not None
        keypoints_data = None
        if has_keypoints:
            keypoints_data = result.keypoints.data.cpu().numpy()
        
        # Convert setiap detection ke ObjectPrediction
        for i in range(len(boxes)):
            score = float(boxes.conf[i])
            
            # PENTING: Jangan filter confidence di sini!
            # Model sudah filter dengan conf parameter di inference
            
            bbox = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = bbox.astype(int)

            # Apply shift untuk SAHI slicing
            shifted_x1 = int(x1) + shift_amount[0]
            shifted_y1 = int(y1) + shift_amount[1]
            shifted_x2 = int(x2) + shift_amount[0]
            shifted_y2 = int(y2) + shift_amount[1]

            object_prediction = ObjectPrediction(
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                category_id=0,
                category_name='face',
                score=score,
                shift_amount=shift_amount,
                full_shape=full_shape
            )
            
            # Simpan keypoints ke cache
            if has_keypoints and i < len(keypoints_data):
                kpts = keypoints_data[i].copy()
                kpts[:, 0] += shift_amount[0]
                kpts[:, 1] += shift_amount[1]
                
                cache_key = f"{shifted_x1}_{shifted_y1}_{shifted_x2}_{shifted_y2}"
                self.keypoints_cache[cache_key] = kpts
            
            object_prediction_list.append(object_prediction)
        
        self._object_prediction_list_per_image = [object_prediction_list]
    
    def attach_keypoints_to_predictions(self, object_prediction_list):
        """
        Attach keypoints dari cache ke ObjectPrediction list.
        Dipanggil setelah SAHI selesai merging predictions.
        """
        attached_count = 0
        
        for pred in object_prediction_list:
            bbox = pred.bbox.to_voc_bbox()
            cache_key = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
            
            # Try exact match
            if cache_key in self.keypoints_cache:
                pred.keypoints = self.keypoints_cache[cache_key]
                attached_count += 1
            else:
                # Fuzzy match dengan IoU
                best_iou = 0.0
                best_kpts = None
                
                for key, kpts in self.keypoints_cache.items():
                    coords = [int(float(x)) for x in key.split('_')]
                    iou = self._calculate_iou(bbox, coords)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_kpts = kpts
                
                if best_iou > 0.5 and best_kpts is not None:
                    pred.keypoints = best_kpts
                    attached_count += 1
        
        return object_prediction_list
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    @property
    def num_categories(self):
        return len(self.category_names)

    @property
    def has_mask(self):
        return False

    @property
    def category_names(self):
        return ["face"]