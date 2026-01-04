import cv2
import numpy as np
from typing import List, Optional, Any
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from insightface.app import FaceAnalysis
import logging

def _safe_int(val, default: int) -> int:
    try:
        if isinstance(val, (int, np.integer)) and val > 0:
            return int(val)
        if isinstance(val, str) and val.isdigit():
            return int(val)
    except Exception:
        pass
    return int(default)

class RetinaFaceSAHI(DetectionModel):
    """
    RetinaFace (InsightFace) wrapper untuk SAHI sliced inference
    FINAL FIXED: Proper handling of original_predictions property
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        category_mapping: Optional[dict] = None,
        load_at_init: bool = True,
        image_size: Optional[int] = 640,
        ctx_id: Optional[int] = None,
    ):
        # Initialize internal storage for raw predictions
        self._raw_predictions = []
        
        # Tetapkan ctx_id SEBELUM super().__init__
        self.ctx_id = -1 if str(device).startswith("cpu") else 0 if ctx_id is None else ctx_id

        # Set image_size SEBELUM super().__init__
        self.image_size = _safe_int(image_size, 640)

        self.logger = logging.getLogger(__name__)

        # Ini akan memanggil self.load_model() jika load_at_init=True
        super().__init__(
            model_path=model_path,
            confidence_threshold=float(confidence_threshold),
            device=device,
            category_mapping=category_mapping,
            load_at_init=load_at_init,
        )

    def _resolved_det_size(self) -> tuple:
        side = _safe_int(getattr(self, "image_size", None), 640)
        return (side, side)

    def load_model(self):
        """Load model InsightFace (RetinaFace)"""
        try:
            print(f"Loading RetinaFace model with ctx_id={self.ctx_id}...")
            providers = ["CPUExecutionProvider"]
            if self.ctx_id >= 0:
                try:
                    import onnxruntime as ort
                    if "CUDAExecutionProvider" in ort.get_available_providers():
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                        print("Using CUDA provider")
                    else:
                        print("CUDA provider not available, using CPU")
                        self.ctx_id = -1
                except Exception as e:
                    print(f"Falling back to CPU execution: {e}")
                    self.ctx_id = -1

            self.model = FaceAnalysis(providers=providers)

            det_size = self._resolved_det_size()
            self.model.prepare(
                ctx_id=self.ctx_id,
                det_size=det_size,
                det_thresh=float(getattr(self, "confidence_threshold", 0.5)),
            )
            print(f"RetinaFace model loaded successfully! det_size={det_size}")
        except Exception as e:
            print(f"Error loading RetinaFace model: {e}")
            raise

    def unload_model(self):
        if hasattr(self, "model"):
            del self.model
        print("Model unloaded")

    def perform_inference(self, image: np.ndarray) -> List[ObjectPrediction]:
        """
        Perform inference and store raw predictions untuk SAHI framework
        """
        try:
            if image is None or image.size == 0:
                print("Invalid input image")
                return []

            # Pastikan uint8
            if image.dtype != np.uint8:
                image = np.clip(image * 255, 0, 255).astype(np.uint8)

            # Pastikan 3 channel
            if len(image.shape) != 3 or image.shape[2] != 3:
                print(f"Invalid image shape: {image.shape}")
                return []

            print(f"Processing image shape: {image.shape}")

            # Guard det_size
            ds = getattr(self.model, "det_size", None)
            if (ds is None) or (None in ds):
                fix = self._resolved_det_size()
                print(f"det_size invalid: {ds}, fixing to {fix}")
                self.model.prepare(
                    ctx_id=self.ctx_id,
                    det_size=fix,
                    det_thresh=float(getattr(self, "confidence_threshold", 0.5)),
                )

            faces = self.model.get(image)
            print(f"Raw detection: {len(faces)} faces found")
            
            # CRITICAL FIX: Store dalam internal variable, bukan property
            self._raw_predictions = faces
            
            if len(faces) == 0:
                return []

            h, w = image.shape[:2]
            object_predictions: List[ObjectPrediction] = []

            for i, face in enumerate(faces):
                try:
                    bbox = getattr(face, "bbox", None)
                    if bbox is None:
                        continue

                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                    except Exception as e:
                        print(f"   Face {i+1}: bbox conversion error: {e}")
                        continue
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Clamp to boundaries
                    x1 = max(0, min(w, x1)); y1 = max(0, min(h, y1))
                    x2 = max(0, min(w, x2)); y2 = max(0, min(h, y2))

                    raw_score = getattr(face, "det_score", None)
                    score = float(raw_score) if raw_score is not None else 0.0
                    if score < float(getattr(self, "confidence_threshold", 0.5)):
                        continue

                    op = ObjectPrediction(
                        bbox=[x1, y1, x2, y2],
                        score=score,
                        category_id=0,
                        category_name="face",
                    )
                    object_predictions.append(op)

                except Exception as e:
                    print(f"   Face {i+1}: Processing error: {e}")
                    continue

            print(f"Successfully converted {len(object_predictions)}/{len(faces)} faces to ObjectPredictions")
            return object_predictions

        except Exception as e:
            print(f"perform_inference error: {e}")
            import traceback
            traceback.print_exc()
            return []

    @property
    def original_predictions(self):
        """SAHI framework mengakses ini untuk mendapatkan raw predictions"""
        return getattr(self, '_raw_predictions', [])

    def _create_object_prediction_list_from_original_predictions(
    self,
    shift_amount_list: Optional[List[List[int]]] = None,
    full_shape_list: Optional[List[List[int]]] = None,
    ) -> List[ObjectPrediction]:
        """
        FIXED: Convert raw predictions ke ObjectPrediction dengan coordinate shifting
        """
        try:
            faces = self._raw_predictions
            if not faces:
                return []
            
            # FIX 1: Gunakan list 'shift_amount_list' secara langsung
            if shift_amount_list is None or not isinstance(shift_amount_list, list) or len(shift_amount_list) < 2:
                shift_amount = [0, 0]
            else:
                shift_amount = shift_amount_list

            # FIX 2: Gunakan list 'full_shape_list' secara langsung
            if full_shape_list is None or not isinstance(full_shape_list, list) or len(full_shape_list) < 2:
                full_shape = [1024, 1024]  # Default aman
            else:
                full_shape = full_shape_list
            
            # Konvensi SAHI: [height, width]
            h, w = full_shape[0], full_shape[1]
            shift_x, shift_y = shift_amount
            
            object_predictions: List[ObjectPrediction] = []
            
            for i, face in enumerate(faces):
                try:
                    bbox = getattr(face, "bbox", None)
                    if bbox is None:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    
                    # Terapkan pergeseran koordinat
                    x1 += shift_x
                    y1 += shift_y
                    x2 += shift_x
                    y2 += shift_y
                    
                    # Batasi koordinat agar tidak keluar dari gambar penuh
                    x1 = max(0, min(w, x1))
                    y1 = max(0, min(h, y1))
                    x2 = max(0, min(w, x2))
                    y2 = max(0, min(h, y2))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue

                    raw_score = getattr(face, "det_score", 0.0)
                    score = float(raw_score)
                    
                    if score < self.confidence_threshold:
                        continue

                    op = ObjectPrediction(
                        bbox=[x1, y1, x2, y2],
                        score=score,
                        category_id=0,
                        category_name="face",
                        shift_amount=shift_amount,
                        full_shape=full_shape
                    )
                    object_predictions.append(op)

                except Exception as e:
                    print(f"Error converting face {i+1}: {e}")
                    continue

            return object_predictions
            
        except Exception as e:
            print(f"_create_object_prediction_list_from_original_predictions error: {e}")
            import traceback
            traceback.print_exc()
            return []

    @property
    def category_names(self) -> List[str]:
        return ["face"]

    @property
    def has_mask(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return "RetinaFace"
