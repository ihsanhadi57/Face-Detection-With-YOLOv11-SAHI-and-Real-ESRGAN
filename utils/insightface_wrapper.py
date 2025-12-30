import numpy as np
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from insightface.app import FaceAnalysis
from typing import Optional, List

class InsightFaceDetectionModel(DetectionModel):
    def __init__(
        self,
        confidence_threshold: float = 0.3,
        providers: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Inisialisasi wrapper model. Memanggil parent constructor.
        """
        self.providers = providers
        kwargs.pop('device', None)
        super().__init__(
            confidence_threshold=confidence_threshold,
            device=None,
            **kwargs,
        )

    def load_model(self):
        """
        Memuat model InsightFace dan menyiapkannya dengan konfigurasi dari `self`.
        """
        if self.providers:
            providers = self.providers
            print(f"Info: Menggunakan providers yang ditentukan: {providers}")
        else:
            providers = ['CPUExecutionProvider']
            print("Info: Tidak ada provider ditentukan, menggunakan CPU.")

        ctx_id = 0 if providers and 'CUDAExecutionProvider' in providers else -1

        self.model = FaceAnalysis(providers=providers)
        
        self.model.prepare(
            ctx_id=ctx_id,
            det_size=(640, 640),
            det_thresh=self.confidence_threshold
        )
        
        self.category_mapping = {'0': 'face'}

    def unload_model(self):
        """Mengosongkan model dari memori CPU/GPU."""
        self.model = None

    def perform_inference(self, image: np.ndarray):
        """
        Melakukan inferensi pada gambar. Hasilnya disimpan di self._original_predictions.
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        faces = self.model.get(image)
        self._original_predictions = faces

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        Mengonversi hasil prediksi mentah (dari self._original_predictions) 
        menjadi daftar objek `ObjectPrediction`.
        """
        original_predictions = self._original_predictions
        
        if not original_predictions:
            self._object_prediction_list_per_image = [[]]
            return

        shift_amount = shift_amount_list
        full_shape = full_shape_list

        object_prediction_list = []
        for face in original_predictions:
            score = float(face.det_score)
            if score < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = face.bbox.astype(int)

            object_prediction = ObjectPrediction(
                bbox=[x1, y1, x2, y2],
                category_id=0,
                category_name='face',
                score=score,
                shift_amount=shift_amount,
                full_shape=full_shape
            )
            object_prediction_list.append(object_prediction)
        
        self._object_prediction_list_per_image = [object_prediction_list]

    @property
    def num_categories(self):
        """Mengembalikan jumlah kategori yang didukung."""
        return len(self.category_names)

    @property
    def has_mask(self):
        """Menunjukkan apakah model mendukung segmentasi (mask)."""
        return False

    @property
    def category_names(self):
        """Mengembalikan nama-nama kategori."""
        return ["face"]
