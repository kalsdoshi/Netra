import cv2
from insightface.app import FaceAnalysis
from core.gpu_utils import GPUDetector


class FaceDetector:
    def __init__(self, use_gpu=None):
        """
        Initialize FaceDetector with optional GPU acceleration.
        
        Args:
            use_gpu: If None, auto-detect GPU availability. If True/False, force setting.
        """
        # Auto-detect GPU if not explicitly set
        if use_gpu is None:
            gpu_detector = GPUDetector()
            use_gpu = gpu_detector.is_available()
        
        ctx_id = 0 if use_gpu else -1
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=ctx_id)
        self.use_gpu = use_gpu

    def detect(self, image):
        faces = self.app.get(image)
        results = []

        for face in faces:
            bbox = face.bbox.astype(int)
            landmarks = face.kps  # 5 points
            results.append({
                "bbox": bbox,
                "landmarks": landmarks,
                "face": face
            })

        return results