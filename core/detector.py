import cv2
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, use_gpu=False):
        ctx_id = 0 if use_gpu else -1
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=ctx_id)

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