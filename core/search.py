import cv2
import numpy as np


class FaceSearch:
    def __init__(self, detector, embedder, faiss_index):
        self.detector = detector
        self.embedder = embedder
        self.index = faiss_index

    def search(self, image_path, top_k=5):
        image = cv2.imread(image_path)

        if image is None:
            print("❌ Could not read image")
            return []

        faces = self.detector.detect(image)

        if len(faces) == 0:
            print("❌ No face detected")
            return []

        # take first face
        face = faces[0]

        embedding = self.embedder.get_embedding(face["face"])

        # FAISS search
        query = embedding.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query, top_k)

        return scores[0], indices[0]