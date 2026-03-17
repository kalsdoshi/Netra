import numpy as np

class FaceEmbedder:
    def __init__(self):
        pass  # embedding already comes from InsightFace face object

    def get_embedding(self, face):
        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding