import os
import cv2
import numpy as np
from tqdm import tqdm

from core.cluster import FaceCluster
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.visualize import save_clusters


class ImageProcessor:
    def __init__(self, image_folder, use_gpu=False):
        self.image_folder = image_folder
        self.detector = FaceDetector(use_gpu)
        self.embedder = FaceEmbedder()
        
        self.embeddings = []
        self.metadata = []

    def process(self):
        image_files = [f for f in os.listdir(self.image_folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_name in tqdm(image_files):
            img_path = os.path.join(self.image_folder, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            faces = self.detector.detect(image)

            for idx, f in enumerate(faces):
                embedding = self.embedder.get_embedding(f["face"])

                self.embeddings.append(embedding)
                bbox = f["bbox"]
                h, w, _ = image.shape

                x1, y1, x2, y2 = bbox

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                face_crop = image[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                self.metadata.append({
                    "image": img_name,
                    "face_id": idx,
                    "face_crop": face_crop
                })

        embeddings_array = np.array(self.embeddings)

        from core.cluster import FaceCluster
        clusterer = FaceCluster(threshold=0.5)
        clusters = clusterer.cluster(embeddings_array)
        
    
        print(f"Total clusters (people): {len(clusters)}")
        

        cluster_dict = {
            f"person_{i}": cluster
            for i, cluster in enumerate(clusters)
        }
        
        save_clusters(cluster_dict, self.metadata)
        
        return embeddings_array, self.metadata, cluster_dict