from importlib.metadata import metadata
import os
import cv2
import numpy as np
from tqdm import tqdm

from core.cluster import FaceCluster
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.visualize import save_clusters
from storage.store import Storage
from core.faiss_index import FaissIndex
from core.cluster import DBSCANCluster
from core.config import Config


class ImageProcessor:
    def __init__(self, image_folder, use_gpu=False):
        self.image_folder = image_folder
        self.detector = FaceDetector(use_gpu)
        self.embedder = FaceEmbedder()
        
        self.embeddings = []
        self.metadata = []

        self.storage = Storage()

    def process(self):
        # Load previous data
        old_embeddings = self.storage.load_embeddings()
        old_metadata = self.storage.load_metadata()

        processed_images = set()

        if old_metadata:
            processed_images = set([m["image"] for m in old_metadata])
            print(f"📂 Found {len(processed_images)} already processed images")
        image_files = [f for f in os.listdir(self.image_folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_name in tqdm(image_files):
            if img_name in processed_images:
                continue
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
                    "face_id": int(idx),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

        # --- PREPARE NEW EMBEDDINGS ---
        new_embeddings = np.array(self.embeddings)

        # --- MERGE LOGIC ---

        # CASE 1: No new images → reuse old data
        if new_embeddings.size == 0:
            print("⚡ No new images found. Using existing data.")

            embeddings_array = old_embeddings
            metadata = old_metadata

        # CASE 2: First run (no old data)
        elif old_embeddings is None:
            embeddings_array = new_embeddings
            metadata = self.metadata

        # CASE 3: Normal merge (old + new)
        else:
            embeddings_array = np.vstack([old_embeddings, new_embeddings])
            # Ensure old_metadata is a list before merging
            if old_metadata is None:
                old_metadata = []
            metadata = old_metadata + self.metadata

        # --- FAISS INDEX ---

        faiss_index = FaissIndex(dim=512)

        old_index = self.storage.load_faiss_index()

        # CASE 1: No new data → reuse index
        if new_embeddings.size == 0 and old_index is not None:
            print("⚡ Using existing FAISS index")
            faiss_index.index = old_index

        # CASE 2: Incremental update
        elif old_index is not None:
            print("⚡ Updating FAISS index with new embeddings")
            faiss_index.index = old_index
            faiss_index.add(new_embeddings)

        # CASE 3: First run
        else:
            print("⚡ Building FAISS index from scratch")
            faiss_index.build(embeddings_array)

        # Save index
        self.storage.save_faiss_index(faiss_index.index)    

        # --- CLUSTERING LOGIC ---

        config = Config()

        eps = config.get("clustering", "eps")
        min_samples = config.get("clustering", "min_samples")

        old_clusters = self.storage.load_clusters()

        # CASE 1: No new data → reuse
        if new_embeddings.size == 0 and old_clusters is not None:
            print("⚡ Using existing clusters (no recomputation)")
            cluster_dict = old_clusters

        # CASE 2: Incremental (keep your existing logic)
        elif old_clusters is not None and old_embeddings is not None:
            print("⚡ Incremental clustering (adding new faces)")

            from core.cluster import assign_to_clusters

            cluster_dict = assign_to_clusters(
                embeddings_array,
                new_embeddings,
                old_clusters,
                len(old_embeddings),
                threshold=0.5
            )

        # CASE 3: First run → DBSCAN
        else:
            print("⚡ First-time clustering (DBSCAN)")

            clusterer = DBSCANCluster(eps=eps, min_samples=min_samples)
            clusters = clusterer.cluster(embeddings_array)

            cluster_dict = {
                f"person_{i}": cluster
                for i, cluster in enumerate(clusters)
            }

        # --- OUTPUT ---
        print(f"Total clusters (people): {len(cluster_dict)}")

        # --- VISUALIZATION ---
        save_clusters(cluster_dict, metadata)

        # --- SAVE (VERY IMPORTANT ORDER) ---
        self.storage.save_embeddings(embeddings_array)
        self.storage.save_metadata(metadata)
        self.storage.save_clusters(cluster_dict)

        return embeddings_array, metadata, cluster_dict