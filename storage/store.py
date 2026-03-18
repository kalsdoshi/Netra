import json
import numpy as np
import os
import faiss

class Storage:
    def __init__(self, base_path="storage_data"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save_embeddings(self, embeddings):
        path = os.path.join(self.base_path, "embeddings.npy")
        np.save(path, embeddings)

    def load_embeddings(self):
        path = os.path.join(self.base_path, "embeddings.npy")
        if os.path.exists(path):
            return np.load(path)
        return None

    def save_metadata(self, metadata):
        path = os.path.join(self.base_path, "metadata.json")

        # remove face_crop (not serializable)
        clean_metadata = [
            {k: v for k, v in m.items() if k != "face_crop"}
            for m in metadata
        ]

        with open(path, "w") as f:
            json.dump(clean_metadata, f, indent=2)

    def load_metadata(self):
        path = os.path.join(self.base_path, "metadata.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    def save_clusters(self, cluster_dict):
        path = os.path.join(self.base_path, "clusters.json")
        with open(path, "w") as f:
            json.dump(cluster_dict, f, indent=2)

    def load_clusters(self):
        path = os.path.join(self.base_path, "clusters.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None
    def save_faiss_index(self, index):
        path = os.path.join(self.base_path, "faiss.index")
        faiss.write_index(index, path)

    def load_faiss_index(self):
        path = os.path.join(self.base_path, "faiss.index")
        if os.path.exists(path):
            return faiss.read_index(path)
        return None