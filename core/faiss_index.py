import faiss 
import numpy as np


class FaissIndex:
    def __init__(self, dim=512):
        self.dim = dim
        self.index = None  # Built dynamically based on size

    def build(self, embeddings):
        embeddings = embeddings.astype("float32")
        n = len(embeddings)
        
        # Auto-select index type based on dataset size
        if n < 1000:
            print("⚡ Using IndexFlatIP (Brute Force)")
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embeddings)
        elif n < 50000:
            print(f"⚡ Using IndexIVFFlat (n={n})")
            nlist = int(np.sqrt(n))  # Rule of thumb
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(nlist, max(1, nlist // 4))  # Tune recall/speed
        else:
            print(f"⚡ Using IndexHNSWFlat (n={n})")
            M = 32  # Number of connections per node
            self.index = faiss.IndexHNSWFlat(self.dim, M, faiss.METRIC_INNER_PRODUCT)
            self.index.add(embeddings)

    def search(self, query, k=5):
        if not self.index:
            return [], []
        query = query.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query, k)
        return scores[0], indices[0]

    def add(self, embeddings):
        if not self.index:
            self.build(embeddings)
            return

        embeddings = embeddings.astype("float32")
        # IVF needs training, so if we're adding a massive amount, it's better to rebuild. 
        # But for streaming increments, we just add.
        self.index.add(embeddings)