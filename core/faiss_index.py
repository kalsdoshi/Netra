import faiss 
import numpy as np


class FaissIndex:
    def __init__(self, dim=512):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity

    def build(self, embeddings):
        embeddings = embeddings.astype("float32")
        self.index.add(embeddings)

    def search(self, query, k=5):
        query = query.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query, k)
        return scores[0], indices[0]

    def add(self, embeddings):
        embeddings = embeddings.astype("float32")
        self.index.add(embeddings)