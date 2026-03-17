import faiss
import numpy as np

class SimilaritySearch:
    def __init__(self, dim=512):
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.embeddings = None

    def build_index(self, embeddings):
        self.embeddings = embeddings.astype('float32')
        self.index.add(self.embeddings)

    def search(self, query_embedding, k=5):
        query = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query, k)
        return scores[0], indices[0]