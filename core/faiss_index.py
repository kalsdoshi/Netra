import faiss 
import numpy as np
import time


class FaissIndex:
    def __init__(self, dim=512):
        self.dim = dim
        self.index = None  # Built dynamically based on size
        self.index_size = 0  # Track current number of vectors in index
        self.creation_time = None
        self.last_update_time = None

    def build(self, embeddings):
        """
        Build FAISS index from embeddings. Auto-selects index type based on dataset size.
        
        Args:
            embeddings: numpy array of shape (n, 512)
        """
        embeddings = embeddings.astype("float32")
        n = len(embeddings)
        self.index_size = n
        self.creation_time = time.time()
        self.last_update_time = time.time()
        
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
        """
        Search for top-k nearest neighbors.
        
        Args:
            query: numpy array of shape (512,) or (m, 512)
            k: number of neighbors to return
            
        Returns:
            Tuple of (scores, indices)
        """
        if not self.index:
            return [], []
        
        # Handle single query vs batch
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        
        query = query.astype("float32")
        scores, indices = self.index.search(query, k)
        
        # If single query, return 1D arrays
        if scores.shape[0] == 1:
            return scores[0], indices[0]
        return scores, indices

    def add(self, embeddings):
        """
        Add new embeddings to existing index (incremental update).
        For large incremental updates, considers rebuilding if needed.
        
        Args:
            embeddings: numpy array of shape (n, 512)
        """
        if not self.index:
            self.build(embeddings)
            return

        embeddings = embeddings.astype("float32")
        num_new = len(embeddings)
        
        # For IVFFlat with massive incremental updates, rebuild is better
        # Threshold: if adding >20% new data to existing index
        if isinstance(self.index, faiss.IndexIVFFlat):
            size_ratio = num_new / max(1, self.index_size)
            if size_ratio > 0.2 and self.index_size > 10000:
                print(f"⚠️  Large incremental update ({num_new}/{self.index_size}). Rebuilding index for accuracy...")
                all_embeddings = self._extract_embeddings()
                if all_embeddings is not None:
                    all_embeddings = np.vstack([all_embeddings, embeddings])
                    self.build(all_embeddings)
                    return
        
        # Standard incremental add
        self.index.add(embeddings)
        self.index_size += num_new
        self.last_update_time = time.time()

    def add_incremental(self, new_embeddings, force_rebuild=False):
        """
        Controlled incremental update with optional force rebuild.
        
        Args:
            new_embeddings: numpy array of new embeddings
            force_rebuild: If True, rebuilds entire index (accurate but slower)
            
        Returns:
            dict with update stats
        """
        if not self.index:
            self.build(new_embeddings)
            return {'mode': 'build', 'count': len(new_embeddings), 'total_size': len(new_embeddings)}
        
        start_time = time.time()
        num_new = len(new_embeddings)
        old_size = self.index_size
        
        if force_rebuild:
            all_embeddings = self._extract_embeddings()
            if all_embeddings is not None:
                all_embeddings = np.vstack([all_embeddings, new_embeddings.astype('float32')])
                self.build(all_embeddings)
        else:
            self.add(new_embeddings)
        
        elapsed = time.time() - start_time
        
        return {
            'mode': 'rebuild' if force_rebuild else 'incremental',
            'added': num_new,
            'old_size': old_size,
            'new_size': self.index_size,
            'elapsed_seconds': elapsed,
            'speed_embeddings_per_sec': num_new / max(0.001, elapsed)
        }

    def _extract_embeddings(self):
        """
        Extract all embeddings currently in the index (for rebuilding).
        Warning: Not all index types support this efficiently.
        
        Returns:
            numpy array of embeddings, or None if extraction not supported
        """
        try:
            # For IndexFlatIP, we can reconstruct
            if isinstance(self.index, faiss.IndexFlatIP):
                n = self.index.ntotal
                embeddings = np.zeros((n, self.dim), dtype='float32')
                for i in range(n):
                    embeddings[i] = self.index.reconstruct(i)
                return embeddings
            else:
                # For other index types, reconstruction is complex
                print("⚠️  Cannot efficiently extract embeddings from this index type")
                return None
        except Exception as e:
            print(f"❌ Error extracting embeddings: {e}")
            return None

    def get_stats(self) -> dict:
        """Get index statistics."""
        if not self.index:
            return {'size': 0, 'status': 'not_initialized'}
        
        return {
            'size': self.index_size,
            'index_type': type(self.index).__name__,
            'creation_time': self.creation_time,
            'last_update': self.last_update_time,
            'age_seconds': time.time() - self.creation_time if self.creation_time else None
        }

