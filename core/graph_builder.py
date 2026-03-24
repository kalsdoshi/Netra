import numpy as np
import faiss

class GraphBuilder:
    """
    Builds a relational graph between images using FAISS ANN candidate selection.
    Complexity: O(n · K · log n) instead of O(n²).
    """
    def __init__(self, embeddings_array, metadata, cluster_dict, 
                 threshold=0.15, alpha=0.6, beta=0.4, top_k=50):
        self.embeddings = embeddings_array
        self.metadata = metadata
        self.cluster_dict = cluster_dict
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.top_k = top_k  # Max neighbors to consider per image

    def _build_image_centroids(self, nodes, images):
        """
        Compute a centroid embedding for each image by averaging
        all face embeddings that appear in that image.
        Returns an (n_images, dim) array aligned with `images` order.
        """
        dim = self.embeddings.shape[1] if len(self.embeddings) > 0 else 512
        centroids = np.zeros((len(images), dim), dtype="float32")
        
        for i, img in enumerate(images):
            face_indices = nodes[img]["face_indices"]
            if face_indices:
                vecs = self.embeddings[face_indices].astype("float32")
                centroid = vecs.mean(axis=0)
                # L2-normalize for cosine similarity via inner product
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid /= norm
                centroids[i] = centroid
        
        return centroids

    def _get_candidate_pairs(self, centroids):
        """
        Use FAISS to find top-K nearest neighbor images for each image.
        Returns a set of (i, j) pairs where i < j.
        """
        n = len(centroids)
        k = min(self.top_k, n)  # Can't search more neighbors than images
        
        # Build a fast inner-product index on the centroids
        dim = centroids.shape[1]
        
        if n < 500:
            index = faiss.IndexFlatIP(dim)
        else:
            nlist = max(4, int(np.sqrt(n)))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(centroids)
            index.nprobe = max(1, nlist // 4)
        
        index.add(centroids)
        
        # Search: each image finds its K nearest neighbors
        _, indices = index.search(centroids, k)
        
        # Collect unique pairs
        pairs = set()
        for i in range(n):
            for j_idx in indices[i]:
                if j_idx < 0 or j_idx == i:
                    continue
                pair = (min(i, j_idx), max(i, j_idx))
                pairs.add(pair)
        
        return pairs

    def build_graph(self):
        nodes = {}
        
        # 1. Build Nodes (aggregate face-centric metadata into image-centric)
        for idx, m in enumerate(self.metadata):
            img = m["image"]
            if img not in nodes:
                unique_objects = {}
                for obj in m.get("objects", []):
                    lbl = obj["label"]
                    if lbl not in unique_objects or obj["confidence"] > unique_objects[lbl]["confidence"]:
                        unique_objects[lbl] = obj
                
                nodes[img] = {
                    "id": img,
                    "clusters": set(),
                    "objects": list(unique_objects.values()),
                    "face_indices": []
                }
            
            nodes[img]["face_indices"].append(idx)
            
            for c_id, c_indices in self.cluster_dict.items():
                if idx in c_indices:
                    nodes[img]["clusters"].add(c_id)
                    break

        images = list(nodes.keys())
        n = len(images)
        
        # Convert sets to lists for JSON serialization
        node_list = []
        for img in images:
            node_data = nodes[img].copy()
            node_data["clusters"] = list(node_data["clusters"])
            node_list.append(node_data)
        
        # 2. FAISS Candidate Selection (replaces O(n²) loop)
        print(f"⚡ Building image centroids for {n} images...")
        centroids = self._build_image_centroids(nodes, images)
        
        print(f"🔍 Finding top-{self.top_k} candidate pairs via FAISS ANN...")
        candidate_pairs = self._get_candidate_pairs(centroids)
        print(f"📊 {len(candidate_pairs)} candidate pairs (vs {n*(n-1)//2} brute-force pairs = {len(candidate_pairs)*100/max(1,n*(n-1)//2):.1f}% of total)")
        
        # 3. Score only candidate pairs
        edges = []
        for (i, j) in candidate_pairs:
            img_a = images[i]
            img_b = images[j]
            
            node_a = nodes[img_a]
            node_b = nodes[img_b]
            
            # Face Similarity (Jaccard on clusters)
            clusters_a = node_a["clusters"]
            clusters_b = node_b["clusters"]
            
            if not clusters_a and not clusters_b:
                face_sim = 0.0
            else:
                intersection = clusters_a & clusters_b
                union = clusters_a | clusters_b
                face_sim = len(intersection) / len(union)
            
            # Object Similarity (Jaccard on labels)
            labels_a = {obj["label"] for obj in node_a["objects"]}
            labels_b = {obj["label"] for obj in node_b["objects"]}
            
            if not labels_a and not labels_b:
                object_sim = 0.0
            else:
                intersection = labels_a & labels_b
                union = labels_a | labels_b
                object_sim = len(intersection) / len(union)
            
            # Multimodal Fusion
            if not clusters_a and not clusters_b:
                score = object_sim
                modality = "object"
            else:
                score = self.alpha * face_sim + self.beta * object_sim
                if face_sim > 0 and self.alpha * face_sim > self.beta * object_sim:
                    modality = "face"
                elif object_sim > 0:
                    modality = "object"
                else:
                    modality = "none"
            
            if score > self.threshold:
                edges.append({
                    "source": img_a,
                    "target": img_b,
                    "weight": float(score),
                    "face_sim": float(face_sim),
                    "object_sim": float(object_sim),
                    "modality": modality
                })

        print(f"✅ Graph built: {len(node_list)} nodes, {len(edges)} edges")

        # 4. Form Communities (Louvain) — optional
        try:
            import networkx as nx
            import community as community_louvain
            
            G = nx.Graph()
            for nd in node_list:
                G.add_node(nd["id"])
            for e in edges:
                G.add_edge(e["source"], e["target"], weight=e["weight"])
                
            partition = community_louvain.best_partition(G, weight="weight")
            
            for nd in node_list:
                nd["community"] = partition.get(nd["id"], 0)
            print(f"📊 Detected {len(set(partition.values()))} Graph Communities")
                
        except ImportError:
            for nd in node_list:
                nd["community"] = 0

        return {
            "nodes": node_list,
            "edges": edges,
            "metrics": {
                "total_nodes": len(node_list),
                "total_edges": len(edges),
                "candidate_pairs_evaluated": len(candidate_pairs),
                "brute_force_pairs_avoided": n*(n-1)//2 - len(candidate_pairs)
            }
        }
