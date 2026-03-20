import numpy as np

class GraphBuilder:
    def __init__(self, embeddings_array, metadata, cluster_dict, threshold=0.15, alpha=0.6, beta=0.4):
        self.embeddings = embeddings_array
        self.metadata = metadata
        self.cluster_dict = cluster_dict
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta

    def build_graph(self):
        nodes = {}
        
        # 1. Build Nodes (aggregate face-centric metadata into image-centric)
        for idx, m in enumerate(self.metadata):
            img = m["image"]
            if img not in nodes:
                # Deduplicate objects by label (take max confidence)
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
            
            # Find which cluster this face belongs to
            for c_id, indices in self.cluster_dict.items():
                if idx in indices:
                    nodes[img]["clusters"].add(c_id)
                    break

        # Convert sets to lists for JSON serialization
        images = list(nodes.keys())
        node_list = []
        for img in images:
            node_data = nodes[img].copy()
            node_data["clusters"] = list(node_data["clusters"])
            node_list.append(node_data)

        # 2. Build Edges
        edges = []
        n = len(images)
        
        for i in range(n):
            for j in range(i + 1, n):
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
                # If neither image has faces identified, weight fully on objects
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

        # 3. Form Communities (Louvain)
        try:
            import networkx as nx
            import community as community_louvain  # python-louvain
            
            G = nx.Graph()
            for n in node_list:
                G.add_node(n["id"])
            for e in edges:
                G.add_edge(e["source"], e["target"], weight=e["weight"])
                
            partition = community_louvain.best_partition(G, weight="weight")
            
            for n in node_list:
                n["community"] = partition.get(n["id"], 0)
            print(f"📊 Detected {len(set(partition.values()))} Graph Communities (Events/Albums)")
                
        except ImportError:
            pass  # Optional dependency, default to 0
            for n in node_list:
                n["community"] = 0

        # Return full graph
        return {
            "nodes": node_list,
            "edges": edges,
            "metrics": {
                "total_nodes": len(node_list),
                "total_edges": len(edges)
            }
        }
