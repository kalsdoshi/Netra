import numpy as np
import hdbscan


def assign_to_clusters(all_embeddings, new_embeddings, cluster_dict, old_count, threshold=0.5):
    """
    Assign new embeddings to existing clusters using median similarity
    (more robust to outlier embeddings than mean).
    """

    updated_clusters = cluster_dict.copy()

    for i, new_emb in enumerate(new_embeddings):
        best_cluster = None
        best_score = -1

        new_index = old_count + i  # global index

        for person_id, indices in updated_clusters.items():
            sims = []

            for idx in indices:
                if idx < len(all_embeddings):  # safety
                    sims.append(np.dot(new_emb, all_embeddings[idx]))

            if len(sims) == 0:
                continue

            # Use median instead of mean — more robust to outliers
            med_sim = np.median(sims)

            if med_sim > best_score:
                best_score = med_sim
                best_cluster = person_id

        # assign
        if best_score > threshold and best_cluster is not None:
            updated_clusters[best_cluster].append(new_index)
        else:
            new_id = f"person_{len(updated_clusters)}"
            updated_clusters[new_id] = [new_index]

    return updated_clusters

def suggest_merges_fast(embeddings, cluster_dict, threshold):
    """
    Suggest cluster merges using centroid similarity
    """

    cluster_ids = list(cluster_dict.keys())
    centroids = {}

    # compute centroid for each cluster
    for cid in cluster_ids:
        indices = cluster_dict[cid]
        vecs = [embeddings[i] for i in indices]
        centroids[cid] = np.mean(vecs, axis=0)

    suggestions = []

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            c1 = cluster_ids[i]
            c2 = cluster_ids[j]

            sim = np.dot(centroids[c1], centroids[c2])

            if sim > threshold:
                suggestions.append((c1, c2, float(sim)))

    # sort by similarity (best first)
    suggestions.sort(key=lambda x: x[2], reverse=True)

    return suggestions

def merge_two_clusters(cluster_dict, id1, id2):
    if id1 not in cluster_dict or id2 not in cluster_dict:
        return cluster_dict

    # merge id2 into id1
    cluster_dict[id1].extend(cluster_dict[id2])

    # delete id2
    del cluster_dict[id2]

    return cluster_dict


class HDBSCANCluster:
    """
    HDBSCAN for face clustering — automatically determines optimal cluster count.
    Superior to DBSCAN because it doesn't require a fixed eps parameter and
    handles varying-density clusters more naturally.
    """
    def __init__(self, min_cluster_size=2, min_samples=1, 
                 cluster_selection_epsilon=0.35):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon

    def cluster(self, embeddings):
        if len(embeddings) == 0:
            return []

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # avoid division by zero
        embeddings = embeddings / norms

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",  # on L2-normalized vectors, euclidean ≈ cosine
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method="eom",  # Excess of Mass — better for faces
        )

        labels = clusterer.fit_predict(embeddings)

        clusters = {}

        for idx, label in enumerate(labels):
            if label == -1:
                # noise → treat as separate person (singleton cluster)
                clusters[f"person_noise_{idx}"] = [idx]
            else:
                key = f"person_{label}"
                if key not in clusters:
                    clusters[key] = []
                clusters[key].append(idx)

        return list(clusters.values())




def get_cluster_representatives(embeddings, cluster_dict):
    """
    Return best representative index for each cluster.
    Uses vectorized matrix multiply for O(n) per cluster instead of O(n²).
    """
    representatives = {}

    for person_id, indices in cluster_dict.items():
        if len(indices) == 1:
            representatives[person_id] = indices[0]
            continue

        # Vectorized: compute all pairwise similarities at once
        idx_array = np.array(indices)
        vecs = embeddings[idx_array].astype(np.float32)
        
        # Similarity matrix via matrix multiply (n×d @ d×n = n×n)
        sim_matrix = vecs @ vecs.T
        
        # Best representative = highest average similarity to all others
        avg_sims = sim_matrix.mean(axis=1)
        best_local = int(np.argmax(avg_sims))
        
        representatives[person_id] = indices[best_local]

    return representatives