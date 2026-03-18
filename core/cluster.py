import numpy as np

class FaceCluster:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def cluster(self, embeddings):
        clusters = []
        visited = set()

        for i in range(len(embeddings)):
            if i in visited:
                continue

            group = [i]
            visited.add(i)

            for j in range(i + 1, len(embeddings)):
                if j in visited:
                    continue

                sim = np.dot(embeddings[i], embeddings[j])

                if sim > self.threshold:
                    group.append(j)
                    visited.add(j)

            clusters.append(group)

        return clusters
    
def merge_clusters(cluster_dict, id1, id2):
    if id1 not in cluster_dict or id2 not in cluster_dict:
        return cluster_dict

    # merge id2 into id1
    cluster_dict[id1].extend(cluster_dict[id2])

    # remove id2
    del cluster_dict[id2]

    return cluster_dict


def assign_to_clusters(all_embeddings, new_embeddings, cluster_dict, old_count, threshold=0.5):
    """
    Assign new embeddings to existing clusters
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

            avg_sim = np.mean(sims)

            if avg_sim > best_score:
                best_score = avg_sim
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

from sklearn.cluster import DBSCAN


class DBSCANCluster:
    def __init__(self, eps=0.5, min_samples=2):
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self, embeddings):
        # normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="cosine"
        )

        labels = clustering.fit_predict(embeddings)

        clusters = {}

        for idx, label in enumerate(labels):
            if label == -1:
                # noise → treat as separate person
                clusters[f"person_noise_{idx}"] = [idx]
            else:
                key = f"person_{label}"
                if key not in clusters:
                    clusters[key] = []
                clusters[key].append(idx)

        return list(clusters.values())

import numpy as np

def get_cluster_representatives(embeddings, cluster_dict):
    """
    Return best representative index for each cluster
    """
    representatives = {}

    for person_id, indices in cluster_dict.items():
        if len(indices) == 1:
            representatives[person_id] = indices[0]
            continue

        best_idx = None
        best_score = -1

        for i in indices:
            sims = [np.dot(embeddings[i], embeddings[j]) for j in indices]
            avg_sim = np.mean(sims)

            if avg_sim > best_score:
                best_score = avg_sim
                best_idx = i

        representatives[person_id] = best_idx

    return representatives