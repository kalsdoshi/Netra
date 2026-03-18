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