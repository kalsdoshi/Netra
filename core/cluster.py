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