from pipeline.process_images import ImageProcessor
from core.cluster import merge_clusters
from core.visualize import save_clusters
from storage.store import Storage
from core.faiss_index import FaissIndex
from core.cluster import suggest_merges_fast

if __name__ == "__main__":
    processor = ImageProcessor("data/images", use_gpu=False)

    embeddings, metadata, cluster_dict = processor.process()

    print(f"\nTotal embeddings: {embeddings.shape}")
    print(f"Total faces: {len(metadata)}")
    print(f"Total clusters (people): {len(cluster_dict)}")

    # 🔍 Smart suggestions
    suggestions = suggest_merges_fast(embeddings, cluster_dict, threshold=0.45)

    print("\n🤖 Suggested merges:")
    for s in suggestions[:5]:
        print(f"{s[0]} ↔ {s[1]} (score: {s[2]:.2f})")
    while True:
        user_input = input("\nMerge clusters? (format: person_1 person_3 or 'no'): ")

        if user_input.lower() == "no":
            break

        try:
            id1, id2 = user_input.split()

            if id1 not in cluster_dict or id2 not in cluster_dict:
                print("❌ Cluster ID not found")
                continue

            cluster_dict = merge_clusters(cluster_dict, id1, id2)

            print(f"✅ Merged {id2} into {id1}")
            print(f"Total clusters now: {len(cluster_dict)}")

            # 🔥 REGENERATE FOLDERS
            save_clusters(cluster_dict, metadata)

            print("📁 Folders updated!")

        except Exception as e:
            print("❌ Invalid input:", e)
    storage = Storage()

    storage.save_embeddings(embeddings)
    storage.save_metadata(metadata)
    storage.save_clusters(cluster_dict)

    print("💾 Data saved successfully!")

    index = storage.load_faiss_index()

    if index is not None:
        print("\n🔍 Testing FAISS search...")

        query = embeddings[0]
        scores, indices = index.search(query.reshape(1, -1).astype("float32"), 5)

        print("Similar faces:", indices)