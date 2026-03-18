from fastapi import FastAPI, UploadFile, File
import shutil
import os

from core.search import FaceSearch
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from storage.store import Storage
from core.cluster import merge_two_clusters
from pydantic import BaseModel
from core.cluster import suggest_merges_fast
from core.cluster import get_cluster_representatives

app = FastAPI()

# --- Load system once ---
storage = Storage()
index = storage.load_faiss_index()
metadata = storage.load_metadata()

detector = FaceDetector(use_gpu=False)
embedder = FaceEmbedder()

search_engine = FaceSearch(detector, embedder, index)
embeddings = storage.load_embeddings()


@app.post("/search-face")
async def search_face(file: UploadFile = File(...)):
    # save temp file
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # run search
    scores, indices = search_engine.search(temp_path, top_k=10)

    # remove temp file
    os.remove(temp_path)

    # filter results
    threshold = 0.5
    results = []

    for i, idx in enumerate(indices):
        if scores[i] >= threshold:
            results.append({
                "image": metadata[idx]["image"],
                "score": float(scores[i])
            })

    return {
        "results": results
    }
@app.get("/clusters")
def get_clusters():
    storage = Storage()

    cluster_dict = storage.load_clusters()
    metadata = storage.load_metadata()

    if cluster_dict is None or metadata is None:
        return {"error": "No data found. Run processing first."}

    response = {}

    for person_id, indices in cluster_dict.items():
        faces = []

        for idx in indices:
            data = metadata[idx]

            faces.append({
                "image": data["image"],
                "face_id": data["face_id"],
                "bbox": data["bbox"]
            })

        # compute representatives
        reps = get_cluster_representatives(embeddings, cluster_dict)

        response = {}

        for person_id, indices in cluster_dict.items():
            faces = []

            for idx in indices:
                data = metadata[idx]

                faces.append({
                    "image": data["image"],
                    "face_id": data["face_id"],
                    "bbox": data["bbox"]
                })

            rep_idx = reps[person_id]

            response[person_id] = {
                "representative": metadata[rep_idx]["image"],
                "size": len(indices),
                "faces": faces
            }

    return {
        "total_clusters": len(response),
        "clusters": response
    }

class MergeRequest(BaseModel):
    id1: str
    id2: str

@app.post("/merge")
def merge_clusters(req: MergeRequest):
    storage = Storage()

    cluster_dict = storage.load_clusters()
    metadata = storage.load_metadata()

    if cluster_dict is None:
        return {"error": "No clusters found"}

    # perform merge
    updated_clusters = merge_two_clusters(cluster_dict, req.id1, req.id2)

    # save updated clusters
    storage.save_clusters(updated_clusters)

    return {
        "message": f"Merged {req.id2} into {req.id1}",
        "total_clusters": len(updated_clusters),
        "clusters": updated_clusters
    }

@app.get("/suggestions")
def get_suggestions():
    storage = Storage()

    embeddings = storage.load_embeddings()
    cluster_dict = storage.load_clusters()

    if embeddings is None or cluster_dict is None:
        return {"error": "No data found"}

    suggestions = suggest_merges_fast(
        embeddings,
        cluster_dict,
        threshold=0.5
    )

    response = []

    for s in suggestions[:10]:
        response.append({
            "cluster1": s[0],
            "cluster2": s[1],
            "score": float(s[2])
        })

    return {
        "total_suggestions": len(response),
        "suggestions": response
    }