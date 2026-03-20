from fastapi import FastAPI, UploadFile, File, Query
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware
from core.search import FaceSearch
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from storage.store import Storage
from core.cluster import merge_two_clusters
from pydantic import BaseModel
from core.cluster import suggest_merges_fast
from core.cluster import get_cluster_representatives
from fastapi.staticfiles import StaticFiles


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/images", StaticFiles(directory="data/images"), name="images")
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

    # compute representatives
    reps = get_cluster_representatives(embeddings, cluster_dict)

    response = {}

    for person_id, indices in cluster_dict.items():
        faces = []

        for idx in indices:
            data = metadata[idx]

            faces.append({
                "idx": idx,  # 🔥 ADD THIS
                "image": data["image"],
                "face_id": data["face_id"],
                "bbox": data["bbox"]
            })

        rep_idx = reps[person_id]

        response[person_id] = {
            "representative": rep_idx,
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
def get_suggestions(
    threshold: float = Query(0.3, ge=0.0, le=1.0),
    limit: int = Query(20, ge=1, le=200)
):
    storage = Storage()

    embeddings = storage.load_embeddings()
    cluster_dict = storage.load_clusters()

    if embeddings is None or cluster_dict is None:
        return {"error": "No data found"}

    suggestions = suggest_merges_fast(
        embeddings,
        cluster_dict,
        threshold=threshold
    )

    response = []

    for s in suggestions[:limit]:
        response.append({
            "cluster1": s[0],
            "cluster2": s[1],
            "score": float(s[2])
        })

    return {
        "total_suggestions": len(response),
        "suggestions": response
    }

import cv2
from fastapi.responses import Response

@app.get("/thumbnail/{idx}")
def get_thumbnail(idx: int):
    metadata = storage.load_metadata()

    if idx >= len(metadata):
        return {"error": "Invalid index"}

    data = metadata[idx]

    img_path = os.path.join("data/images", data["image"])
    image = cv2.imread(img_path)

    if image is None:
        return {"error": "Image not found"}

    x1, y1, x2, y2 = data["bbox"]
    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return {"error": "Invalid crop"}

    _, buffer = cv2.imencode(".jpg", face)

    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.get("/objects/{image_name}")
def get_objects(image_name: str):
    metadata = storage.load_metadata()

    seen = set()
    unique_objects = []

    for m in metadata:
        if m["image"] == image_name:
            for obj in m.get("objects", []):
                key = (obj["label"], tuple(obj["bbox"]))

                if key not in seen:
                    seen.add(key)
                    unique_objects.append(obj)

    return {"objects": unique_objects}