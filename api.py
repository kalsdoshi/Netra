from fastapi import FastAPI, UploadFile, File, Query
import shutil
import os
import hashlib
from functools import lru_cache
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

# ─── In-Memory Cache Layer ──────────────────────────────────────
# Avoids repeated SQLite reads on every /thumbnail and /objects request.
_cache = {
    "metadata": metadata,
    "embeddings": embeddings,
    "version": 0  # bumped on mutations (merge, etc.)
}

THUMBS_DIR = os.path.join("storage_data", "thumbs")
os.makedirs(THUMBS_DIR, exist_ok=True)


def _get_metadata():
    """Return cached metadata, reload if invalidated."""
    if _cache["metadata"] is None:
        _cache["metadata"] = storage.load_metadata()
    return _cache["metadata"]


def _invalidate_cache():
    """Call after any mutation (merge, rebuild, etc.)."""
    _cache["metadata"] = None
    _cache["version"] += 1


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
    meta = _get_metadata()

    for i, idx in enumerate(indices):
        if scores[i] >= threshold:
            results.append({
                "image": meta[idx]["image"],
                "score": float(scores[i])
            })

    return {
        "results": results
    }

@app.get("/clusters")
def get_clusters():
    storage_local = Storage()

    cluster_dict = storage_local.load_clusters()
    meta = _get_metadata()

    if cluster_dict is None or meta is None:
        return {"error": "No data found. Run processing first."}

    # compute representatives
    reps = get_cluster_representatives(embeddings, cluster_dict)

    response = {}

    for person_id, indices in cluster_dict.items():
        faces = []

        for idx in indices:
            data = meta[idx]

            faces.append({
                "idx": idx,
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

@app.get("/clusters/paginated")
def get_clusters_paginated(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100)
):
    storage_local = Storage()
    cluster_dict = storage_local.load_clusters()
    meta = _get_metadata()
    
    if cluster_dict is None or meta is None:
        return {"error": "No data found. Run processing first."}
    
    reps = get_cluster_representatives(embeddings, cluster_dict)
    
    sorted_keys = sorted(cluster_dict.keys())
    total = len(sorted_keys)
    start = (page - 1) * limit
    end = start + limit
    page_keys = sorted_keys[start:end]
    
    response = {}
    for person_id in page_keys:
        indices = cluster_dict[person_id]
        faces = []
        for idx in indices:
            data = meta[idx]
            faces.append({
                "idx": idx,
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
        "total_clusters": total,
        "page": page,
        "limit": limit,
        "pages": (total + limit - 1) // limit,
        "clusters": response
    }

class MergeRequest(BaseModel):
    id1: str
    id2: str

@app.post("/merge")
def merge_clusters(req: MergeRequest):
    storage_local = Storage()

    cluster_dict = storage_local.load_clusters()

    if cluster_dict is None:
        return {"error": "No clusters found"}

    # perform merge
    updated_clusters = merge_two_clusters(cluster_dict, req.id1, req.id2)

    # save updated clusters
    storage_local.save_clusters(updated_clusters)

    # Invalidate caches after mutation
    _invalidate_cache()

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
    storage_local = Storage()

    embs = storage_local.load_embeddings()
    cluster_dict = storage_local.load_clusters()

    if embs is None or cluster_dict is None:
        return {"error": "No data found"}

    suggestions = suggest_merges_fast(
        embs,
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
    meta = _get_metadata()

    if meta is None or idx >= len(meta):
        return {"error": "Invalid index"}

    # Check disk cache first — avoids expensive cv2.imread + crop
    cache_path = os.path.join(THUMBS_DIR, f"{idx}.jpg")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            content = f.read()
        return Response(
            content=content, 
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=31536000, immutable"}
        )

    data = meta[idx]

    img_path = os.path.join("data/images", data["image"])
    image = cv2.imread(img_path)

    if image is None:
        return {"error": "Image not found"}

    x1, y1, x2, y2 = data["bbox"]
    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return {"error": "Invalid crop"}

    _, buffer = cv2.imencode(".jpg", face, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    content = buffer.tobytes()

    # Save to disk cache for future requests
    try:
        with open(cache_path, "wb") as f:
            f.write(content)
    except Exception:
        pass  # Non-critical — cache miss is fine

    return Response(
        content=content, 
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"}
    )

@app.get("/image-thumb/{image_name}")
def get_image_thumb(image_name: str):
    # Fast whole-image thumbnail for the 3D graph display
    img_path = os.path.join("data/images", image_name)
    
    # Check disk cache
    safe_name = hashlib.md5(image_name.encode()).hexdigest()
    cache_path = os.path.join(THUMBS_DIR, f"img_{safe_name}.jpg")
    
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            content = f.read()
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=31536000, immutable"}
        )
    
    image = cv2.imread(img_path)
    
    if image is None:
         return {"error": "Image not found"}
         
    # Resize to lightweight 256x256 resolution
    thumb = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    
    # Compress aggressive to load fast over network
    _, buffer = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    content = buffer.tobytes()
    
    # Save to disk cache
    try:
        with open(cache_path, "wb") as f:
            f.write(content)
    except Exception:
        pass
    
    return Response(
        content=content, 
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"}
    )

@app.get("/objects/{image_name}")
def get_objects(image_name: str):
    meta = _get_metadata()

    seen = set()
    unique_objects = []

    for m in meta:
        if m["image"] == image_name:
            for obj in m.get("objects", []):
                key = (obj["label"], tuple(obj["bbox"]))

                if key not in seen:
                    seen.add(key)
                    unique_objects.append(obj)

    return {"objects": unique_objects}

@app.get("/graph")
def get_graph():
    storage_local = Storage()
    graph_data = storage_local.load_graph()
    
    if graph_data is None:
        return {"error": "Graph data not found. Run processing first or trigger a rebuild."}
        
    return graph_data

@app.post("/graph/rebuild")
def rebuild_graph():
    storage_local = Storage()
    embs = storage_local.load_embeddings()
    meta = storage_local.load_metadata()
    cluster_dict = storage_local.load_clusters()
    
    if embs is None or meta is None or cluster_dict is None:
        return {"error": "Missing base data to build graph."}
        
    from core.graph_builder import GraphBuilder
    graph_builder = GraphBuilder(embs, meta, cluster_dict, threshold=0.15)
    graph_data = graph_builder.build_graph()
    
    storage_local.save_graph(graph_data)
    
    # Invalidate caches after rebuild
    _invalidate_cache()
    
    return {
        "message": "Graph rebuilt successfully",
        "metrics": graph_data.get("metrics", {})
    }


# ─── TIMELINE ENDPOINT ─────────────────────────────────────────
from PIL import Image as PILImage
from datetime import datetime
from collections import defaultdict

@app.get("/timeline")
def get_timeline():
    """
    Classify ALL images by date taken (EXIF) or file modification time.
    Returns images grouped by date, sorted chronologically.
    """
    img_dir = "data/images"
    if not os.path.isdir(img_dir):
        return {"error": "Image directory not found"}
    
    image_files = [
        f for f in os.listdir(img_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ]
    
    timeline = defaultdict(list)
    
    for fname in image_files:
        fpath = os.path.join(img_dir, fname)
        date_taken = None
        
        # Try EXIF DateTimeOriginal first (most accurate)
        try:
            img = PILImage.open(fpath)
            exif = img._getexif()
            if exif:
                # 36867 = DateTimeOriginal, 36868 = DateTimeDigitized, 306 = DateTime
                for tag_id in (36867, 36868, 306):
                    raw = exif.get(tag_id)
                    if raw:
                        dt = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
                        date_taken = dt.strftime("%Y-%m-%d")
                        break
        except Exception:
            pass
        
        # Fallback to file modification time
        if not date_taken:
            try:
                mtime = os.path.getmtime(fpath)
                date_taken = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            except Exception:
                date_taken = "Unknown"
        
        timeline[date_taken].append({
            "image": fname,
            "date": date_taken,
        })
    
    # Sort groups by date (newest first)
    sorted_dates = sorted(timeline.keys(), reverse=True)
    
    groups = []
    for date_str in sorted_dates:
        imgs = timeline[date_str]
        # Parse for display label
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            label = dt.strftime("%B %d, %Y")  # "June 09, 2025"
        except Exception:
            label = date_str
        
        groups.append({
            "date": date_str,
            "label": label,
            "count": len(imgs),
            "images": sorted(imgs, key=lambda x: x["image"]),
        })
    
    return {
        "total_images": len(image_files),
        "total_dates": len(groups),
        "groups": groups,
    }


# ─── CONTENT GROUPS ENDPOINT ───────────────────────────────────
from core.object_detector import ObjectDetector as _OD

# Lazy singleton to avoid re-loading YOLO
_content_detector = None

def _get_content_detector():
    global _content_detector
    if _content_detector is None:
        _content_detector = _OD(min_confidence=0.4)
    return _content_detector


@app.get("/content-groups")
def get_content_groups():
    """
    Classify ALL images by detected content (objects).
    Groups images by their dominant detected labels.
    Also includes a special 'People' group using face metadata.
    """
    img_dir = "data/images"
    if not os.path.isdir(img_dir):
        return {"error": "Image directory not found"}
    
    image_files = [
        f for f in os.listdir(img_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ]
    
    meta = _get_metadata()
    
    # Build face-image lookup from existing metadata
    face_images = set()
    if meta:
        face_images = set(m["image"] for m in meta)
    
    # Build object lookup from metadata (already processed)
    image_objects = {}
    if meta:
        for m in meta:
            img_name = m["image"]
            if img_name not in image_objects:
                image_objects[img_name] = set()
            for obj in m.get("objects", []):
                image_objects[img_name].add(obj["label"])
    
    # For images NOT in metadata, run object detection on the fly
    # (these are faceless images not in the pipeline output)
    detector = _get_content_detector()
    for fname in image_files:
        if fname not in image_objects:
            try:
                fpath = os.path.join(img_dir, fname)
                img = cv2.imread(fpath)
                if img is not None:
                    objs = detector.detect(img)
                    image_objects[fname] = set(o["label"] for o in objs)
                else:
                    image_objects[fname] = set()
            except Exception:
                image_objects[fname] = set()
    
    # Build content groups
    content_groups = defaultdict(list)
    uncategorized = []
    
    for fname in image_files:
        labels = image_objects.get(fname, set())
        has_face = fname in face_images
        
        if has_face:
            content_groups["People"].append(fname)
        
        if labels:
            for label in labels:
                if label != "person":  # avoid duplicating People group
                    # Capitalize the label nicely
                    display_label = label.replace("_", " ").title()
                    content_groups[display_label].append(fname)
        
        if not has_face and not labels:
            uncategorized.append(fname)
    
    if uncategorized:
        content_groups["Uncategorized"] = uncategorized
    
    # Sort groups by size (largest first)
    sorted_groups = sorted(
        content_groups.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )
    
    result = []
    for label, imgs in sorted_groups:
        result.append({
            "label": label,
            "count": len(imgs),
            "images": sorted(set(imgs)),  # deduplicate
        })
    
    return {
        "total_images": len(image_files),
        "total_groups": len(result),
        "groups": result,
    }