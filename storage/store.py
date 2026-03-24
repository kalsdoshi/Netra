import json
import numpy as np
import os
import sqlite3
import faiss

class Storage:
    """
    Hybrid storage backend:
    - SQLite for metadata, clusters, graph (fast indexed queries)
    - Binary files for embeddings.npy and faiss.index (already optimized)
    """
    def __init__(self, base_path="storage_data"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.db_path = os.path.join(base_path, "netra.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for concurrency
            conn.execute("PRAGMA synchronous=NORMAL")
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    idx INTEGER PRIMARY KEY,
                    image TEXT NOT NULL,
                    face_id INTEGER,
                    bbox TEXT,
                    objects TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_image ON metadata(image)")
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    cluster_id TEXT NOT NULL,
                    face_idx INTEGER NOT NULL,
                    PRIMARY KEY (cluster_id, face_idx)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_clusters_id ON clusters(cluster_id)")
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id TEXT PRIMARY KEY,
                    clusters TEXT,
                    objects TEXT,
                    face_indices TEXT,
                    community INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_edges (
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    weight REAL,
                    face_sim REAL,
                    object_sim REAL,
                    modality TEXT,
                    PRIMARY KEY (source, target)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target)")
            
            conn.commit()

    # ─── Embeddings (binary, unchanged) ───
    def save_embeddings(self, embeddings):
        np.save(os.path.join(self.base_path, "embeddings.npy"), embeddings)

    def load_embeddings(self):
        path = os.path.join(self.base_path, "embeddings.npy")
        if os.path.exists(path):
            return np.load(path)
        return None

    # ─── FAISS Index (binary, unchanged) ───
    def save_faiss_index(self, index):
        faiss.write_index(index, os.path.join(self.base_path, "faiss.index"))

    def load_faiss_index(self):
        path = os.path.join(self.base_path, "faiss.index")
        if os.path.exists(path):
            return faiss.read_index(path)
        return None

    # ─── Metadata (SQLite) ───
    def save_metadata(self, metadata):
        clean_metadata = [
            {k: v for k, v in m.items() if k != "face_crop"}
            for m in metadata
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM metadata")
            conn.executemany(
                "INSERT INTO metadata (idx, image, face_id, bbox, objects) VALUES (?, ?, ?, ?, ?)",
                [
                    (i, m["image"], m.get("face_id"), 
                     json.dumps(m.get("bbox", [])), 
                     json.dumps(m.get("objects", [])))
                    for i, m in enumerate(clean_metadata)
                ]
            )
            conn.commit()
        
        # Also keep JSON for backward compatibility during transition
        path = os.path.join(self.base_path, "metadata.json")
        with open(path, "w") as f:
            json.dump(clean_metadata, f, indent=2)

    def load_metadata(self):
        # Try SQLite first (fast)
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT idx, image, face_id, bbox, objects FROM metadata ORDER BY idx"
                ).fetchall()
                
                if rows:
                    return [
                        {
                            "image": r[1],
                            "face_id": r[2],
                            "bbox": json.loads(r[3]) if r[3] else [],
                            "objects": json.loads(r[4]) if r[4] else []
                        }
                        for r in rows
                    ]
        except Exception:
            pass
        
        # Fallback to JSON
        path = os.path.join(self.base_path, "metadata.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    # ─── Clusters (SQLite) ───
    def save_clusters(self, cluster_dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM clusters")
            rows = []
            for cluster_id, indices in cluster_dict.items():
                for idx in indices:
                    rows.append((cluster_id, idx))
            conn.executemany(
                "INSERT INTO clusters (cluster_id, face_idx) VALUES (?, ?)", rows
            )
            conn.commit()
        
        # Also keep JSON for backward compatibility
        path = os.path.join(self.base_path, "clusters.json")
        with open(path, "w") as f:
            json.dump(cluster_dict, f, indent=2)

    def load_clusters(self):
        # Try SQLite first
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("SELECT cluster_id, face_idx FROM clusters").fetchall()
                if rows:
                    cluster_dict = {}
                    for cid, fidx in rows:
                        if cid not in cluster_dict:
                            cluster_dict[cid] = []
                        cluster_dict[cid].append(fidx)
                    return cluster_dict
        except Exception:
            pass
        
        # Fallback to JSON
        path = os.path.join(self.base_path, "clusters.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    # ─── Graph (SQLite) ───
    def save_graph(self, graph_data):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM graph_nodes")
            conn.execute("DELETE FROM graph_edges")
            
            # Save nodes
            conn.executemany(
                "INSERT INTO graph_nodes (id, clusters, objects, face_indices, community) VALUES (?, ?, ?, ?, ?)",
                [
                    (n["id"], json.dumps(n.get("clusters", [])), 
                     json.dumps(n.get("objects", [])),
                     json.dumps(n.get("face_indices", [])),
                     n.get("community", 0))
                    for n in graph_data.get("nodes", [])
                ]
            )
            
            # Save edges
            conn.executemany(
                "INSERT INTO graph_edges (source, target, weight, face_sim, object_sim, modality) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (e["source"], e["target"], e["weight"], 
                     e.get("face_sim", 0), e.get("object_sim", 0), 
                     e.get("modality", "none"))
                    for e in graph_data.get("edges", [])
                ]
            )
            conn.commit()
        
        # Also keep JSON for backward compatibility
        path = os.path.join(self.base_path, "graph.json")
        with open(path, "w") as f:
            json.dump(graph_data, f, indent=2)

    def load_graph(self):
        # Try SQLite first (fast indexed queries)
        try:
            with sqlite3.connect(self.db_path) as conn:
                node_rows = conn.execute("SELECT id, clusters, objects, face_indices, community FROM graph_nodes").fetchall()
                edge_rows = conn.execute("SELECT source, target, weight, face_sim, object_sim, modality FROM graph_edges").fetchall()
                
                if node_rows:
                    nodes = [
                        {
                            "id": r[0],
                            "clusters": json.loads(r[1]) if r[1] else [],
                            "objects": json.loads(r[2]) if r[2] else [],
                            "face_indices": json.loads(r[3]) if r[3] else [],
                            "community": r[4] or 0
                        }
                        for r in node_rows
                    ]
                    edges = [
                        {
                            "source": r[0], "target": r[1], "weight": r[2],
                            "face_sim": r[3], "object_sim": r[4], "modality": r[5]
                        }
                        for r in edge_rows
                    ]
                    return {
                        "nodes": nodes, "edges": edges,
                        "metrics": {"total_nodes": len(nodes), "total_edges": len(edges)}
                    }
        except Exception:
            pass
        
        # Fallback to JSON
        path = os.path.join(self.base_path, "graph.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    # ─── Paginated Queries (new for 10K+ UI) ───
    def load_metadata_paginated(self, page=1, limit=50, image_filter=None):
        offset = (page - 1) * limit
        with sqlite3.connect(self.db_path) as conn:
            if image_filter:
                rows = conn.execute(
                    "SELECT idx, image, face_id, bbox, objects FROM metadata WHERE image LIKE ? ORDER BY idx LIMIT ? OFFSET ?",
                    (f"%{image_filter}%", limit, offset)
                ).fetchall()
                total = conn.execute("SELECT COUNT(*) FROM metadata WHERE image LIKE ?", (f"%{image_filter}%",)).fetchone()[0]
            else:
                rows = conn.execute(
                    "SELECT idx, image, face_id, bbox, objects FROM metadata ORDER BY idx LIMIT ? OFFSET ?",
                    (limit, offset)
                ).fetchall()
                total = conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
            
            items = [
                {"image": r[1], "face_id": r[2], "bbox": json.loads(r[3]) if r[3] else [], "objects": json.loads(r[4]) if r[4] else []}
                for r in rows
            ]
            return {"items": items, "total": total, "page": page, "limit": limit, "pages": (total + limit - 1) // limit}

    def load_clusters_paginated(self, page=1, limit=20):
        """Load clusters with pagination for scalable UI."""
        full_clusters = self.load_clusters()
        if not full_clusters:
            return {"items": {}, "total": 0, "page": page, "limit": limit, "pages": 0}
        
        sorted_keys = sorted(full_clusters.keys())
        total = len(sorted_keys)
        start = (page - 1) * limit
        end = start + limit
        page_keys = sorted_keys[start:end]
        
        items = {k: full_clusters[k] for k in page_keys}
        return {"items": items, "total": total, "page": page, "limit": limit, "pages": (total + limit - 1) // limit}