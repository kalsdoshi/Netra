from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import onnxruntime as ort


@dataclass
class Config:
    # ─── Paths ───────────────────────────────────────────────────
    model_dir: Path = Path("models")
    data_dir: Path = Path("data/images")
    storage_dir: Path = Path("data/storage")

    detector_model: str = "retinaface_10g.onnx"
    embedder_model: str = "arcface_w600k_r50.onnx"

    # ─── Detection ───────────────────────────────────────────────
    det_input_size: tuple = (640, 640)   # RetinaFace expects 640×640
    det_conf_threshold: float = 0.5      # face confidence minimum
    det_nms_threshold: float = 0.4       # NMS overlap threshold
    det_max_faces: int = 20              # per image cap

    # ─── Alignment ───────────────────────────────────────────────
    align_output_size: int = 112         # ArcFace canonical: 112×112
    align_reference_points: list = field(default_factory=lambda: [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # left mouth corner
        [70.7299, 92.2041],   # right mouth corner
    ])  # ArcFace canonical landmark coordinates

    # ─── Embedding ───────────────────────────────────────────────
    emb_dim: int = 512
    emb_batch_size: int = 32             # faces per forward pass

    # ─── FAISS ───────────────────────────────────────────────────
    faiss_index_type: str = "flat"       # "flat" | "ivf" | "hnsw"
    faiss_metric: str = "cosine"         # cosine on L2-normalized vecs

    # ─── Clustering ──────────────────────────────────────────────
    cluster_min_cluster_size: int = 2    # min faces per identity
    cluster_min_samples: int = 1
    cluster_metric: str = "cosine"
    cluster_similarity_threshold: float = 0.68  # tune per dataset

    # ─── Runtime ─────────────────────────────────────────────────
    use_gpu: bool = True                 # auto-detected, falls back to CPU
    num_threads: int = 4                 # ONNX intra-op threads
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    cache_embeddings: bool = True        # skip recomputation on rerun

    cluster_selection_epsilon: float = 0.0   # existing line — change default:
    cluster_selection_epsilon: float = 0.35  # cosine dist units [0, 2]
    
    def __post_init__(self):
        import os
        os.environ["ORT_DISABLE_ALL_WARNINGS"] = "1"  
        # Resolve paths
        self.model_dir = Path(self.model_dir)
        self.data_dir = Path(self.data_dir)
        self.storage_dir = Path(self.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect GPU
        available_providers = ort.get_available_providers()
        if self.use_gpu and "CUDAExecutionProvider" in available_providers:
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]
            if self.use_gpu:
                print("[Config] GPU requested but CUDA unavailable — using CPU")

    @property
    def detector_path(self) -> Path:
        return self.model_dir / self.detector_model

    @property
    def embedder_path(self) -> Path:
        return self.model_dir / self.embedder_model

    @property
    def embeddings_path(self) -> Path:
        return self.storage_dir / "embeddings.npy"

    @property
    def metadata_path(self) -> Path:
        return self.storage_dir / "metadata.json"

    @property
    def faiss_index_path(self) -> Path:
        return self.storage_dir / "index.faiss"

    @property
    def clusters_path(self) -> Path:
        return self.storage_dir / "clusters.json"