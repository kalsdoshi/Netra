"""
Smoke tests for individual pipeline components.
Run: python -m pytest tests/ -v
"""
import pytest
import numpy as np
import cv2
from config import Config
from core.aligner import FaceAligner
from core.embedder import FaceEmbedder
from core.detector import FaceDetection
from core.cluster import FaceClusterer


@pytest.fixture
def config():
    return Config(use_gpu=False)


def test_aligner_produces_correct_shape(config):
    """Aligner must produce exactly (112, 112, 3) crops."""
    aligner = FaceAligner(config)
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Synthesize a detection with plausible landmarks
    landmarks = np.array([
        [200, 180], [250, 180], [225, 210],
        [200, 240], [250, 240]
    ], dtype=np.float32)

    det = FaceDetection(
        bbox=np.array([180, 160, 270, 270]),
        landmarks=landmarks,
        confidence=0.95,
    )
    aligned = aligner.align(dummy_image, det)
    assert aligned is not None
    assert aligned.shape == (112, 112, 3)


def test_l2_normalization():
    """L2 normalization must produce unit vectors."""
    vecs = np.random.randn(10, 512).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normalized = vecs / norms
    lengths = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(lengths, np.ones(10), atol=1e-5)


def test_clustering_empty_input(config):
    """Clusterer must not crash on empty input."""
    clusterer = FaceClusterer(config)
    labels, probs = clusterer.cluster(np.empty((0, 512), dtype=np.float32))
    assert len(labels) == 0


def test_clustering_single_face(config):
    """Single face must be assigned to cluster 0."""
    clusterer = FaceClusterer(config)
    emb = np.random.randn(1, 512).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels, probs = clusterer.cluster(emb)
    assert labels[0] == 0


def test_clustering_two_distinct_identities(config):
    """Two tight clusters should be separated."""
    clusterer = FaceClusterer(config)

    # Build two well-separated clusters of 5 faces each
    center_a = np.random.randn(512).astype(np.float32)
    center_b = -center_a  # opposite direction = maximally dissimilar

    embs_a = center_a + np.random.randn(5, 512).astype(np.float32) * 0.01
    embs_b = center_b + np.random.randn(5, 512).astype(np.float32) * 0.01

    embeddings = np.vstack([embs_a, embs_b])
    # L2 normalize
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    labels, _ = clusterer.cluster(embeddings)
    unique_labels = set(labels[labels != -1])
    assert len(unique_labels) == 2, f"Expected 2 clusters, got {unique_labels}"

## Complete File Tree
"""
face_recognition/
├── config.py
├── main.py
├── requirements.txt
│
├── core/
│   ├── __init__.py
│   ├── detector.py        ← RetinaFace ONNX
│   ├── aligner.py         ← Similarity transform, 112×112 crop
│   ├── embedder.py        ← ArcFace ONNX, L2-normalized 512-d
│   ├── similarity.py      ← FAISS IndexFlatIP
│   └── cluster.py         ← HDBSCAN identity grouping
│
├── pipeline/
│   ├── __init__.py
│   └── process_images.py  ← Orchestrator with caching
│
├── storage/
│   ├── __init__.py
│   └── store.py           ← Atomic save/load, dedup
│
├── models/
│   ├── retinaface_10g.onnx
│   └── arcface_w600k_r50.onnx
│
├── data/
│   ├── images/            ← your input images
│   └── storage/           ← auto-created
│       ├── embeddings.npy
│       ├── metadata.json
│       ├── index.faiss
│       └── clusters.json
│
├── scripts/
│   └── download_models.py
│
└── tests/
    └── test_pipeline.py
    """