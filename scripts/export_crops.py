"""
Export aligned 112x112 face crops organized by cluster identity.

Output structure:
    data/crops/
        person_1/
            img001_face0.jpg
            img003_face1.jpg
        person_2/
            ...
        unassigned/
            ...

Run: python scripts/export_crops.py
"""
import cv2
import json
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config
from core.detector import FaceDetector
from core.aligner import FaceAligner


def export_crops(
    clusters_path: Path,
    output_dir: Path,
    config: Config,
    padding_factor: float = 0.20,   # extra padding around bbox for context
    save_aligned: bool = True,       # save the 112×112 aligned crop
    save_padded: bool = True,        # save a natural-looking padded crop too
):
    # Load cluster assignments
    if not clusters_path.exists():
        print(f"No clusters file found at {clusters_path}")
        print("Run: python main.py --stage cluster")
        return

    with open(clusters_path) as f:
        clusters = json.load(f)

    total_faces = sum(len(v) for v in clusters.values())
    print(f"Exporting {total_faces} face crops across {len(clusters)} groups...")

    detector = FaceDetector(config)
    aligner  = FaceAligner(config)

    # Image cache — avoid reloading the same image for multiple faces
    image_cache: dict[str, np.ndarray] = {}

    def load_image(path: str):
        if path not in image_cache:
            img = cv2.imread(path)
            if img is None:
                return None
            # Simple LRU: keep cache small
            if len(image_cache) > 10:
                oldest = next(iter(image_cache))
                del image_cache[oldest]
            image_cache[path] = img
        return image_cache[path]

    output_dir.mkdir(parents=True, exist_ok=True)
    exported = 0
    failed   = 0

    for identity, faces in tqdm(clusters.items(), desc="Identities"):
        person_dir = output_dir / identity
        person_dir.mkdir(exist_ok=True)

        for face_entry in faces:
            img_path = face_entry["image_path"]
            face_idx = face_entry["face_idx"]
            bbox     = np.array(face_entry["bbox"], dtype=np.float32)

            image = load_image(img_path)
            if image is None:
                failed += 1
                continue

            img_h, img_w = image.shape[:2]
            stem = Path(img_path).stem

            # ── Padded natural crop ──────────────────────────────────────
            if save_padded:
                x1, y1, x2, y2 = bbox
                bw = x2 - x1
                bh = y2 - y1
                pad_x = bw * padding_factor
                pad_y = bh * padding_factor

                px1 = max(0,     int(x1 - pad_x))
                py1 = max(0,     int(y1 - pad_y))
                px2 = min(img_w, int(x2 + pad_x))
                py2 = min(img_h, int(y2 + pad_y))

                padded_crop = image[py1:py2, px1:px2]
                if padded_crop.size > 0:
                    out_path = person_dir / f"{stem}_face{face_idx}_natural.jpg"
                    cv2.imwrite(str(out_path), padded_crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])

            # ── Aligned 112×112 crop ─────────────────────────────────────
            if save_aligned:
                # Re-detect to get landmarks (bbox alone isn't enough for alignment)
                detections = detector.detect(image)

                # Match detection to this face by bbox IoU
                best_det  = None
                best_iou  = 0.0
                for det in detections:
                    iou = _bbox_iou(bbox, det.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_det = det

                if best_det is not None and best_iou > 0.4:
                    aligned = aligner.align(image, best_det)
                    if aligned is not None:
                        out_path = person_dir / f"{stem}_face{face_idx}_aligned.jpg"
                        cv2.imwrite(str(out_path), aligned,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
                        exported += 1
                    else:
                        failed += 1
                else:
                    # Fallback: resize bbox crop to 112×112
                    x1, y1, x2, y2 = bbox.astype(int)
                    crop = image[y1:y2, x1:x2]
                    if crop.size > 0:
                        resized  = cv2.resize(crop, (112, 112))
                        out_path = person_dir / f"{stem}_face{face_idx}_aligned.jpg"
                        cv2.imwrite(str(out_path), resized,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
                        exported += 1

    print(f"\nDone.")
    print(f"  Exported: {exported} aligned crops")
    print(f"  Failed:   {failed}")
    print(f"  Output:   {output_dir.resolve()}")

    # Print folder summary
    print("\nCrop counts per identity:")
    for identity in sorted(output_dir.iterdir(), key=lambda p: p.name):
        if identity.is_dir():
            n = len(list(identity.glob("*.jpg")))
            print(f"  {identity.name:<20}  {n} crops")


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-7)


if __name__ == "__main__":
    cfg    = Config()
    export_crops(
        clusters_path=cfg.clusters_path,
        output_dir=cfg.data_dir.parent / "crops",
        config=cfg,
        padding_factor=0.20,
        save_aligned=True,
        save_padded=True,
    )