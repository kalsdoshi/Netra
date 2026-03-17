"""
Face aligner using similarity transform to canonical 112×112 coordinates.

Why this matters:
  ArcFace was trained on aligned faces. If you feed an unaligned crop,
  accuracy drops dramatically. The similarity transform (rotation + scale + 
  translation, no shear) maps the 5 detected landmarks to the 5 reference
  positions the model was trained on.

Reference coordinates:
  These are the ArcFace canonical landmark positions within a 112×112 face.
  Using the wrong reference kills accuracy.
"""

from __future__ import annotations
import numpy as np
import cv2
from scipy import linalg
from typing import Optional

from config import Config
from core.detector import FaceDetection


class FaceAligner:
    """
    Aligns a face crop to ArcFace canonical coordinates using a
    similarity transform estimated from 5 facial landmarks.
    """

    def __init__(self, config: Config):
        self.config = config
        self.output_size = config.align_output_size  # 112
        # Reference landmarks in canonical coordinate space
        self.reference_pts = np.array(
            config.align_reference_points, dtype=np.float32
        )

    def _estimate_norm(
        self, landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Estimate the 2×3 similarity transform matrix (rotation + scale +
        translation) that maps detected landmarks to canonical positions.

        Uses least-squares fitting — more robust than picking 2 points.
        Returns a 2×3 affine transformation matrix.
        """
        assert landmarks.shape == (5, 2), f"Expected (5, 2) landmarks, got {landmarks.shape}"

        src = np.array(landmarks, dtype=np.float64)
        dst = np.array(self.reference_pts, dtype=np.float64)

        # Build the normal equations for similarity transform (4 DOF: tx, ty, scale, rotation)
        # This is the Umeyama method — better than OpenCV estimateAffinePartial2D for faces
        n = src.shape[0]

        mean_src = src.mean(axis=0)
        mean_dst = dst.mean(axis=0)

        src_centered = src - mean_src
        dst_centered = dst - mean_dst

        var_src = np.mean(np.sum(src_centered ** 2, axis=1))

        cov = (dst_centered.T @ src_centered) / n

        U, D, Vt = linalg.svd(cov)
        S = np.ones(2)
        if linalg.det(cov) < 0:
            S[-1] = -1  # reflection correction

        R = U @ np.diag(S) @ Vt
        scale = (D @ S) / var_src if var_src > 1e-8 else 1.0

        t = mean_dst - scale * R @ mean_src

        # Build 2×3 affine matrix
        M = np.zeros((2, 3), dtype=np.float64)
        M[:, :2] = scale * R
        M[:, 2] = t

        return M.astype(np.float32)

    def align(
        self, image: np.ndarray, detection: FaceDetection
    ) -> Optional[np.ndarray]:
        """
        Produce a 112×112 aligned face crop.

        Args:
            image: Full BGR image
            detection: FaceDetection with 5-point landmarks

        Returns:
            uint8 BGR array of shape (112, 112, 3), or None if alignment fails.
        """
        try:
            M = self._estimate_norm(detection.landmarks)
            aligned = cv2.warpAffine(
                image,
                M,
                (self.output_size, self.output_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            return aligned
        except Exception as e:
            print(f"[Aligner] Alignment failed: {e}")
            return None

    def align_batch(
        self, image: np.ndarray, detections: list[FaceDetection]
    ) -> list[Optional[np.ndarray]]:
        """Align all detected faces in a single image."""
        return [self.align(image, det) for det in detections]