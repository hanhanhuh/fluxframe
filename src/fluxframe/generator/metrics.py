#!/usr/bin/env python3
"""Distance metrics for image similarity matching."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .config import Config


class DistanceMetric(Protocol):
    """Protocol defining the interface for distance metrics."""

    def compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute distance between two LAB vectors.

        Args:
            vec1: First LAB vector (flattened 64x64x3)
            vec2: Second LAB vector (flattened 64x64x3)

        Returns:
            Distance score (lower = more similar)
        """
        ...

    def compute_batch_distance(
        self, vecs: np.ndarray, query: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Compute distances between multiple vectors and a query.

        Args:
            vecs: Array of LAB vectors (N, dims)
            query: Query LAB vector (dims,)
            weights: Weight vector for LAB channels (dims,)

        Returns:
            Array of distances (N,)
        """
        ...


class LABWeightedDistance:
    """Fast weighted Euclidean distance in LAB color space.

    This is the baseline metric from the original implementation.
    Emphasizes perceptual color channels (A, B) over luminance (L).
    """

    def __init__(self, cfg: Config):
        """Initialize metric with configuration.

        Args:
            cfg: Configuration object containing weights
        """
        self.cfg = cfg
        self.weights = np.array(cfg.weights, dtype=np.float32)

    def compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute weighted Euclidean distance."""
        diff = vec1 - vec2
        # Reshape to apply per-channel weights
        diff_reshaped = diff.reshape(-1, 3)
        weighted_diff = diff_reshaped * self.weights
        return float(np.sum(weighted_diff * weighted_diff))

    def compute_batch_distance(
        self, vecs: np.ndarray, query: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Vectorized batch distance computation."""
        diff = vecs - query
        weighted_diff = diff * weights
        return np.einsum("ij,ij->i", weighted_diff, weighted_diff)


class SSIMMetric:
    """SSIM-based metric for perceptually accurate similarity.

    Uses Structural Similarity Index on luminance channel (L).
    Slower than LAB distance but better correlates with human perception.
    """

    def __init__(self, cfg: Config):
        """Initialize SSIM metric.

        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.image_size = 64  # Fixed size from preprocessing

    def compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute SSIM-based distance.

        SSIM returns values in [-1, 1] where 1 = identical.
        We convert to distance: distance = (1 - ssim) * scale_factor
        """
        # Reshape vectors to 2D images (L channel only)
        img1 = vec1.reshape(self.image_size, self.image_size, 3)[:, :, 0]
        img2 = vec2.reshape(self.image_size, self.image_size, 3)[:, :, 0]

        # Compute SSIM on L channel
        ssim_value = ssim(img1, img2, data_range=255)

        # Convert to distance (scale to match LAB distance range ~0-50000)
        distance = (1.0 - ssim_value) * 25000.0
        return float(distance)

    def compute_batch_distance(
        self, vecs: np.ndarray, query: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Compute SSIM distances for batch of vectors."""
        n = len(vecs)
        distances = np.empty(n, dtype=np.float32)

        query_img = query.reshape(self.image_size, self.image_size, 3)[:, :, 0]

        for i in range(n):
            img = vecs[i].reshape(self.image_size, self.image_size, 3)[:, :, 0]
            ssim_value = ssim(query_img, img, data_range=255)
            distances[i] = (1.0 - ssim_value) * 25000.0

        return distances


class HybridMetric:
    """Hybrid metric combining LAB distance and SSIM.

    Balances speed (LAB) with perceptual accuracy (SSIM).
    Configurable weight controls the balance between methods.
    """

    def __init__(self, cfg: Config):
        """Initialize hybrid metric.

        Args:
            cfg: Configuration object with ssim_weight parameter
        """
        self.cfg = cfg
        self.lab_metric = LABWeightedDistance(cfg)
        self.ssim_metric = SSIMMetric(cfg)

        # Weight for SSIM (1 - weight goes to LAB)
        self.ssim_weight = cfg.ssim_weight
        self.lab_weight = 1.0 - cfg.ssim_weight

    def compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute weighted combination of LAB and SSIM distances."""
        lab_dist = self.lab_metric.compute_distance(vec1, vec2)
        ssim_dist = self.ssim_metric.compute_distance(vec1, vec2)

        return float(self.lab_weight * lab_dist + self.ssim_weight * ssim_dist)

    def compute_batch_distance(
        self, vecs: np.ndarray, query: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Compute hybrid distances for batch of vectors."""
        lab_dists = self.lab_metric.compute_batch_distance(vecs, query, weights)
        ssim_dists = self.ssim_metric.compute_batch_distance(vecs, query, weights)

        return self.lab_weight * lab_dists + self.ssim_weight * ssim_dists


def create_metric(cfg: Config) -> DistanceMetric:
    """Factory function to create appropriate distance metric.

    Args:
        cfg: Configuration object

    Returns:
        Distance metric instance based on cfg.metric

    Raises:
        ValueError: If metric type is unknown
    """
    if cfg.metric == "lab":
        return LABWeightedDistance(cfg)
    elif cfg.metric == "ssim":
        return SSIMMetric(cfg)
    elif cfg.metric == "lab+ssim":
        return HybridMetric(cfg)
    else:
        raise ValueError(f"Unknown metric type: {cfg.metric}")
