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
        """Compute weighted Euclidean distance.

        Args:
            vec1: First LAB vector (flattened 64x64x3 image)
            vec2: Second LAB vector (flattened 64x64x3 image)

        Returns:
            Weighted squared Euclidean distance between vectors.
        """
        diff = vec1 - vec2
        diff_reshaped = diff.reshape(-1, 3)
        weighted_diff = diff_reshaped * self.weights
        return float(np.sum(weighted_diff * weighted_diff))

    def compute_batch_distance(
        self, vecs: np.ndarray, query: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Vectorized batch distance computation.

        Args:
            vecs: Batch of LAB vectors (N x dims)
            query: Query LAB vector (dims,)
            weights: Channel weights for LAB (tiled to match dims)

        Returns:
            Array of distances for each vector in batch.
        """
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

        Args:
            vec1: First LAB vector (flattened 64x64x3 image)
            vec2: Second LAB vector (flattened 64x64x3 image)

        Returns:
            SSIM distance scaled to match LAB range (~0-50000).
        """
        img1 = vec1.reshape(self.image_size, self.image_size, 3)[:, :, 0]
        img2 = vec2.reshape(self.image_size, self.image_size, 3)[:, :, 0]

        ssim_value = ssim(img1, img2, data_range=255)

        distance = (1.0 - ssim_value) * 25000.0
        return float(distance)

    def compute_batch_distance(
        self, vecs: np.ndarray, query: np.ndarray, _weights: np.ndarray
    ) -> np.ndarray:
        """Compute SSIM distances for batch of vectors.

        Args:
            vecs: Batch of image vectors (N x dims)
            query: Query image vector (dims,)
            _weights: Weights (unused, SSIM doesn't use weights)

        Returns:
            Array of SSIM distances for each vector in batch.
        """
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
        """Compute weighted combination of LAB and SSIM distances.

        Args:
            vec1: First LAB vector (flattened 64x64x3 image)
            vec2: Second LAB vector (flattened 64x64x3 image)

        Returns:
            Weighted hybrid distance combining LAB and SSIM.
        """
        lab_dist = self.lab_metric.compute_distance(vec1, vec2)
        ssim_dist = self.ssim_metric.compute_distance(vec1, vec2)

        return float(self.lab_weight * lab_dist + self.ssim_weight * ssim_dist)

    def compute_batch_distance(
        self, vecs: np.ndarray, query: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Compute hybrid distances for batch of vectors.

        Args:
            vecs: Batch of LAB vectors (N x dims)
            query: Query LAB vector (dims,)
            weights: Channel weights for LAB component

        Returns:
            Array of weighted hybrid distances for each vector in batch.
        """
        lab_dists = self.lab_metric.compute_batch_distance(vecs, query, weights)
        ssim_dists = self.ssim_metric.compute_batch_distance(vecs, query, weights)

        return self.lab_weight * lab_dists + self.ssim_weight * ssim_dists


class GISTMetric:
    """GIST descriptor-based metric for scene-level matching.

    GIST (Generalized Search Tree) uses Gabor filters at multiple scales
    and orientations to capture the scene's spatial envelope.
    Best for matching overall scene structure and layout.
    """

    def __init__(self, cfg: Config):
        """Initialize GIST metric.

        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.image_size = 64
        self.scales = 3
        self.orientations = 8
        self.grid_size = 4

    def _compute_gist(self, img: np.ndarray) -> np.ndarray:
        """Compute GIST descriptor for an image.

        Args:
            img: Image array (grayscale, H x W)

        Returns:
            GIST feature vector
        """
        features = []

        for scale in range(self.scales):
            # Create Gabor filters at different orientations
            for orientation in range(self.orientations):
                # Gabor filter parameters
                theta = orientation * np.pi / self.orientations
                sigma = 2.0 * (scale + 1)
                lambd = sigma * 1.5
                gamma = 0.5
                psi = 0

                # Create Gabor kernel
                kernel_size = int(6 * sigma + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1

                kernel = cv2.getGaborKernel(
                    (kernel_size, kernel_size), sigma, theta, lambd, gamma, psi
                )

                # Apply filter
                filtered = cv2.filter2D(img, cv2.CV_32F, kernel)

                # Divide into grid and compute mean
                h, w = filtered.shape
                cell_h = h // self.grid_size
                cell_w = w // self.grid_size

                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        cell = filtered[
                            i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
                        ]
                        features.append(np.mean(np.abs(cell)))

        return np.array(features, dtype=np.float32)

    def compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute GIST-based distance.

        Args:
            vec1: First LAB vector (will extract L channel)
            vec2: Second LAB vector (will extract L channel)

        Returns:
            Distance score (lower = more similar)
        """
        # Extract L channel and reshape to 2D
        img1 = vec1.reshape(self.image_size, self.image_size, 3)[:, :, 0].astype(np.uint8)
        img2 = vec2.reshape(self.image_size, self.image_size, 3)[:, :, 0].astype(np.uint8)

        # Compute GIST features
        gist1 = self._compute_gist(img1)
        gist2 = self._compute_gist(img2)

        # Euclidean distance
        diff = gist1 - gist2
        distance = np.sum(diff * diff)

        # Scale to match LAB distance range
        return float(distance * 1000.0)

    def compute_batch_distance(
        self, vecs: np.ndarray, query: np.ndarray, _weights: np.ndarray
    ) -> np.ndarray:
        """Compute GIST distances for batch of vectors.

        Args:
            vecs: Batch of image vectors (N x dims)
            query: Query image vector (dims,)
            _weights: Weights (unused, GIST doesn't use weights)

        Returns:
            Array of GIST distances for each vector in batch.
        """
        n = len(vecs)
        distances = np.empty(n, dtype=np.float32)

        # Compute query GIST once
        query_img = query.reshape(self.image_size, self.image_size, 3)[:, :, 0].astype(np.uint8)
        query_gist = self._compute_gist(query_img)

        for i in range(n):
            img = vecs[i].reshape(self.image_size, self.image_size, 3)[:, :, 0].astype(np.uint8)
            gist = self._compute_gist(img)
            diff = gist - query_gist
            distances[i] = np.sum(diff * diff) * 1000.0

        return distances


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
    if cfg.metric == "ssim":
        return SSIMMetric(cfg)
    if cfg.metric == "lab+ssim":
        return HybridMetric(cfg)
    if cfg.metric == "gist":
        return GISTMetric(cfg)
    msg = f"Unknown metric type: {cfg.metric}"
    raise ValueError(msg)
