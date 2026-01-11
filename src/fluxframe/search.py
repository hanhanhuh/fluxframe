#!/usr/bin/env python3
"""Search index module with FAISS and re-ranking capabilities."""

from __future__ import annotations

import json

import faiss
import numpy as np
from tqdm import tqdm

from .config import Config
from .database import ImageDatabase
from .metrics import create_metric

# Try to import Numba for JIT acceleration
try:
    from numba import jit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


class SearchIndex:
    """Hybrid search index using FAISS for coarse search + metric-based re-ranking."""

    def __init__(self, cfg: Config, db: ImageDatabase):
        """Initialize search index.

        Args:
            cfg: Configuration object
            db: Image database
        """
        self.cfg = cfg
        self.db = db
        self.index_path = cfg.img_dir / cfg.fn_index
        self.meta_path = cfg.img_dir / cfg.fn_meta
        self.index: faiss.Index | None = None

        # Distance metric for re-ranking
        self.metric = create_metric(cfg)

        # Pre-computed weight tile for LAB channels
        self.w_tile = np.tile(cfg.weights, 64 * 64).astype(np.float32)

        self._load_or_train()

    def _load_or_train(self) -> None:
        """Load existing index or train new one."""
        rebuild = True

        if self.index_path.exists() and self.meta_path.exists():
            try:
                with self.meta_path.open() as f:
                    meta = json.load(f)

                # Check if weights match (FAISS index depends on weights)
                cached_weights = meta.get("weights", [])
                if len(cached_weights) == 3 and np.allclose(cached_weights, self.cfg.weights):
                    rebuild = False
                    print(f"[Index] Using existing FAISS index (K={self.cfg.smoothing_k}).")
                else:
                    print("[Index] Weights changed. Re-indexing required.")
            except Exception:
                print("[Index] Metadata corrupted. Re-indexing.")

        if rebuild:
            self._train_index()
        else:
            self.index = faiss.read_index(str(self.index_path))

    def _train_index(self) -> None:
        """Train PCA and FAISS index."""
        print("[Index] Training PCA & Index...")

        if HAS_NUMBA:
            print("[Index] Numba JIT detected - using accelerated re-ranking")

        mat = faiss.PCAMatrix(self.cfg.dims_raw, self.cfg.dims_pca)
        n_samples = len(self.db.filenames)

        # Choose index type based on dataset size
        if n_samples > 100000:
            quantizer = faiss.IndexFlatL2(self.cfg.dims_pca)
            idx_ivf = faiss.IndexIVFFlat(quantizer, self.cfg.dims_pca, min(4096, n_samples // 100))
            self.index = faiss.IndexPreTransform(mat, idx_ivf)
        else:
            idx_flat = faiss.IndexFlatL2(self.cfg.dims_pca)
            self.index = faiss.IndexPreTransform(mat, idx_flat)

        # Training on subset
        train_size = min(n_samples, 4096)
        indices = np.linspace(0, n_samples - 1, train_size, dtype=int)

        print(f"[Index] Training on {train_size} samples...")
        train_data = self.db.data[indices].astype(np.float32) * self.w_tile  # type: ignore
        self.index.train(train_data)

        # Add all images to index
        print("[Index] Adding all images...")
        chunk_size = 20000
        for i in tqdm(range(0, n_samples, chunk_size)):
            end = min(i + chunk_size, n_samples)
            batch = self.db.data[i:end].astype(np.float32) * self.w_tile  # type: ignore
            self.index.add(batch)

        # Save index and metadata
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("w") as f:
            json.dump(
                {"weights": list(self.cfg.weights), "metric": self.cfg.metric}, f
            )

    def search_vector(
        self, query_vec: np.ndarray, k_candidates: int = 200
    ) -> tuple[np.ndarray, np.ndarray]:
        """Core search function based on arbitrary vector.

        Args:
            query_vec: float32 array with shape (dims_raw,)
            k_candidates: Number of candidates to retrieve

        Returns:
            Tuple of (distances, indices) sorted by distance
        """
        # 1. Apply LAB weights to query vector (for PCA search)
        query_weighted = query_vec * self.w_tile

        # 2. FAISS coarse search
        _dists_pca, inds = self.index.search(query_weighted.reshape(1, -1), k_candidates)  # type: ignore
        candidates_indices = inds[0]

        # Filter invalid (-1) indices
        valid_mask = candidates_indices != -1
        candidates_indices = candidates_indices[valid_mask]

        if len(candidates_indices) == 0:
            return np.array([]), np.array([])

        # 3. Re-ranking with selected metric
        cands_raw = self.db.data[candidates_indices].astype(np.float32)  # type: ignore

        # Use optimized batch distance computation
        if HAS_NUMBA and self.cfg.metric == "lab":
            # Fast path for LAB metric with Numba JIT
            dists_sq = _batch_lab_distance_jit(cands_raw, query_vec, self.w_tile)
        else:
            # Use metric's batch computation
            dists_sq = self.metric.compute_batch_distance(cands_raw, query_vec, self.w_tile)

        # Sort by distance
        sort_order = np.argsort(dists_sq)
        return dists_sq[sort_order], candidates_indices[sort_order]

    def search_id(
        self, idx: int, k_candidates: int = 200
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search based on existing image ID.

        Args:
            idx: Database index of query image
            k_candidates: Number of candidates to retrieve

        Returns:
            Tuple of (distances, indices) sorted by distance
        """
        query_vec = self.db.data[idx].astype(np.float32)  # type: ignore
        return self.search_vector(query_vec, k_candidates)


# Numba-accelerated LAB distance computation
if HAS_NUMBA:

    @jit(nopython=True, parallel=True, cache=True)
    def _batch_lab_distance_jit(
        candidates: np.ndarray, query: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """JIT-compiled batch LAB distance computation.

        Args:
            candidates: Array of candidate vectors (N, dims)
            query: Query vector (dims,)
            weights: Weight vector (dims,)

        Returns:
            Array of squared distances (N,)
        """
        n = candidates.shape[0]
        distances = np.empty(n, dtype=np.float32)

        for i in prange(n):  # Parallel loop
            dist_sq = 0.0
            for j in range(candidates.shape[1]):
                diff = (candidates[i, j] - query[j]) * weights[j]
                dist_sq += diff * diff
            distances[i] = dist_sq

        return distances

else:
    # Fallback if Numba not available
    def _batch_lab_distance_jit(
        candidates: np.ndarray, query: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Fallback batch LAB distance computation."""
        diff = (candidates - query) * weights
        return np.einsum("ij,ij->i", diff, diff)
