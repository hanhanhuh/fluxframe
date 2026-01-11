#!/usr/bin/env python3
"""Image database module with memmap caching and optional TurboJPEG acceleration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm.contrib.concurrent import process_map

from .config import Config

# Try to import TurboJPEG for 3x faster JPEG loading
try:
    from turbojpeg import TurboJPEG

    HAS_TURBOJPEG = True
    _jpeg_decoder = TurboJPEG()
except ImportError:
    HAS_TURBOJPEG = False
    _jpeg_decoder = None


class ImageDatabase:
    """Manages access to image data via memory-mapped cache files."""

    def __init__(self, cfg: Config):
        """Initialize database with configuration.

        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.raw_path = cfg.img_dir / cfg.fn_raw
        self.names_path = cfg.img_dir / cfg.fn_names

        self.filenames: list[str] = []
        self.data: np.memmap | None = None

        self._load_or_build()

    def _load_or_build(self) -> None:
        """Load existing cache or build new database."""
        # Check if cache exists
        if self.raw_path.exists() and self.names_path.exists():
            print(f"[DB] Using existing cache: {self.raw_path.name}")
            with self.names_path.open() as f:
                self.filenames = json.load(f)

            # Verify cache file size matches expected size
            expected_bytes = len(self.filenames) * self.cfg.dims_raw
            actual_bytes = self.raw_path.stat().st_size

            if expected_bytes != actual_bytes:
                print(
                    f"[DB] Warning: Cache size mismatch (Expected: {expected_bytes}, "
                    f"Got: {actual_bytes}). Rebuilding..."
                )
                self._build_database()
            else:
                shape = (len(self.filenames), self.cfg.dims_raw)
                self.data = np.memmap(self.raw_path, dtype="uint8", mode="r", shape=shape)
        else:
            self._build_database()

    def _build_database(self) -> None:
        """Build database from image directory using chunked processing for memory efficiency."""
        print(f"[DB] Building cache from {self.cfg.img_dir} (one-time operation)...")

        if HAS_TURBOJPEG:
            print("[DB] TurboJPEG detected - using accelerated JPEG loading")

        # Find image files
        files = sorted(
            [
                f
                for f in self.cfg.img_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
        )
        if not files:
            sys.exit("Error: No images found in directory")

        # Pre-allocate memmap for worst case (all images valid)
        # This will be trimmed later if some images are invalid
        temp_mm = np.memmap(
            self.raw_path, dtype="uint8", mode="w+", shape=(len(files), self.cfg.dims_raw)
        )

        valid_files: list[str] = []
        valid_count = 0
        chunk_size = 300000  # ~3.6 GB per chunk for 64x64x3 images

        # Process images in chunks to limit memory usage
        num_chunks = (len(files) + chunk_size - 1) // chunk_size
        for chunk_idx in range(0, len(files), chunk_size):
            chunk_num = chunk_idx // chunk_size + 1
            chunk_files = files[chunk_idx : chunk_idx + chunk_size]

            # Process this chunk in parallel
            chunk_results = process_map(
                _worker_load_img,
                chunk_files,
                chunksize=50,
                max_workers=None,
                desc=f"Chunk {chunk_num}/{num_chunks}",
            )

            # Write valid results to memmap
            for fname, data in zip(chunk_files, chunk_results, strict=False):
                if data is not None:
                    temp_mm[valid_count] = data
                    valid_files.append(fname.name)
                    valid_count += 1

            temp_mm.flush()
            del chunk_results  # Free memory immediately

        if valid_count == 0:
            sys.exit("Error: Could not read any images")

        # Trim memmap to actual size if some images were invalid
        if valid_count < len(files):
            temp_mm.flush()
            del temp_mm
            temp_mm = np.memmap(
                self.raw_path, dtype="uint8", mode="r+", shape=(valid_count, self.cfg.dims_raw)
            )

        # Save filenames list
        with self.names_path.open("w") as f:
            json.dump(valid_files, f)

        self.filenames = valid_files
        self.data = temp_mm
        print(f"[DB] Complete: {len(valid_files)} images indexed")

    def __len__(self) -> int:
        """Get number of images in database.

        Returns:
            Number of images.
        """
        """Return number of images in database."""
        return len(self.filenames)


def _worker_load_img(path: Path) -> np.ndarray | None:
    """Worker function for parallel image loading.

    Args:
        path: Path to image file

    Returns:
        Flattened LAB vector (64x64x3) or None if loading failed
    """
    try:
        # Try TurboJPEG first for JPEG files (3x faster)
        if HAS_TURBOJPEG and path.suffix.lower() in {".jpg", ".jpeg"}:
            with path.open("rb") as f:
                img_data = f.read()
            img = _jpeg_decoder.decode(img_data, pixel_format=0)  # BGR format
        else:
            # Fallback to OpenCV
            img = cv2.imread(str(path))

        if img is None:
            return None

        # Resize to 64x64 and convert to LAB
        small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(small, cv2.COLOR_BGR2LAB).flatten()

    except Exception:
        return None
