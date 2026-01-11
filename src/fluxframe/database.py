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
            with open(self.names_path) as f:
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
        """Build database from image directory."""
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

        # Process images in parallel
        results = process_map(
            _worker_load_img, files, chunksize=50, max_workers=None, desc="Indexing images"
        )

        valid_data = [res for res in results if res is not None]
        valid_files = [f.name for f, res in zip(files, results, strict=False) if res is not None]

        if not valid_data:
            sys.exit("Error: Could not read any images")

        # Write to memmap
        shape = (len(valid_data), self.cfg.dims_raw)
        mm = np.memmap(self.raw_path, dtype="uint8", mode="w+", shape=shape)

        batch_size = 5000
        for i in range(0, len(valid_data), batch_size):
            batch = np.array(valid_data[i : i + batch_size], dtype=np.uint8)
            mm[i : i + len(batch)] = batch
            mm.flush()

        # Save filenames list
        with open(self.names_path, "w") as f:
            json.dump(valid_files, f)

        self.filenames = valid_files
        self.data = mm
        print(f"[DB] Complete: {len(valid_files)} images indexed")

    def __len__(self) -> int:
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
            with open(path, "rb") as f:
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
