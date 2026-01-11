#!/usr/bin/env python3
"""Configuration dataclasses for sad_picker package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Type aliases
MetricType = Literal["lab", "ssim", "lab+ssim", "gist"]
ColorGradingMethod = Literal["histogram", "color_transfer", "lut"]


@dataclass
class RenderTarget:
    """Video rendering target specification."""

    width: int
    height: int
    filename: str


@dataclass
class Config:
    """Main configuration for video generation pipeline."""

    # Required Settings
    img_dir: Path
    targets: list[RenderTarget]

    # Video Settings
    output_dir: Path
    fps: int = 30
    duration: int = 10

    # Algorithm Settings (LAB weighting)
    weights: tuple[float, float, float] = (1.0, 2.0, 2.0)
    threshold: float = 40000.0
    start_filename: str | None = None

    # Smoothing Settings
    smoothing_k: int = 1  # 1 = Off (Greedy), 3-5 = Smooth, 10+ = Slow Drift

    # Unique Frame Enforcement
    enforce_unique: bool = True

    # NEW: Metric Configuration
    metric: MetricType = "lab"  # Default to LAB for backward compat
    ssim_weight: float = 0.5  # Weight for hybrid metric (0.0-1.0)

    # NEW: Color Grading Configuration
    enable_color_grading: bool = False
    color_grading_method: ColorGradingMethod = "histogram"
    color_grading_strength: float = 0.7  # 0.0-1.0

    # Internal Constants
    dims_raw: int = 64 * 64 * 3
    dims_pca: int = 128

    # Cache Filenames
    fn_raw: str = "cache_raw_lab.bin"
    fn_names: str = "cache_filenames.json"
    fn_meta: str = "cache_meta.json"
    fn_index: str = "cache_faiss.index"

    @property
    def total_frames(self) -> int:
        """Total number of frames to generate."""
        return self.fps * self.duration

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if not 0.0 <= self.ssim_weight <= 1.0:
            raise ValueError(f"ssim_weight must be in [0,1], got {self.ssim_weight}")
        if not 0.0 <= self.color_grading_strength <= 1.0:
            raise ValueError(
                f"color_grading_strength must be in [0,1], got {self.color_grading_strength}"
            )
        if self.smoothing_k < 1:
            raise ValueError(f"smoothing_k must be >= 1, got {self.smoothing_k}")
