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
    """Main configuration for video generation and frame matching."""

    # Required Settings
    img_dir: Path
    output_dir: Path
    targets: list[RenderTarget] | None = None  # Empty for matching mode

    # Video Generation Settings
    fps: int = 30
    duration: int = 10

    # Algorithm Settings (LAB weighting)
    weights: tuple[float, float, float] = (1.0, 2.0, 2.0)
    threshold: float = 40000.0
    start_filename: str | None = None

    # Smoothing Settings (generation only)
    smoothing_k: int = 1  # 1 = Off (Greedy), 3-5 = Smooth, 10+ = Slow Drift

    # Unique Frame Enforcement
    enforce_unique: bool = True

    # Metric Configuration (shared)
    metric: MetricType = "lab"  # Default to LAB for backward compat
    ssim_weight: float = 0.5  # Weight for hybrid metric (0.0-1.0)

    # Color Grading Configuration (shared)
    # If empty list, no color grading is applied
    # If list contains methods, generates one video per method plus ungraded
    color_grading_methods: list[ColorGradingMethod] | None = None
    color_grading_strength: float = 0.7  # 0.0-1.0

    # Legacy single-method support (deprecated, use color_grading_methods)
    enable_color_grading: bool = False
    color_grading_method: ColorGradingMethod = "histogram"

    # Frame Matching Settings (None = not in matching mode)
    video_path: Path | None = None
    top_n: int = 10  # Number of candidates to consider
    checkpoint_batch_size: int = 10
    fps_override: float | None = None
    demo_mode: bool = False
    demo_seconds: int = 20
    demo_images: int = 1000
    save_samples: int = 0
    sample_interval: int = 1
    seed: int | None = None

    # Internal Constants
    dims_raw: int = 64 * 64 * 3
    dims_pca: int = 128

    # Cache Filenames
    fn_raw: str = "cache_raw_lab.bin"
    fn_names: str = "cache_filenames.json"
    fn_meta: str = "cache_meta.json"
    fn_index: str = "cache_faiss.index"

    def __post_init__(self) -> None:
        """Post-initialization processing.

        Converts None values to empty lists and handles legacy config.
        """
        # Convert targets to empty list if None (for matching mode)
        if self.targets is None:
            self.targets = []

        # Convert color_grading_methods to empty list if None
        if self.color_grading_methods is None:
            # Legacy: if enable_color_grading is True, use single method
            if self.enable_color_grading:
                self.color_grading_methods = [self.color_grading_method]
            else:
                self.color_grading_methods = []

    @property
    def total_frames(self) -> int:
        """Total number of frames to generate.

        Returns:
            fps * duration as integer frame count.
        """
        return self.fps * self.duration

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if self.fps <= 0:
            msg = f"fps must be positive, got {self.fps}"
            raise ValueError(msg)
        if self.duration <= 0:
            msg = f"duration must be positive, got {self.duration}"
            raise ValueError(msg)
        if not 0.0 <= self.ssim_weight <= 1.0:
            msg = f"ssim_weight must be in [0,1], got {self.ssim_weight}"
            raise ValueError(msg)
        if not 0.0 <= self.color_grading_strength <= 1.0:
            msg = f"color_grading_strength must be in [0,1], got {self.color_grading_strength}"
            raise ValueError(msg)
        if self.smoothing_k < 1:
            msg = f"smoothing_k must be >= 1, got {self.smoothing_k}"
            raise ValueError(msg)
