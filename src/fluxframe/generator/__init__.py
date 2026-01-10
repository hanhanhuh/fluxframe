#!/usr/bin/env python3
"""Video generation from image collections using perceptual similarity."""

from __future__ import annotations

__version__ = "2.0.0"

# Core components
from .config import Config, RenderTarget
from .database import ImageDatabase
from .pathfinding import PathFinder, find_path
from .rendering import VideoRenderer, render_videos, smart_crop
from .search import SearchIndex

# Metrics
from .metrics import (
    DistanceMetric,
    HybridMetric,
    LABWeightedDistance,
    SSIMMetric,
    create_metric,
)

# Color grading
from .color_grading import ColorGrader, create_color_grader

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Config
    "Config",
    "RenderTarget",
    # Core components
    "ImageDatabase",
    "SearchIndex",
    "PathFinder",
    "find_path",
    "VideoRenderer",
    "render_videos",
    "smart_crop",
    # Metrics
    "DistanceMetric",
    "LABWeightedDistance",
    "SSIMMetric",
    "HybridMetric",
    "create_metric",
    # Color grading
    "ColorGrader",
    "create_color_grader",
]
