"""FluxFrame - Generate videos from image collections using perceptual similarity."""

# Core generator functionality
from .checkpoint import CheckpointManager
from .color_grading import ColorGrader, create_color_grader
from .config import Config, RenderTarget
from .database import ImageDatabase

# Video frame matching
from .frame_matching import VideoFrameMatcher
from .metrics import create_metric
from .models import VideoInfo
from .pathfinding import PathFinder, find_path
from .rendering import VideoRenderer, render_videos, smart_crop
from .search import SearchIndex
from .video import VideoReader

__version__ = "0.2.0"

__all__ = [
    "CheckpointManager",
    "ColorGrader",
    "Config",
    "ImageDatabase",
    "PathFinder",
    "RenderTarget",
    "SearchIndex",
    "VideoFrameMatcher",
    "VideoInfo",
    "VideoReader",
    "VideoRenderer",
    "create_color_grader",
    "create_metric",
    "find_path",
    "render_videos",
    "smart_crop",
]
