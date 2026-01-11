"""FluxFrame - Generate videos from image collections using perceptual similarity."""

# Core generator functionality
from .config import Config, RenderTarget
from .database import ImageDatabase
from .search import SearchIndex
from .pathfinding import PathFinder, find_path
from .rendering import VideoRenderer, render_videos, smart_crop
from .metrics import create_metric
from .color_grading import ColorGrader, create_color_grader

# Video frame matching
from .frame_matching import VideoFrameMatcher
from .models import FrameResult, VideoInfo

__version__ = "0.2.0"

__all__ = [
    # Main functionality - Video generation from image collections
    "Config",
    "RenderTarget",
    "ImageDatabase",
    "SearchIndex",
    "PathFinder",
    "find_path",
    "VideoRenderer",
    "render_videos",
    "smart_crop",
    "create_metric",
    "ColorGrader",
    "create_color_grader",
    # Video frame matching (uses same metrics)
    "VideoFrameMatcher",
    "FrameResult",
    "VideoInfo",
]
