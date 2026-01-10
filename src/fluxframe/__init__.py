"""FluxFrame - Generate videos from image collections using perceptual similarity."""

from .matcher import FeatureDict, FeatureMethod, ImageMatcher, PoolingMethod
from .models import FrameResult, ImageMatch, VideoInfo
from .processor import VideoImageMatcher

# Core generator functionality (now at top level)
from .config import Config, RenderTarget
from .database import ImageDatabase
from .search import SearchIndex
from .pathfinding import PathFinder, find_path
from .rendering import VideoRenderer, render_videos, smart_crop
from .metrics import create_metric
from .color_grading import ColorGrader, create_color_grader

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
    # Video frame matching (alternative input source)
    "FeatureDict",
    "FeatureMethod",
    "FrameResult",
    "ImageMatch",
    "ImageMatcher",
    "PoolingMethod",
    "VideoImageMatcher",
    "VideoInfo",
]
