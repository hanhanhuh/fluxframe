"""FluxFrame - Video frame matching and smooth transition generation."""

from .matcher import FeatureDict, FeatureMethod, ImageMatcher, PoolingMethod
from .models import FrameResult, ImageMatch, VideoInfo
from .processor import VideoImageMatcher

# Generator submodule for video generation from image collections
from .generator import (
    Config,
    RenderTarget,
    ImageDatabase,
    SearchIndex,
    PathFinder,
    find_path,
    VideoRenderer,
    render_videos,
    smart_crop,
    create_metric,
    ColorGrader,
    create_color_grader,
)

__version__ = "0.2.0"

__all__ = [
    # Matcher (original functionality)
    "FeatureDict",
    "FeatureMethod",
    "FrameResult",
    "ImageMatch",
    "ImageMatcher",
    "PoolingMethod",
    "VideoImageMatcher",
    "VideoInfo",
    # Generator (new functionality)
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
]
