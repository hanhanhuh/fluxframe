"""Video Image Matcher - Match video frames to similar images from datasets."""

from .matcher import FeatureDict, ImageMatcher
from .models import FrameResult, ImageMatch, VideoInfo
from .processor import VideoImageMatcher

__version__ = "0.1.0"

__all__ = [
    "FeatureDict",
    "FrameResult",
    "ImageMatch",
    "ImageMatcher",
    "VideoImageMatcher",
    "VideoInfo",
]
