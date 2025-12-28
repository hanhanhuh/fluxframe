"""Video Image Matcher - Match video frames to similar images from datasets."""

from .matcher import ImageMatcher
from .models import FrameResult, ImageFeatures, ImageMatch, VideoInfo
from .processor import VideoImageMatcher

__version__ = "0.1.0"

__all__ = [
    "FrameResult",
    "ImageFeatures",
    "ImageMatch",
    "ImageMatcher",
    "VideoImageMatcher",
    "VideoInfo",
]
