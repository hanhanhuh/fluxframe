"""Pydantic models for type-safe data structures."""


from pydantic import BaseModel, ConfigDict, Field


class ImageMatch(BaseModel):
    """Single image match with similarity score.

    Attributes:
        image_path: Path to the matched image file.
        similarity: Similarity score between 0.0 and 1.0 (higher is more similar).
    """
    image_path: str
    similarity: float = Field(ge=0.0, le=1.0)


class FrameResult(BaseModel):
    """Matching results for a single video frame.

    Attributes:
        top_matches: List of (image_path, similarity_score) tuples for top N matches.
        selected: Path to the randomly selected image from top matches, or None.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    top_matches: list[tuple[str, float]]
    selected: str | None = None


class VideoInfo(BaseModel):
    """Video file metadata.

    Attributes:
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        total_frames: Total number of frames in video.
    """
    fps: float
    width: int
    height: int
    total_frames: int


class ImageFeatures(BaseModel):
    """Pre-computed image features for a specific aspect ratio.

    Attributes:
        aspect_ratio: The aspect ratio (width/height) these features represent.
        edge: Normalized edge histogram (256 bins).
        texture: Normalized texture gradient histogram (64 bins).
        color: Normalized HSV color histogram (512 bins: 8x8x8).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    aspect_ratio: float
    edge: list[float]
    texture: list[float]
    color: list[float]
