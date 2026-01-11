"""Video frame matching using unified metrics (LAB, SSIM, GIST)."""

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .checkpoint import CheckpointManager
from .config import Config
from .database import ImageDatabase
from .models import FrameResult, VideoInfo
from .search import SearchIndex
from .video import VideoReader

logger = logging.getLogger(__name__)


class VideoFrameMatcher:
    """Match video frames to image dataset using unified metrics."""

    def __init__(self, config: Config):
        """Initialize video frame matcher.

        Args:
            config: Configuration object with matching settings
        """
        if config.video_path is None:
            raise ValueError("Config must have video_path set for frame matching")

        self.cfg = config

        # Set random seed
        if config.seed is not None:
            np.random.seed(config.seed)

        # Initialize video reader
        self.video_reader = VideoReader(
            video_path=config.video_path,
            fps_override=config.fps_override,
            demo_mode=config.demo_mode,
            demo_seconds=config.demo_seconds,
        )

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config.output_dir / "checkpoint.json"
        )

        # Initialize database and search index
        logger.info(f"Loading image database from {config.img_dir}")
        self.db = ImageDatabase(config)

        # Apply demo mode image limit
        if config.demo_mode and len(self.db) > config.demo_images:
            logger.info(f"Demo mode: Using {config.demo_images} of {len(self.db)} images")
            # Randomly sample images
            indices = np.random.choice(len(self.db), config.demo_images, replace=False)
            self.db.filenames = [self.db.filenames[i] for i in indices]
            # Note: Can't slice memmap directly, but we track via filenames

        logger.info(f"Building search index with metric: {config.metric}")
        self.search_index = SearchIndex(config, self.db)

        # Tracking
        self.used_indices: set[int] = set()

    @classmethod
    def from_paths(
        cls,
        video_path: str,
        image_folder: str,
        output_dir: str,
        **kwargs,
    ) -> "VideoFrameMatcher":
        """Legacy constructor for backward compatibility with tests.

        Args:
            video_path: Path to video file
            image_folder: Path to image directory
            output_dir: Output directory path
            **kwargs: Additional config parameters (metric, threshold, top_n, etc.)

        Returns:
            VideoFrameMatcher instance
        """
        # Map old parameter names to new ones
        if "similarity_threshold" in kwargs:
            kwargs["threshold"] = kwargs.pop("similarity_threshold")
        if "no_repeat" in kwargs:
            kwargs["enforce_unique"] = kwargs.pop("no_repeat")

        config = Config(
            img_dir=Path(image_folder),
            output_dir=Path(output_dir),
            video_path=Path(video_path),
            targets=[],  # Not generating
            **kwargs,
        )

        return cls(config)

    # Property accessors for test compatibility
    @property
    def faiss_index(self):
        """Access FAISS index for test compatibility."""
        return self.search_index.index

    @property
    def image_paths(self):
        """Access image paths for test compatibility."""
        return self.db.filenames

    @property
    def vectors(self):
        """Access LAB vectors for test compatibility."""
        return self.db.data

    @property
    def video_info(self) -> VideoInfo:
        """Get video information."""
        return self.video_reader.get_info()

    @property
    def similarity_threshold(self) -> float:
        """Backward compat: map threshold to similarity_threshold."""
        return self.cfg.threshold

    @property
    def no_repeat(self) -> bool:
        """Backward compat: map enforce_unique to no_repeat."""
        return self.cfg.enforce_unique

    @property
    def demo_mode(self) -> bool:
        """Access demo mode setting."""
        return self.cfg.demo_mode

    @property
    def demo_seconds(self) -> int:
        """Access demo seconds setting."""
        return self.cfg.demo_seconds

    @property
    def demo_images(self) -> int:
        """Access demo images setting."""
        return self.cfg.demo_images

    @property
    def fps_override(self) -> float | None:
        """Access FPS override setting."""
        return self.cfg.fps_override

    @property
    def input_fps(self) -> float:
        """Get input video FPS."""
        return self.video_info.fps

    @property
    def input_fps_skip_interval(self) -> int:
        """Calculate skip interval based on FPS override."""
        if self.cfg.fps_override is not None:
            return max(1, round(self.input_fps / self.cfg.fps_override))
        return 1

    # Test compatibility methods
    def get_image_files(self) -> list[Path]:
        """Get list of image file paths.

        For test compatibility - returns Path objects from filenames.

        Returns:
            List of image file paths
        """
        return [self.cfg.img_dir / name for name in self.db.filenames]

    def get_video_info(self) -> VideoInfo:
        """Get video information.

        Returns:
            VideoInfo object
        """
        return self.video_info

    def _build_faiss_index(self, image_files: list[Path]) -> None:
        """Build FAISS index (for test compatibility).

        The actual index is already built by SearchIndex in __init__.
        This method exists for test compatibility only.

        Args:
            image_files: List of image files (ignored, uses existing index)
        """
        # Index is already built - this is a no-op for compatibility
        pass

    def _generate_cache_key(self, image_files: list[Path]) -> str:
        """Generate cache key for test compatibility.

        Args:
            image_files: List of image files

        Returns:
            Cache key string (deterministic hash)
        """
        import hashlib

        # Create deterministic hash from config params
        key_data = (
            str(sorted(str(f) for f in image_files))
            + str(self.cfg.weights)
            + str(self.cfg.metric)
        )
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _validate_cache(self, image_files: list[Path]) -> bool:
        """Validate cache exists for test compatibility.

        Args:
            image_files: List of image files

        Returns:
            True if cache is valid
        """
        # Check if FAISS index exists
        return (self.cfg.img_dir / self.cfg.fn_index).exists()

    @property
    def cache_metadata_path(self) -> Path:
        """Get cache metadata path for test compatibility."""
        return self.cfg.img_dir / self.cfg.fn_meta

    @property
    def faiss_index_path(self) -> Path:
        """Get FAISS index path for test compatibility."""
        return self.cfg.img_dir / self.cfg.fn_index

    @property
    def vectors_path(self) -> Path:
        """Get vectors path for test compatibility."""
        return self.cfg.img_dir / self.cfg.fn_raw

    def find_top_matches(
        self, frame: np.ndarray, frame_num: int, aspect_ratio: float
    ) -> list[tuple[str, float]]:
        """Find top N matches for a frame.

        Args:
            frame: BGR frame from video
            frame_num: Frame number (unused, for compat)
            aspect_ratio: Aspect ratio (unused, for compat)

        Returns:
            List of (image_path, similarity_score) tuples
        """
        # Convert frame to LAB
        frame_lab = VideoReader.frame_to_lab(frame)

        # Search for nearest neighbors
        k = self.cfg.top_n + len(self.used_indices) if self.cfg.enforce_unique else self.cfg.top_n
        distances, indices = self.search_index.search_vector(frame_lab, k_candidates=k)

        # Convert to (path, score) tuples
        results = []
        for dist, idx in zip(distances, indices):
            if idx < len(self.db.filenames):
                img_path = str(self.cfg.img_dir / self.db.filenames[idx])
                # Convert distance to similarity score (inverse)
                similarity = 1.0 / (1.0 + float(dist))
                results.append((img_path, similarity))

        return results[:self.cfg.top_n]

    def select_match(self, top_matches: list[tuple[str, float]]) -> str | None:
        """Select best match from candidates.

        Args:
            top_matches: List of (image_path, similarity_score) tuples

        Returns:
            Selected image path or None if no matches
        """
        if not top_matches:
            return None

        # If enforce_unique, pick best unused match
        if self.cfg.enforce_unique:
            for img_path, score in top_matches:
                # Convert path to index
                img_name = Path(img_path).name
                if img_name in self.db.filenames:
                    idx = self.db.filenames.index(img_name)
                    if idx not in self.used_indices:
                        return img_path

            # All used - return best anyway
            return top_matches[0][0]
        else:
            # Return best match
            return top_matches[0][0]

    def save_checkpoint(self, checkpoint: dict) -> None:
        """Save checkpoint to disk.

        Args:
            checkpoint: Checkpoint dictionary
        """
        self.checkpoint_manager.save(checkpoint)

    def load_checkpoint(self) -> dict:
        """Load checkpoint from disk.

        Returns:
            Checkpoint dictionary or empty dict
        """
        checkpoint = self.checkpoint_manager.load()
        return checkpoint if checkpoint is not None else {"frames": []}

    def process(self) -> dict:
        """Process video and match frames to images.

        Returns:
            Checkpoint dictionary with results
        """
        # Load checkpoint if exists
        checkpoint = self.load_checkpoint()
        start_frame = len(checkpoint.get("frames", []))

        if start_frame > 0:
            logger.info(f"Resuming from frame {start_frame}")

        # Open video
        cap = self.video_reader.open()

        # Skip to start frame
        if start_frame > 0:
            self.video_reader.skip_to(cap, start_frame)

        try:
            for frame_idx in tqdm(
                range(start_frame, self.video_info.total_frames),
                desc="Processing frames",
                initial=start_frame,
                total=self.video_info.total_frames,
            ):
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to LAB
                frame_lab = VideoReader.frame_to_lab(frame)

                # Search for nearest neighbors
                k = self.cfg.top_n + len(self.used_indices) if self.cfg.enforce_unique else self.cfg.top_n
                distances, indices = self.search_index.search_vector(frame_lab, k_candidates=k)

                # Select best match
                selected_idx = self._select_match(indices, distances)

                # Convert distance to similarity
                similarity = 1.0 / (1.0 + float(distances[0]))

                # Save result
                checkpoint["frames"].append({
                    "frame_number": frame_idx,
                    "selected_image": self.db.filenames[selected_idx],
                    "similarity_score": similarity,
                })

                # Mark as used
                if self.cfg.enforce_unique:
                    self.used_indices.add(selected_idx)

                # Save checkpoint
                if (frame_idx + 1) % self.cfg.checkpoint_batch_size == 0:
                    self.save_checkpoint(checkpoint)

        finally:
            cap.release()

        # Final checkpoint save
        self.save_checkpoint(checkpoint)

        return checkpoint

    def _select_match(self, indices: np.ndarray, distances: np.ndarray) -> int:
        """Select best match from candidates (internal method).

        Args:
            indices: Array of candidate indices
            distances: Array of distances

        Returns:
            Selected image index
        """
        if self.cfg.enforce_unique:
            # Filter out used indices
            valid_mask = np.array([idx not in self.used_indices for idx in indices])
            if not valid_mask.any():
                # All used - pick best even if used
                return int(indices[0])

            valid_indices = indices[valid_mask]
            # Return best unused
            return int(valid_indices[0])
        else:
            # Return best match
            return int(indices[0])

    def generate_output(self, checkpoint: dict) -> None:
        """Generate output video from checkpoint.

        Args:
            checkpoint: Checkpoint dictionary with matched frames
        """
        logger.info("Generating output video...")

        # Get video properties
        output_path = self.cfg.output_dir / f"{self.cfg.video_path.stem}_matched.mp4"

        # Create output directory for matched frames
        output_images_dir = self.cfg.output_dir / "matched_frames"
        output_images_dir.mkdir(parents=True, exist_ok=True)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.video_info.fps,
            (self.video_info.width, self.video_info.height),
        )

        try:
            for frame_data in tqdm(checkpoint["frames"], desc="Writing video"):
                img_path = self.cfg.img_dir / frame_data["selected_image"]
                img = cv2.imread(str(img_path))

                if img is not None:
                    # Resize to match video dimensions
                    img_resized = cv2.resize(
                        img,
                        (self.video_info.width, self.video_info.height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    out.write(img_resized)

                    # Save individual frame
                    frame_num = frame_data["frame_number"]
                    frame_img_path = output_images_dir / f"frame_{frame_num:06d}.jpg"
                    cv2.imwrite(str(frame_img_path), img_resized)

        finally:
            out.release()

        logger.info(f"Output video saved to {output_path}")
