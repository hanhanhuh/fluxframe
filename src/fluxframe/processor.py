"""Video frame to image matching processor."""

import json
import logging
import pickle
import random
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .matcher import FeatureMethod, ImageMatcher
from .models import FrameResult, VideoInfo

logger = logging.getLogger(__name__)


def _compute_similarity_worker(  # noqa: PLR0913
    img_path_and_features: tuple[str, dict[float, dict[str, npt.NDArray[Any]]]],
    frame_features: dict[str, npt.NDArray[Any]],
    closest_ratio: float,
    edge_weight: float,
    texture_weight: float,
    color_weight: float
) -> tuple[str, float]:
    """Worker function for parallel similarity computation.

    Args:
        img_path_and_features: Tuple of (image_path, features_dict).
        frame_features: Pre-computed frame features.
        closest_ratio: Closest aspect ratio to use from cached features.
        edge_weight: Weight for edge similarity.
        texture_weight: Weight for texture similarity.
        color_weight: Weight for color similarity.

    Returns:
        Tuple of (image_path, similarity_score).
    """
    img_path, img_features_dict = img_path_and_features
    img_features = img_features_dict[closest_ratio]

    # Compute similarities using histogram correlation
    edge_sim = cv2.compareHist(
        frame_features["edge"], img_features["edge"], cv2.HISTCMP_CORREL
    )
    edge_sim = (edge_sim + 1) / 2  # Normalize to 0-1

    texture_sim = cv2.compareHist(
        frame_features["texture"], img_features["texture"], cv2.HISTCMP_CORREL
    )
    texture_sim = (texture_sim + 1) / 2

    color_sim = cv2.compareHist(
        frame_features["color"], img_features["color"], cv2.HISTCMP_CORREL
    )
    color_sim = (color_sim + 1) / 2

    # Weighted combination
    total_sim = edge_weight * edge_sim + texture_weight * texture_sim + color_weight * color_sim

    return (img_path, total_sim)


class VideoImageMatcher:
    """Main class for matching video frames to image dataset."""

    def __init__(  # noqa: PLR0913
        self, video_path: str, image_folder: str, output_dir: str,
        top_n: int = 10, edge_weight: float = 0.33,
        texture_weight: float = 0.33, color_weight: float = 0.34,
        similarity_threshold: float = 0.0, no_repeat: bool = False,
        comparison_size: int = 256, demo_mode: bool = False,
        demo_seconds: int = 20, demo_images: int = 1000,
        checkpoint_batch_size: int = 10, seed: int | None = None,
        num_workers: int | None = None, fps_override: float | None = None,
        feature_method: FeatureMethod = "canny"
    ):
        """
        Initialize the video-image matcher.

        Args:
            video_path: Path to input video file
            image_folder: Path to folder containing images
            output_dir: Directory for outputs and checkpoints
            top_n: Number of top similar images to consider
            edge_weight: Weight for edge similarity
            texture_weight: Weight for texture similarity
            color_weight: Weight for color similarity
            similarity_threshold: Minimum similarity for selection
            no_repeat: If True, use each image only once
            comparison_size: Resize images to this size for comparison (speed optimization)
            demo_mode: If True, use only subset of video and images
            demo_seconds: Number of seconds to process in demo mode
            demo_images: Number of images to use in demo mode
            checkpoint_batch_size: Save checkpoint every N frames
            seed: Random seed for reproducibility
            num_workers: Number of parallel workers (None = auto-detect CPU count)
            fps_override: Override video FPS for output generation (None = use video FPS)
            feature_method: Edge/structure feature extraction method
                - "canny": Fast, no spatial info (original)
                - "spatial_pyramid": 4x4 grid, preserves layout
                - "hog": Best motion preservation
        """
        self.video_path = Path(video_path)
        self.image_folder = Path(image_folder)
        self.output_dir = Path(output_dir)
        self.top_n = top_n
        self.similarity_threshold = similarity_threshold
        self.no_repeat = no_repeat
        self.comparison_size = comparison_size
        self.demo_mode = demo_mode
        self.demo_seconds = demo_seconds
        self.demo_images = demo_images
        self.checkpoint_batch_size = checkpoint_batch_size
        self.num_workers = num_workers if num_workers is not None else cpu_count()
        self.fps_override = fps_override

        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.matcher = ImageMatcher(edge_weight, texture_weight, color_weight, feature_method)
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.results_path = self.output_dir / "results.json"
        self.features_cache_path = self.output_dir / "image_features.pkl"

        self.used_images: set[str] = set()
        self.results: dict[str, Any] = {}
        # Cache for pre-computed features per image and aspect ratio
        self.image_features: dict[str, dict[float, dict[str, npt.NDArray[Any]]]] = {}

    def load_checkpoint(self) -> dict[str, Any]:
        """Load checkpoint from disk if it exists.

        Returns:
            Dictionary mapping frame keys to FrameResult data, or empty dict if no checkpoint.
        """
        if self.checkpoint_path.exists():
            with self.checkpoint_path.open() as f:
                return json.load(f)  # type: ignore[no-any-return]
        return {}

    def save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save checkpoint to disk as JSON.

        Args:
            checkpoint: Dictionary mapping frame keys to FrameResult data.
        """
        with self.checkpoint_path.open("w") as f:
            json.dump(checkpoint, f, indent=2)

    def save_results(self) -> None:
        """Save final results to disk as JSON."""
        with self.results_path.open("w") as f:
            json.dump(self.results, f, indent=2)

    def get_image_files(self) -> list[Path]:
        """Get all image files from the image folder.

        Returns:
            Sorted list of image file paths (limited by demo_images if demo mode enabled).
        """
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files: list[Path] = []

        for ext in extensions:
            image_files.extend(self.image_folder.glob(f"*{ext}"))
            image_files.extend(self.image_folder.glob(f"*{ext.upper()}"))

        image_files = sorted(image_files)

        # Apply demo mode limit
        if self.demo_mode and len(image_files) > self.demo_images:
            logger.info(f"Demo mode: Using first {self.demo_images} images "
                       f"out of {len(image_files)}")
            image_files = image_files[:self.demo_images]

        return image_files

    def precompute_image_features(
        self, image_files: list[Path], force_recompute: bool = False
    ) -> None:
        """Pre-compute and cache feature vectors for all images.

        Caches features for multiple aspect ratios (16:9, 4:3, 1:1, 9:16, 3:4) so the
        cache can be reused across videos with different aspect ratios.

        Args:
            image_files: List of image file paths to process.
            force_recompute: If True, recompute even if valid cache exists.
        """
        # Try to load existing cache
        if not force_recompute and self.features_cache_path.exists():
            logger.info(f"Loading pre-computed features from {self.features_cache_path}")
            try:
                with self.features_cache_path.open("rb") as f:
                    cached_data = pickle.load(f)

                # Verify cache contains all needed images and parameters match
                if cached_data.get("comparison_size") == self.comparison_size:
                    cached_paths = set(cached_data["features"].keys())
                    needed_paths = {str(p) for p in image_files}

                    if needed_paths.issubset(cached_paths):
                        logger.info(f"Cache valid! Loaded features for "
                                   f"{len(needed_paths)} images")
                        self.image_features = {k: v for k, v in cached_data["features"].items()
                                              if k in needed_paths}
                        return
                    missing = len(needed_paths - cached_paths)
                    logger.info(f"Cache incomplete ({missing} images missing), "
                               f"recomputing...")
                else:
                    logger.info("Cache parameters don't match, recomputing...")
            except Exception as e:
                logger.warning(f"Error loading cache: {e}, recomputing...")

        # Compute features for all images
        # Store features for common aspect ratios to avoid recomputation
        logger.info(f"Pre-computing features for {len(image_files)} images...")
        logger.info(f"  Comparison size: {self.comparison_size}px")
        logger.info("  Caching multiple aspect ratios for reusability")
        self.image_features = {}

        # Common aspect ratios: 16:9, 4:3, 1:1, 9:16, 3:4
        aspect_ratios = [16/9, 4/3, 1.0, 9/16, 3/4]

        for img_path in tqdm(image_files, desc="Computing image features"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Store features for multiple aspect ratios
            img_features = {}
            for aspect_ratio in aspect_ratios:
                # Crop to aspect ratio and resize
                img_cropped = self.matcher.aspect_ratio_crop(img, aspect_ratio)
                img_small = cv2.resize(img_cropped,
                                      (self.comparison_size,
                                       int(self.comparison_size / aspect_ratio)))

                # Compute and store features
                features = self.matcher.compute_all_features(img_small)
                img_features[aspect_ratio] = features

            self.image_features[str(img_path)] = img_features

        # Save cache with metadata
        cache_data = {
            "comparison_size": self.comparison_size,
            "aspect_ratios": aspect_ratios,
            "features": self.image_features
        }

        logger.info(f"Saving features cache to {self.features_cache_path}")
        with self.features_cache_path.open("wb") as f:
            pickle.dump(cache_data, f)

        logger.info(f"Pre-computation complete! Cached {len(self.image_features)} "
                   f"feature vectors")

    def get_video_info(self) -> VideoInfo:
        """Get video metadata without loading frames into memory.

        Returns:
            VideoInfo object with fps, width, height, and total_frames.

        Raises:
            ValueError: If video file cannot be opened.
        """
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            msg = f"Cannot open video: {self.video_path}"
            raise ValueError(msg)

        info = VideoInfo(
            fps=cap.get(cv2.CAP_PROP_FPS),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )

        cap.release()
        return info

    def find_top_matches(
        self, frame: npt.NDArray[Any], frame_num: int,  # noqa: ARG002
        target_aspect_ratio: float
    ) -> list[tuple[str, float]]:
        """Find top N matches for a frame using pre-computed features.

        Uses parallel processing to speed up similarity computation.

        Args:
            frame: Video frame to match (BGR image).
            frame_num: Frame number (currently unused, kept for API compatibility).
            target_aspect_ratio: Target aspect ratio as width/height.

        Returns:
            List of (image_path, similarity_score) tuples sorted by similarity.
        """
        # Prepare frame for comparison
        frame_small = cv2.resize(frame, (self.comparison_size,
                                         int(self.comparison_size / target_aspect_ratio)))

        # Compute frame features
        frame_features = self.matcher.compute_all_features(frame_small)

        # Find closest pre-computed aspect ratio
        cached_ratios = list(next(iter(self.image_features.values())).keys())
        closest_ratio = min(cached_ratios, key=lambda r: abs(r - target_aspect_ratio))

        # Compare against all pre-computed image features using parallel processing
        worker_func = partial(
            _compute_similarity_worker,
            frame_features=frame_features,
            closest_ratio=closest_ratio,
            edge_weight=self.matcher.edge_weight,
            texture_weight=self.matcher.texture_weight,
            color_weight=self.matcher.color_weight
        )

        with Pool(processes=self.num_workers) as pool:
            similarities = pool.map(worker_func, self.image_features.items())

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.top_n]

    def select_random_match(self, top_matches: list[tuple[str, float]]) -> str | None:
        """Select a random match from top matches.

        Filters by similarity threshold and no-repeat constraint before random selection.

        Args:
            top_matches: List of (image_path, similarity_score) tuples.

        Returns:
            Selected image path or None if no valid match after filtering.
        """
        # Filter by threshold
        valid_matches = [(path, score) for path, score in top_matches
                        if score >= self.similarity_threshold]

        if not valid_matches:
            return None

        # Filter out used images if no_repeat is enabled
        if self.no_repeat:
            valid_matches = [(path, score) for path, score in valid_matches
                           if path not in self.used_images]

        if not valid_matches:
            return None

        # Random selection
        selected_path = random.choice(valid_matches)[0]

        if self.no_repeat:
            self.used_images.add(selected_path)

        return selected_path

    def process(self) -> dict[str, Any]:  # noqa: PLR0915
        """Main processing pipeline with frame-by-frame processing and batched checkpoints.

        Returns:
            dict: Checkpoint data with results for all processed frames.

        Raises:
            ValueError: If video cannot be opened or no images found in folder.
        """
        logger.info(f"Video: {self.video_path}")
        logger.info(f"Image folder: {self.image_folder}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Top N: {self.top_n}")
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
        logger.info(f"No repeat: {self.no_repeat}")
        logger.info(f"Checkpoint batch size: {self.checkpoint_batch_size}")
        logger.info(f"Parallel workers: {self.num_workers}")
        logger.info(f"Feature method: {self.matcher.feature_method}")
        if self.demo_mode:
            logger.info(f"Demo mode: ON (max {self.demo_seconds}s video, "
                       f"{self.demo_images} images)")
        logger.info(f"Weights - Edge: {self.matcher.edge_weight:.2f}, "
                   f"Texture: {self.matcher.texture_weight:.2f}, "
                   f"Color: {self.matcher.color_weight:.2f}")

        # Get video info
        video_info = self.get_video_info()
        self.fps = video_info.fps
        self.frame_width = video_info.width
        self.frame_height = video_info.height
        total_frames = video_info.total_frames

        # Calculate frames to process
        if self.demo_mode:
            max_frames = int(self.fps * self.demo_seconds)
            frames_to_process = min(max_frames, total_frames)
            logger.info(f"Demo mode: Processing first {self.demo_seconds} seconds "
                       f"({frames_to_process} frames) out of {total_frames}")
        else:
            frames_to_process = total_frames

        # Load checkpoint
        checkpoint = self.load_checkpoint()
        if checkpoint:
            logger.info(f"Loaded checkpoint with {len(checkpoint)} frames")
            # Restore used images
            for data in checkpoint.values():
                if data.get("selected"):
                    self.used_images.add(data["selected"])

        # Get image files
        image_files = self.get_image_files()
        logger.info(f"Found {len(image_files)} images in folder")

        if not image_files:
            msg = f"No images found in {self.image_folder}"
            raise ValueError(msg)

        # Pre-compute image features (aspect-ratio independent)
        self.precompute_image_features(image_files)

        # Calculate target aspect ratio
        target_aspect_ratio = self.frame_width / self.frame_height
        logger.info(f"Video aspect ratio: {target_aspect_ratio:.3f}")

        # Process frames one by one
        logger.info(f"Processing {frames_to_process} frames...")
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            msg = f"Cannot open video: {self.video_path}"
            raise ValueError(msg)

        frame_num = 0
        frames_since_checkpoint = 0

        with tqdm(total=frames_to_process, desc="Matching frames") as pbar:
            while frame_num < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_key = f"frame_{frame_num:06d}"

                # Skip if already processed
                if frame_key not in checkpoint or checkpoint[frame_key].get("selected") is None:
                    # Find top matches
                    top_matches = self.find_top_matches(frame, frame_num, target_aspect_ratio)

                    # Select random match
                    selected = self.select_random_match(top_matches)

                    # Store result
                    frame_result = FrameResult(top_matches=top_matches, selected=selected)
                    checkpoint[frame_key] = frame_result.model_dump()

                    frames_since_checkpoint += 1

                    # Save checkpoint every N frames
                    if frames_since_checkpoint >= self.checkpoint_batch_size:
                        self.save_checkpoint(checkpoint)
                        frames_since_checkpoint = 0

                self.results[frame_key] = checkpoint[frame_key]
                frame_num += 1
                pbar.update(1)

        cap.release()

        # Save final checkpoint and results
        self.save_checkpoint(checkpoint)
        self.save_results()
        logger.info(f"Results saved to {self.results_path}")

        return checkpoint

    def generate_output(self, checkpoint: dict[str, Any]) -> None:
        """Generate output video and individual frame images from matching results.

        Creates output video with matched images cropped and resized to match
        video dimensions. Also saves individual frames as JPG images.

        Creates:
            - {video_name}_matched.mp4: Video file with matched images.
            - matched_frames/: Folder with individual frame images.

        Args:
            checkpoint: Dictionary mapping frame keys to FrameResult data.
        """
        output_video_path = self.output_dir / f"{self.video_path.stem}_matched.mp4"
        output_images_dir = self.output_dir / "matched_frames"
        output_images_dir.mkdir(exist_ok=True)

        logger.info("Generating output...")

        # Use FPS override if specified, otherwise use video FPS
        output_fps = self.fps_override if self.fps_override is not None else self.fps
        if self.fps_override is not None:
            logger.info(f"Using FPS override: {self.fps_override} (original: {self.fps})")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]  # VideoWriter_fourcc exists but not in type stubs
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, output_fps,
                                      (self.frame_width, self.frame_height))

        # Sort frames by frame number
        sorted_frames = sorted(checkpoint.items(),
                             key=lambda x: int(x[0].split("_")[1]))

        for frame_key, data in tqdm(sorted_frames, desc="Generating output"):
            selected_path = data.get("selected")

            if selected_path and Path(selected_path).exists():
                # Load selected image
                img = cv2.imread(selected_path)

                if img is not None:
                    # Crop to video aspect ratio
                    video_ratio = self.frame_width / self.frame_height
                    img_cropped = self.matcher.aspect_ratio_crop(img, video_ratio)

                    # Resize to video dimensions
                    img_resized = cv2.resize(img_cropped,
                                            (self.frame_width, self.frame_height))
                else:
                    # Create blank frame if image failed to load
                    img_resized = np.zeros((self.frame_height, self.frame_width, 3),
                                          dtype=np.uint8)
            else:
                # Create blank frame if no match
                img_resized = np.zeros((self.frame_height, self.frame_width, 3),
                                      dtype=np.uint8)

            # Write to video
            video_writer.write(img_resized)

            # Save as individual image
            output_img_path = output_images_dir / f"{frame_key}.jpg"
            cv2.imwrite(str(output_img_path), img_resized)

        video_writer.release()

        logger.info(f"Output video saved to: {output_video_path}")
        logger.info(f"Output images saved to: {output_images_dir}")

