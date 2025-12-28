"""Video frame to image matching processor with FAISS vector search."""

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import cv2
import faiss
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .matcher import FeatureMethod, ImageMatcher
from .models import FrameResult, VideoInfo

logger = logging.getLogger(__name__)


class VideoImageMatcher:
    """Main class for matching video frames to image dataset using FAISS."""

    # Adaptive search constants
    SUCCESS_STREAK_THRESHOLD = 3  # Successes needed before shrinking search depth
    SEARCH_DEPTH_REDUCTION = 0.8  # Multiply by this to shrink (20% reduction)
    SEARCH_DEPTH_EXPANSION = 1.5  # Multiply by this to expand (50% increase)

    def __init__(  # noqa: PLR0913
        self, video_path: str, image_folder: str, output_dir: str,
        top_n: int = 10, edge_weight: float = 0.33,
        texture_weight: float = 0.33, color_weight: float = 0.34,
        similarity_threshold: float = 0.0, no_repeat: bool = False,
        comparison_size: int = 256, demo_mode: bool = False,
        demo_seconds: int = 20, demo_images: int = 1000,
        checkpoint_batch_size: int = 10, seed: int | None = None,
        num_workers: int | None = None, fps_override: float | None = None,  # noqa: ARG002
        feature_method: FeatureMethod = "canny"
    ):
        """
        Initialize the video-image matcher with FAISS indexing.

        Args:
            video_path: Path to input video file
            image_folder: Path to folder containing images
            output_dir: Directory for outputs and checkpoints
            top_n: Initial number of top similar images to consider
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
            num_workers: Number of parallel workers (deprecated, kept for compatibility)
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
        self.fps_override = fps_override

        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.matcher = ImageMatcher(edge_weight, texture_weight, color_weight, feature_method)
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.results_path = self.output_dir / "results.json"

        # FAISS cache files
        self.cache_metadata_path = self.output_dir / "cache_metadata.json"
        self.faiss_index_path = self.output_dir / "faiss_index.bin"
        self.vectors_path = self.output_dir / "vectors.npy"

        self.used_images: set[str] = set()
        self.results: dict[str, Any] = {}

        # FAISS attributes
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.image_paths: list[str] = []
        self.vectors: npt.NDArray[np.float32] | None = None
        self.feature_dim = 768  # Will be adjusted based on actual feature dimensions

        # Adaptive search attributes
        self.current_search_k = top_n
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.max_search_k = min(top_n * 5, 100)  # Cap at 5x initial or 100

        # Frame rate control
        self.input_fps: float = 0.0
        self.input_fps_skip_interval = 1

    def _generate_cache_key(self, image_files: list[Path]) -> str:
        """Generate a unique cache key based on parameters.

        Args:
            image_files: List of image file paths.

        Returns:
            SHA256 hash of relevant parameters.
        """
        # Include all parameters that affect feature computation
        params = {
            "comparison_size": self.comparison_size,
            "edge_weight": self.matcher.edge_weight,
            "texture_weight": self.matcher.texture_weight,
            "color_weight": self.matcher.color_weight,
            "feature_method": self.matcher.feature_method,
            "image_paths": sorted([str(p) for p in image_files])
        }

        params_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(params_str.encode()).hexdigest()

    def _validate_cache(self, image_files: list[Path]) -> bool:
        """Validate that cache exists and matches current parameters.

        Args:
            image_files: List of image file paths to validate against.

        Returns:
            True if cache is valid and can be used.
        """
        if not all([
            self.cache_metadata_path.exists(),
            self.faiss_index_path.exists(),
            self.vectors_path.exists()
        ]):
            logger.info("Cache files not found, will rebuild")
            return False

        try:
            with self.cache_metadata_path.open() as f:
                metadata = json.load(f)

            current_key = self._generate_cache_key(image_files)

            if metadata.get("cache_key") != current_key:
                logger.info("Cache parameters changed, will rebuild")
                return False

            logger.info("Cache validation successful")
            return True

        except Exception as e:
            logger.warning(f"Error validating cache: {e}, will rebuild")
            return False

    def _build_faiss_index(self, image_files: list[Path]) -> None:
        """Build FAISS index from image features with caching.

        Args:
            image_files: List of image file paths to index.
        """
        # Check cache validity
        if self._validate_cache(image_files):
            logger.info("Loading FAISS index from cache...")
            try:
                # Load metadata
                with self.cache_metadata_path.open() as f:
                    metadata = json.load(f)

                # Load image paths
                self.image_paths = metadata["image_paths"]

                # Load vectors
                self.vectors = np.load(self.vectors_path)
                self.feature_dim = self.vectors.shape[1]

                # Load FAISS index
                self.faiss_index = faiss.read_index(str(self.faiss_index_path))

                logger.info(f"Loaded FAISS index with {len(self.image_paths)} images, "
                          f"dimension {self.feature_dim}")
                return

            except Exception as e:
                logger.warning(f"Error loading cache: {e}, rebuilding...")

        # Build new index
        logger.info(f"Building FAISS index for {len(image_files)} images...")
        logger.info(f"  Comparison size: {self.comparison_size}px")

        feature_vectors = []
        valid_paths = []

        # Common aspect ratios to test
        test_aspect_ratio = 16 / 9  # Use most common aspect ratio for feature dimension

        for img_path in tqdm(image_files, desc="Computing features"):
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue

            # Crop to aspect ratio and resize
            img_cropped = self.matcher.aspect_ratio_crop(img, test_aspect_ratio)
            img_small = cv2.resize(
                img_cropped,
                (self.comparison_size, int(self.comparison_size / test_aspect_ratio))
            )

            # Compute features
            features = self.matcher.compute_all_features(img_small)

            # Concatenate weighted features into single vector
            edge_vec = features["edge"] * self.matcher.edge_weight
            texture_vec = features["texture"] * self.matcher.texture_weight
            color_vec = features["color"] * self.matcher.color_weight

            combined = np.concatenate([edge_vec, texture_vec, color_vec])
            feature_vectors.append(combined)
            valid_paths.append(str(img_path))

        if not feature_vectors:
            msg = "No valid images found"
            raise ValueError(msg)

        # Convert to numpy array
        self.vectors = np.array(feature_vectors, dtype=np.float32)
        self.image_paths = valid_paths
        self.feature_dim = self.vectors.shape[1]

        # Normalize vectors for inner product (cosine similarity)
        faiss.normalize_L2(self.vectors)

        # Build FAISS index (IndexFlatIP for exact inner product search)
        self.faiss_index = faiss.IndexFlatIP(self.feature_dim)
        self.faiss_index.add(self.vectors)

        logger.info(f"Built FAISS index: {len(self.image_paths)} images, "
                   f"dimension {self.feature_dim}")

        # Save cache
        logger.info("Saving FAISS cache...")

        # Save metadata
        metadata = {
            "cache_key": self._generate_cache_key(image_files),
            "image_paths": self.image_paths,
            "feature_dim": self.feature_dim,
            "num_images": len(self.image_paths)
        }
        with self.cache_metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)

        # Save FAISS index
        faiss.write_index(self.faiss_index, str(self.faiss_index_path))

        # Save vectors
        np.save(self.vectors_path, self.vectors)

        logger.info("FAISS cache saved successfully")

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
        """Find top matches using FAISS with adaptive search depth.

        Args:
            frame: Video frame to match (BGR image).
            frame_num: Frame number (currently unused, kept for API compatibility).
            target_aspect_ratio: Target aspect ratio as width/height.

        Returns:
            List of (image_path, similarity_score) tuples sorted by similarity.
        """
        if self.faiss_index is None or not self.image_paths:
            msg = "FAISS index not built"
            raise ValueError(msg)

        # Prepare frame for comparison
        frame_small = cv2.resize(
            frame,
            (self.comparison_size, int(self.comparison_size / target_aspect_ratio))
        )

        # Compute frame features
        features = self.matcher.compute_all_features(frame_small)

        # Combine weighted features
        edge_vec = features["edge"] * self.matcher.edge_weight
        texture_vec = features["texture"] * self.matcher.texture_weight
        color_vec = features["color"] * self.matcher.color_weight

        query_vector = np.concatenate([edge_vec, texture_vec, color_vec])
        query_vector = query_vector.reshape(1, -1).astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)

        # Search with adaptive k
        k = min(self.current_search_k, len(self.image_paths))
        distances, indices = self.faiss_index.search(query_vector, k)

        # Convert to list of (path, score) tuples
        # Inner product distances are already similarity scores (higher = better)
        return [
            (self.image_paths[idx], float(dist))
            for dist, idx in zip(distances[0], indices[0], strict=True)
            if idx >= 0  # Valid index
        ]

    def _adjust_search_depth(self, success: bool) -> None:
        """Adjust adaptive search depth based on success/failure.

        Args:
            success: Whether the last selection was successful.
        """
        if success:
            self.consecutive_successes += 1
            self.consecutive_failures = 0

            # Shrink toward top_n on success streak
            if (
                self.consecutive_successes >= self.SUCCESS_STREAK_THRESHOLD
                and self.current_search_k > self.top_n
            ):
                old_k = self.current_search_k
                self.current_search_k = max(
                    self.top_n,
                    int(self.current_search_k * self.SEARCH_DEPTH_REDUCTION)
                )
                logger.debug(f"Search depth reduced: {old_k} -> {self.current_search_k}")
                self.consecutive_successes = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0

            # Expand on failure (every failure)
            if self.current_search_k < self.max_search_k:
                old_k = self.current_search_k
                self.current_search_k = min(
                    self.max_search_k,
                    int(self.current_search_k * self.SEARCH_DEPTH_EXPANSION)
                )
                logger.debug(f"Search depth increased: {old_k} -> {self.current_search_k}")

    def select_match(self, top_matches: list[tuple[str, float]]) -> str:  # noqa: PLR0912
        """Intelligent multi-level fallback selection strategy.

        Priority: threshold-compliant → unused → best available → random from dataset.

        Args:
            top_matches: List of (image_path, similarity_score) tuples.

        Returns:
            Selected image path (never None).
        """
        # Level 1: Try threshold-compliant matches
        valid_matches = [
            (path, score) for path, score in top_matches
            if score >= self.similarity_threshold
        ]

        if valid_matches:
            candidates = valid_matches
        elif top_matches:
            # Level 2: Use best available from top_matches
            logger.debug("No matches above threshold, using best available from search")
            candidates = top_matches
            self._adjust_search_depth(success=False)
        else:
            # Level 3: Fallback to random from entire dataset
            logger.warning("No matches found, using random selection from dataset")
            if not self.image_paths:
                msg = "No images available in dataset"
                raise ValueError(msg)

            if self.no_repeat:
                unused = [p for p in self.image_paths if p not in self.used_images]
                if unused:
                    selected = random.choice(unused)
                else:
                    logger.warning("All images used, reusing from dataset")
                    selected = random.choice(self.image_paths)
            else:
                selected = random.choice(self.image_paths)

            self.used_images.add(selected)
            self._adjust_search_depth(success=False)
            return selected

        # Filter for unused images if no_repeat enabled
        if self.no_repeat:
            unused_candidates = [(p, s) for p, s in candidates if p not in self.used_images]

            if unused_candidates:
                candidates = unused_candidates
                self._adjust_search_depth(success=True)
            else:
                # Level 4: All candidates used, expand search or reuse
                logger.debug("All candidates used, expanding search")
                self._adjust_search_depth(success=False)

                # Reuse best available as last resort
                candidates = candidates[:1]  # Best match
        else:
            self._adjust_search_depth(success=True)

        # Random selection from final candidates
        selected_path = random.choice(candidates)[0]

        if self.no_repeat:
            self.used_images.add(selected_path)

        return selected_path

    def process(self) -> dict[str, Any]:  # noqa: PLR0915, PLR0912
        """Main processing pipeline with FAISS indexing and adaptive search.

        Returns:
            dict: Checkpoint data with results for all processed frames.

        Raises:
            ValueError: If video cannot be opened or no images found in folder.
        """
        logger.info(f"Video: {self.video_path}")
        logger.info(f"Image folder: {self.image_folder}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Top N (initial): {self.top_n}")
        logger.info(f"Max search depth: {self.max_search_k}")
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
        logger.info(f"No repeat: {self.no_repeat}")
        logger.info(f"Checkpoint batch size: {self.checkpoint_batch_size}")
        logger.info(f"Feature method: {self.matcher.feature_method}")
        if self.demo_mode:
            logger.info(f"Demo mode: ON (max {self.demo_seconds}s video, "
                       f"{self.demo_images} images)")
        logger.info(f"Weights - Edge: {self.matcher.edge_weight:.2f}, "
                   f"Texture: {self.matcher.texture_weight:.2f}, "
                   f"Color: {self.matcher.color_weight:.2f}")

        # Get video info
        video_info = self.get_video_info()
        self.input_fps = video_info.fps
        frame_width = video_info.width
        frame_height = video_info.height
        total_frames = video_info.total_frames

        # Calculate frame skip interval if fps_override is set
        if self.fps_override is not None:
            self.input_fps_skip_interval = max(1, round(self.input_fps / self.fps_override))
            logger.info(f"FPS override: {self.fps_override} (input: {self.input_fps:.2f}, "
                       f"skip interval: {self.input_fps_skip_interval})")
        else:
            self.input_fps_skip_interval = 1

        # Calculate frames to process
        if self.demo_mode:
            max_frames = int(self.input_fps * self.demo_seconds)
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

        # Build FAISS index
        self._build_faiss_index(image_files)

        # Calculate target aspect ratio
        target_aspect_ratio = frame_width / frame_height
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

                # Check if frame should be processed based on skip interval
                should_process = (frame_num % self.input_fps_skip_interval == 0)

                # Skip if already processed or if not in skip interval
                if should_process and (
                    frame_key not in checkpoint or checkpoint[frame_key].get("selected") is None
                ):
                    # Find top matches
                    top_matches = self.find_top_matches(frame, frame_num, target_aspect_ratio)

                    # Select match with intelligent fallback
                    selected = self.select_match(top_matches)

                    # Store result
                    frame_result = FrameResult(top_matches=top_matches, selected=selected)
                    checkpoint[frame_key] = frame_result.model_dump()

                    frames_since_checkpoint += 1

                    # Save checkpoint every N frames
                    if frames_since_checkpoint >= self.checkpoint_batch_size:
                        self.save_checkpoint(checkpoint)
                        frames_since_checkpoint = 0

                self.results[frame_key] = checkpoint.get(frame_key, {})
                frame_num += 1
                pbar.update(1)

        cap.release()

        # Save final checkpoint and results
        self.save_checkpoint(checkpoint)
        self.save_results()
        logger.info(f"Results saved to {self.results_path}")

        # Verify no null selections
        null_count = sum(1 for v in checkpoint.values() if v.get("selected") is None)
        if null_count > 0:
            logger.warning(f"Found {null_count} frames with null selections!")
        else:
            logger.info("Verification: All frames have valid selections")

        return checkpoint

    def generate_output(self, checkpoint: dict[str, Any]) -> None:
        """Generate output video and individual frame images from matching results.

        Creates output video with matched images cropped and resized to match
        video dimensions. Respects demo mode settings.

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

        # Get video info for dimensions
        video_info = self.get_video_info()
        frame_width = video_info.width
        frame_height = video_info.height

        # Use FPS override if specified, otherwise use video FPS
        output_fps = self.fps_override if self.fps_override is not None else self.input_fps
        if self.fps_override is not None:
            logger.info(f"Output FPS: {self.fps_override} (input: {self.input_fps:.2f})")
        else:
            logger.info(f"Output FPS: {output_fps:.2f}")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        video_writer = cv2.VideoWriter(
            str(output_video_path), fourcc, output_fps,
            (frame_width, frame_height)
        )

        # Sort frames by frame number
        sorted_frames = sorted(
            checkpoint.items(),
            key=lambda x: int(x[0].split("_")[1])
        )

        for frame_key, data in tqdm(sorted_frames, desc="Generating output"):
            selected_path = data.get("selected")

            if selected_path and Path(selected_path).exists():
                # Load selected image
                img = cv2.imread(selected_path)

                if img is not None:
                    # Crop to video aspect ratio
                    video_ratio = frame_width / frame_height
                    img_cropped = self.matcher.aspect_ratio_crop(img, video_ratio)

                    # Resize to video dimensions
                    img_resized = cv2.resize(img_cropped, (frame_width, frame_height))
                else:
                    # Create blank frame if image failed to load
                    logger.warning(f"Failed to load image: {selected_path}")
                    img_resized = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            else:
                # Create blank frame if no match
                if selected_path is None:
                    logger.warning(f"No match for {frame_key}")
                else:
                    logger.warning(f"Image not found: {selected_path}")
                img_resized = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            # Write to video
            video_writer.write(img_resized)

            # Save as individual image
            output_img_path = output_images_dir / f"{frame_key}.jpg"
            cv2.imwrite(str(output_img_path), img_resized)

        video_writer.release()

        logger.info(f"Output video saved to: {output_video_path}")
        logger.info(f"Output images saved to: {output_images_dir}")
