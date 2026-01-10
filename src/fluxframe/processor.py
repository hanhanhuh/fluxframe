"""Video frame to image matching processor with FAISS vector search."""

import hashlib
import json
import logging
import multiprocessing as mp
import pickle
import random
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import faiss
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .matcher import FeatureMethod, ImageMatcher, PoolingMethod
from .models import FrameResult, VideoInfo

logger = logging.getLogger(__name__)

# Try to import TurboJPEG for 3x faster image loading
try:
    from turbojpeg import TurboJPEG  # type: ignore[import-not-found]

    _jpeg_decoder = TurboJPEG()
    HAS_TURBOJPEG = True
except ImportError:
    HAS_TURBOJPEG = False


def _fast_imread(img_path: Path) -> npt.NDArray[Any] | None:
    """Fast image loading with TurboJPEG fallback to OpenCV.

    Args:
        img_path: Path to image file

    Returns:
        Loaded image as BGR numpy array, or None if loading fails
    """
    try:
        if HAS_TURBOJPEG and img_path.suffix.lower() in {".jpg", ".jpeg"}:
            # Use TurboJPEG for 3x faster JPEG decoding
            with img_path.open("rb") as f:
                return _jpeg_decoder.decode(f.read())  # type: ignore[no-any-return]
    except Exception:
        pass  # Fall through to OpenCV

    # Fallback to OpenCV
    return cv2.imread(str(img_path))


def _compute_features_worker(  # noqa: PLR0913
    img_path: Path,
    comparison_size: int,
    test_aspect_ratio: float,
    edge_weight: float,
    texture_weight: float,
    color_weight: float,
    feature_method: FeatureMethod,
    pooling_method: PoolingMethod = "avg",
    gem_p: float = 3.0,
    spatial_grid: int = 2,
    use_global_pooling: bool = False,
    spatial_color_scales: list[int] | None = None,
) -> tuple[str, npt.NDArray[np.float32]] | None:
    """Worker function to compute features for a single image (for multiprocessing).

    Args:
        img_path: Path to image file.
        comparison_size: Size to resize images for comparison.
        test_aspect_ratio: Aspect ratio to crop to.
        edge_weight: Weight for edge features.
        texture_weight: Weight for texture features.
        color_weight: Weight for color features.
        feature_method: Feature extraction method to use.

    Returns:
        Tuple of (image_path_str, feature_vector) or None if image fails to load.
    """
    img = _fast_imread(img_path)
    if img is None:
        return None

    # Create matcher for this worker
    matcher = ImageMatcher(
        edge_weight=edge_weight,
        texture_weight=texture_weight,
        color_weight=color_weight,
        feature_method=feature_method,
        pooling_method=pooling_method,
        gem_p=gem_p,
        spatial_grid=spatial_grid,
        use_global_pooling=use_global_pooling,
        spatial_color_scales=spatial_color_scales,
    )

    # Crop to aspect ratio and resize
    img_cropped = matcher.aspect_ratio_crop(img, test_aspect_ratio)
    img_small = cv2.resize(
        img_cropped,
        (comparison_size, int(comparison_size / test_aspect_ratio)),
        interpolation=cv2.INTER_AREA,  # Best for downscaling, 10-15% faster
    )

    # Compute features and convert to vector for FAISS
    features = matcher.compute_all_features(img_small)
    feature_vector = matcher.features_to_vector(features)

    return (str(img_path), feature_vector)


class VideoImageMatcher:
    """Main class for matching video frames to image dataset using FAISS."""

    def __init__(  # noqa: PLR0913
        self,
        video_path: str,
        image_folder: str,
        output_dir: str,
        top_n: int = 10,
        edge_weight: float = 0.33,
        texture_weight: float = 0.33,
        color_weight: float = 0.34,
        similarity_threshold: float = 0.0,
        no_repeat: bool = False,
        comparison_size: int = 256,
        demo_mode: bool = False,
        demo_seconds: int = 20,
        demo_images: int = 1000,
        checkpoint_batch_size: int = 10,
        seed: int | None = None,
        num_workers: int | None = None,
        fps_override: float | None = None,
        feature_method: FeatureMethod = "canny",
        save_samples: int = 0,
        sample_interval: int = 1,
        force_mobilenet_export: bool = False,
        pooling_method: PoolingMethod = "avg",
        gem_p: float = 3.0,
        spatial_grid: int = 2,
        use_ivf_index: bool = False,
        ivf_nlist: int | None = None,
        ivf_nprobe: int | None = None,
        use_global_pooling: bool = False,
        spatial_color_scales: list[int] | None = None,
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
            feature_method: Feature extraction method
                - "canny": Fast edge histogram
                - "spatial_pyramid": Spatial layout (2×2 or 3×3)
                - "hog": Motion preservation
                - "spatial_color": SPM color pyramid [1,3] (1120D, no reduction)
                - "spatial_color_edge": SPM color+edge pyramid [1,3] (1600D, 20% edges, recommended)
                - "mobilenet"/"efficientnet": Neural features
            save_samples: Number of comparison samples to save (0 = disabled)
            sample_interval: Save every Nth frame as sample (1 = every frame)
            use_ivf_index: Use IndexIVFFlat for 16x faster search (default: True)
            ivf_nlist: Number of IVF clusters (auto: sqrt(N) * 4, max 4096)
            ivf_nprobe: Number of clusters to search (auto: nlist/32, min 1)
            spatial_color_scales: Grid scales for spatial_color method (default [1,3,5])
        """
        self.video_path = Path(video_path)
        self.image_folder = Path(image_folder)
        self.output_dir = Path(output_dir)
        self.top_n = top_n
        self.save_samples = save_samples
        self.sample_interval = sample_interval
        self.samples_saved = 0
        self.similarity_threshold = similarity_threshold
        self.no_repeat = no_repeat
        self.comparison_size = comparison_size
        self.demo_mode = demo_mode
        self.demo_seconds = demo_seconds
        self.demo_images = demo_images
        self.checkpoint_batch_size = checkpoint_batch_size
        self.fps_override = fps_override
        self.use_ivf_index = use_ivf_index
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe

        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.matcher = ImageMatcher(
            edge_weight=edge_weight,
            texture_weight=texture_weight,
            color_weight=color_weight,
            feature_method=feature_method,
            force_mobilenet_export=force_mobilenet_export,
            pooling_method=pooling_method,
            gem_p=gem_p,
            spatial_grid=spatial_grid,
            use_global_pooling=use_global_pooling,
            spatial_color_scales=spatial_color_scales,
        )
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.results_path = self.output_dir / "results.json"

        # Sample comparison directory
        self.samples_dir = self.output_dir / "comparison_samples"
        if self.save_samples > 0:
            self.samples_dir.mkdir(parents=True, exist_ok=True)

        # FAISS cache files
        self.cache_metadata_path = self.output_dir / "cache_metadata.json"
        self.faiss_index_path = self.output_dir / "faiss_index.bin"
        self.vectors_path = self.output_dir / "vectors.npy"
        self.dim_reducer_path = self.output_dir / "dim_reducer.pkl"

        self.used_images: set[str] = set()
        self.results: dict[str, Any] = {}

        # FAISS attributes
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.image_paths: list[str] = []
        self.vectors: npt.NDArray[np.float32] | None = None
        self.feature_dim = 768  # Will be adjusted based on actual feature dimensions

        # Search depth (fixed, no adaptive search)
        self.search_k = top_n

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
            "image_paths": sorted([str(p) for p in image_files]),
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
        if not all(
            [
                self.cache_metadata_path.exists(),
                self.faiss_index_path.exists(),
                self.vectors_path.exists(),
            ]
        ):
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

    def _build_faiss_index(self, image_files: list[Path]) -> None:  # noqa: PLR0912, PLR0915
        """Build FAISS index from image features with caching.

        Args:
            image_files: List of image file paths to index.
        """
        # Check cache validity
        if self._validate_cache(image_files):
            logger.info("Loading FAISS index from cache...")
            try:
                with self.cache_metadata_path.open() as f:
                    metadata = json.load(f)

                self.image_paths = metadata["image_paths"]
                self.vectors = np.load(self.vectors_path)
                self.feature_dim = self.vectors.shape[1]

                # Load dimensionality reducer if it exists
                if self.dim_reducer_path.exists():
                    with self.dim_reducer_path.open("rb") as f:
                        self.matcher.dim_reducer = pickle.load(f)
                    logger.info(f"Loaded dimensionality reducer ({self.matcher.reduced_dims}D)")

                self.faiss_index = faiss.read_index(str(self.faiss_index_path), faiss.IO_FLAG_MMAP)

                logger.info(
                    f"Loaded FAISS index (memory-mapped) with {len(self.image_paths)} images, "
                    f"dimension {self.feature_dim}"
                )
                return

            except Exception as e:
                logger.warning(f"Error loading cache: {e}, rebuilding...")

        # Build new index
        logger.info(f"Building FAISS index for {len(image_files)} images...")

        test_aspect_ratio = 16 / 9
        worker_func = partial(
            _compute_features_worker,
            comparison_size=self.comparison_size,
            test_aspect_ratio=test_aspect_ratio,
            edge_weight=self.matcher.edge_weight,
            texture_weight=self.matcher.texture_weight,
            color_weight=self.matcher.color_weight,
            feature_method=self.matcher.feature_method,
            pooling_method=self.matcher.pooling_method,
            gem_p=self.matcher.gem_p,
            spatial_grid=self.matcher.spatial_grid,
            use_global_pooling=self.matcher.use_global_pooling,
            spatial_color_scales=self.matcher.spatial_color_scales,
        )

        # Zero-disk random projection approach for spatial_color and spatial_color_edge
        if self.matcher.reduce_spatial_color and self.matcher.feature_method in (
            "spatial_color",
            "spatial_color_edge",
        ):
            logger.info(
                "Using zero-disk random projection for spatial_color (instant dimensionality reduction!)"
            )

            # Randomize image order for representative sampling
            shuffled_files = image_files.copy()
            random.shuffle(shuffled_files)

            # Determine random projection fit size
            # Use more samples for better projection quality (especially for high-D features)
            fit_size = min(10000, len(shuffled_files))  # 10k samples for better quality
            logger.info(f"Phase 1/2: Fitting random projection on {fit_size} images...")

            # Phase 1: Extract small subset to fit random projection
            subset_files = shuffled_files[:fit_size]

            if self.matcher.feature_method in ("mobilenet", "efficientnet"):
                logger.info("  Using single-process mode (ONNX Runtime handles threading)")
                num_workers = 0
            else:
                num_workers = max(1, mp.cpu_count() - 1)
                logger.info(f"  Using {num_workers} parallel workers for feature extraction")

            if num_workers > 0:
                pool_fit = mp.Pool(processes=num_workers)
                fit_iterator = pool_fit.imap_unordered(worker_func, subset_files, chunksize=50)
            else:
                fit_iterator = (worker_func(img_path) for img_path in subset_files)

            fit_features = []

            for result in tqdm(
                fit_iterator, total=len(subset_files), desc="Extracting for RP fit", smoothing=0.05
            ):
                if result is not None:
                    _, features = result
                    fit_features.append(features)

            if num_workers > 0:
                pool_fit.close()
                pool_fit.join()

            # Fit dimensionality reducer (instant - just sets up random matrix)
            fit_array = np.array(fit_features, dtype=np.float32)
            logger.info(f"Fitting random projection on {len(fit_array)} samples (instant)...")
            self.matcher.fit_reducer(fit_array)  # Single call, no batch loop needed

            del fit_features, fit_array  # Free memory
            logger.info("Random projection fitted")

            # Phase 2: Extract all features with mini-batch transform (34x faster than individual)
            logger.info("Phase 2/2: Extracting all features with mini-batch transform...")

            if num_workers > 0:
                pool_transform = mp.Pool(processes=num_workers)
                feature_iterator = pool_transform.imap_unordered(
                    worker_func, image_files, chunksize=50
                )
            else:
                feature_iterator = (worker_func(img_path) for img_path in image_files)

            feature_vectors = []
            valid_paths = []
            batch_features = []
            batch_paths = []
            batch_size = 5000  # Transform in batches to avoid individual transform overhead

            for result in tqdm(
                feature_iterator, total=len(image_files), desc="Computing features", smoothing=0.05
            ):
                if result is not None:
                    path, features = result
                    batch_features.append(features)
                    batch_paths.append(str(path))

                    # Transform batch when full (34x faster than individual transforms)
                    if len(batch_features) >= batch_size:
                        batch_array = np.array(batch_features, dtype=np.float32)
                        transformed_batch = self.matcher.transform_features(batch_array)
                        feature_vectors.extend(transformed_batch)
                        valid_paths.extend(batch_paths)
                        batch_features.clear()
                        batch_paths.clear()

            # Transform remaining batch
            if batch_features:
                batch_array = np.array(batch_features, dtype=np.float32)
                transformed_batch = self.matcher.transform_features(batch_array)
                feature_vectors.extend(transformed_batch)
                valid_paths.extend(batch_paths)

            if num_workers > 0:
                pool_transform.close()
                pool_transform.join()

            vectors = np.array(feature_vectors, dtype=np.float32)

            # Save dimensionality reducer to cache
            with self.dim_reducer_path.open("wb") as f:
                pickle.dump(self.matcher.dim_reducer, f)

            logger.info(f"Random projection reduction complete: {vectors.shape[1]}D vectors")

            self.vectors = vectors
            self.image_paths = valid_paths
            self.feature_dim = self.vectors.shape[1]
        else:
            # Original non-streaming path for other methods
            feature_vectors = []
            valid_paths = []

            if self.matcher.feature_method in ("mobilenet", "efficientnet"):
                logger.info("  Using single-process mode (ONNX Runtime handles threading)")
                for img_path in tqdm(image_files, desc="Computing features", smoothing=0.05):
                    result = worker_func(img_path)
                    if result is not None:
                        path, features = result
                        valid_paths.append(path)
                        feature_vectors.append(features)
            else:
                num_workers = max(1, mp.cpu_count() - 1)
                logger.info(f"  Using {num_workers} parallel workers for feature extraction")
                with mp.Pool(processes=num_workers) as pool:
                    results = pool.imap_unordered(worker_func, image_files, chunksize=50)
                    for result in tqdm(
                        results, total=len(image_files), desc="Computing features", smoothing=0.05
                    ):
                        if result is not None:
                            path, features = result
                            valid_paths.append(path)
                            feature_vectors.append(features)

            if not feature_vectors:
                msg = "No valid images found"
                raise ValueError(msg)

            vectors = np.array(feature_vectors, dtype=np.float32)
            self.vectors = vectors
            self.image_paths = valid_paths
            self.feature_dim = self.vectors.shape[1]

        # Normalize vectors
        faiss.normalize_L2(self.vectors)

        # Build FAISS index (existing code unchanged)
        min_ivf_images = 100
        if self.use_ivf_index and len(self.image_paths) >= min_ivf_images:
            if self.ivf_nlist is None:
                nlist = min(int(np.sqrt(len(self.image_paths)) * 4), 4096)
            else:
                nlist = self.ivf_nlist

            nprobe = max(1, nlist // 32) if self.ivf_nprobe is None else self.ivf_nprobe

            logger.info(f"Building IVF index with {nlist} clusters, nprobe={nprobe}...")
            quantizer = faiss.IndexFlatIP(self.feature_dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.feature_dim, nlist)
            self.faiss_index.train(self.vectors)
            self.faiss_index.add(self.vectors)
            self.faiss_index.nprobe = nprobe

            logger.info(
                f"Built IVF index: {len(self.image_paths)} images, dimension {self.feature_dim}"
            )
        else:
            self.faiss_index = faiss.IndexFlatIP(self.feature_dim)
            self.faiss_index.add(self.vectors)
            logger.info(
                f"Built flat index: {len(self.image_paths)} images, dimension {self.feature_dim}"
            )

        # Save cache
        logger.info("Saving FAISS cache...")
        metadata = {
            "cache_key": self._generate_cache_key(image_files),
            "image_paths": self.image_paths,
            "feature_dim": self.feature_dim,
            "num_images": len(self.image_paths),
        }
        with self.cache_metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)

        faiss.write_index(self.faiss_index, str(self.faiss_index_path))
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

        Uses iterdir() for faster scanning than multiple glob() calls.

        Returns:
            Sorted list of image file paths (limited by demo_images if demo mode enabled).
        """
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        # Use iterdir() - faster than multiple glob() calls (11x faster per benchmarks)
        image_files = [
            p for p in self.image_folder.iterdir() if p.is_file() and p.suffix.lower() in extensions
        ]

        logger.info(f"Sorting {len(image_files)} image paths...")
        image_files = sorted(image_files)

        # Apply demo mode limit
        if self.demo_mode and len(image_files) > self.demo_images:
            logger.info(
                f"Demo mode: Using first {self.demo_images} images out of {len(image_files)}"
            )
            image_files = image_files[: self.demo_images]

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
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

        cap.release()
        return info

    def find_top_matches(
        self,
        frame: npt.NDArray[Any],
        frame_num: int,  # noqa: ARG002
        target_aspect_ratio: float,
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
            frame, (self.comparison_size, int(self.comparison_size / target_aspect_ratio))
        )

        # Compute frame features and convert to vector
        features = self.matcher.compute_all_features(frame_small)
        query_vector = self.matcher.features_to_vector(features)

        if self.matcher.reduce_spatial_color and self.matcher.feature_method in (
            "spatial_color",
            "spatial_color_edge",
        ):
            query_vector = self.matcher.transform_features(query_vector)

        query_vector = query_vector.reshape(1, -1)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)

        # Search FAISS - expand k if filtering needed
        if self.no_repeat and self.used_images:
            # Search more to compensate for filtering
            search_k = min(self.search_k + len(self.used_images), len(self.image_paths))
        else:
            search_k = self.search_k

        distances, indices = self.faiss_index.search(query_vector, search_k)

        # Filter out used images and return top k
        results = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx < 0:  # Invalid index
                continue
            path = self.image_paths[idx]
            if self.no_repeat and path in self.used_images:
                continue  # Skip used images
            results.append((path, float(dist)))
            if len(results) >= self.search_k:
                break  # Got enough results

        return results

    def save_comparison_sample(
        self, frame: npt.NDArray[Any], matched_image_path: str, frame_num: int, similarity: float
    ) -> None:
        """Save a side-by-side comparison of frame and matched image.

        Args:
            frame: Original video frame.
            matched_image_path: Path to the matched image.
            frame_num: Frame number.
            similarity: Similarity score of the match.
        """
        # Load matched image
        matched_img = cv2.imread(matched_image_path)
        if matched_img is None:
            logger.warning(f"Failed to load matched image for sample: {matched_image_path}")
            return

        # Resize both to same height for side-by-side comparison
        target_height = 480
        frame_aspect = frame.shape[1] / frame.shape[0]
        matched_aspect = matched_img.shape[1] / matched_img.shape[0]

        frame_resized = cv2.resize(frame, (int(target_height * frame_aspect), target_height))
        matched_resized = cv2.resize(
            matched_img, (int(target_height * matched_aspect), target_height)
        )

        # Create side-by-side comparison
        comparison = np.hstack([frame_resized, matched_resized])

        # Add text overlay with frame number and similarity
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Frame {frame_num:06d} | Similarity: {similarity:.3f}"
        cv2.putText(comparison, text, (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Save comparison image
        sample_path = self.samples_dir / f"sample_{frame_num:06d}.jpg"
        cv2.imwrite(str(sample_path), comparison)

    def select_match(self, top_matches: list[tuple[str, float]]) -> str | None:
        """Select best match from candidates.

        With no_repeat mode, always returns the best (highest similarity) match.
        Without no_repeat, randomly samples from top matches for variety.

        Args:
            top_matches: List of (image_path, similarity_score) tuples, sorted by score.

        Returns:
            Selected image path, or None if no valid matches found.
        """
        if not top_matches:
            logger.warning("No matches found in search")
            return None

        # Filter by threshold if set
        valid_matches = [
            (path, score) for path, score in top_matches if score >= self.similarity_threshold
        ]

        candidates = valid_matches if valid_matches else top_matches

        if not candidates:
            return None

        # Select match based on mode
        if self.no_repeat:
            # Always take the best (first) match - already filtered by find_top_matches
            selected_path = candidates[0][0]
            self.used_images.add(selected_path)
        else:
            # Random selection for variety
            selected_path = random.choice(candidates)[0]

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
        logger.info(f"Search depth: {self.search_k}")
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
        logger.info(f"No repeat: {self.no_repeat}")
        logger.info(f"Checkpoint batch size: {self.checkpoint_batch_size}")
        logger.info(f"Feature method: {self.matcher.feature_method}")
        if self.demo_mode:
            logger.info(
                f"Demo mode: ON (max {self.demo_seconds}s video, {self.demo_images} images)"
            )
        logger.info(
            f"Weights - Edge: {self.matcher.edge_weight:.2f}, "
            f"Texture: {self.matcher.texture_weight:.2f}, "
            f"Color: {self.matcher.color_weight:.2f}"
        )

        # Get video info
        video_info = self.get_video_info()
        self.input_fps = video_info.fps
        frame_width = video_info.width
        frame_height = video_info.height
        total_frames = video_info.total_frames

        # Calculate frame skip interval if fps_override is set
        if self.fps_override is not None:
            self.input_fps_skip_interval = max(1, round(self.input_fps / self.fps_override))
            logger.info(
                f"FPS override: {self.fps_override} (input: {self.input_fps:.2f}, "
                f"skip interval: {self.input_fps_skip_interval})"
            )
        else:
            self.input_fps_skip_interval = 1

        # Calculate frames to process
        if self.demo_mode:
            max_frames = int(self.input_fps * self.demo_seconds)
            frames_to_process = min(max_frames, total_frames)
            logger.info(
                f"Demo mode: Processing first {self.demo_seconds} seconds "
                f"({frames_to_process} frames) out of {total_frames}"
            )
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
        logger.info("Scanning image folder...")
        image_files = self.get_image_files()
        logger.info(f"Found {len(image_files)} images in folder")

        if not image_files:
            msg = f"No images found in {self.image_folder}"
            raise ValueError(msg)

        # Build FAISS index
        logger.info("Preparing FAISS index...")
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

        with tqdm(
            total=frames_to_process,
            desc="Matching frames",
            smoothing=0.05,  # Exponential moving average for smoother ETA
            mininterval=0.5,  # Update display every 0.5 seconds minimum
        ) as pbar:
            while frame_num < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_key = f"frame_{frame_num:06d}"

                # Check if frame should be processed based on skip interval
                should_process = frame_num % self.input_fps_skip_interval == 0

                # Skip if already processed or if not in skip interval
                if should_process and (
                    frame_key not in checkpoint or checkpoint[frame_key].get("selected") is None
                ):
                    # Find top matches (automatically filters used images if no_repeat)
                    top_matches = self.find_top_matches(frame, frame_num, target_aspect_ratio)
                    selected = self.select_match(top_matches)

                    # Fallback if still None (extremely rare - dataset exhausted)
                    if selected is None:
                        logger.warning(
                            f"Frame {frame_num}: All images exhausted, selecting randomly"
                        )
                        unused = [p for p in self.image_paths if p not in self.used_images]
                        selected = (
                            random.choice(unused) if unused else random.choice(self.image_paths)
                        )
                        self.used_images.add(selected)
                        top_matches = [(selected, 0.0)]  # Dummy match

                    # Store result
                    frame_result = FrameResult(top_matches=top_matches, selected=selected)
                    checkpoint[frame_key] = frame_result.model_dump()

                    # Save comparison sample if enabled
                    if (
                        self.save_samples > 0
                        and self.samples_saved < self.save_samples
                        and frame_num % self.sample_interval == 0
                    ):
                        similarity = top_matches[0][1] if top_matches else 0.0
                        self.save_comparison_sample(frame, selected, frame_num, similarity)
                        self.samples_saved += 1

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
            str(output_video_path), fourcc, output_fps, (frame_width, frame_height)
        )

        # Sort frames by frame number
        sorted_frames = sorted(checkpoint.items(), key=lambda x: int(x[0].split("_")[1]))

        for frame_key, data in tqdm(
            sorted_frames,
            desc="Generating output",
            smoothing=0.05,  # Exponential moving average for smoother ETA
            mininterval=0.5,  # Update display every 0.5 seconds minimum
        ):
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
