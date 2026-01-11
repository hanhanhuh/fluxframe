"""Video frame matching using unified metrics (LAB, SSIM, GIST)."""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import Config, RenderTarget
from .database import ImageDatabase
from .models import FrameResult, VideoInfo
from .search import SearchIndex

logger = logging.getLogger(__name__)


class VideoFrameMatcher:
    """Match video frames to image dataset using unified metrics."""

    def __init__(
        self,
        video_path: str,
        image_folder: str,
        output_dir: str,
        metric: str = "lab",
        top_n: int = 10,
        threshold: float = 0.0,
        no_repeat: bool = False,
        demo_mode: bool = False,
        demo_seconds: int = 20,
        demo_images: int = 1000,
        checkpoint_batch_size: int = 10,
        seed: int | None = None,
        fps_override: float | None = None,
        save_samples: int = 0,
        sample_interval: int = 1,
        weights: tuple[float, float, float] = (1.0, 2.0, 2.0),
        ssim_weight: float = 0.5,
    ):
        """Initialize video frame matcher.

        Args:
            video_path: Path to input video file
            image_folder: Path to folder containing images
            output_dir: Directory for outputs and checkpoints
            metric: Distance metric (lab, ssim, lab+ssim, gist)
            top_n: Number of top matches to consider
            threshold: Minimum similarity threshold
            no_repeat: If True, use each image only once
            demo_mode: If True, use only subset of video and images
            demo_seconds: Number of seconds to process in demo mode
            demo_images: Number of images to use in demo mode
            checkpoint_batch_size: Save checkpoint every N frames
            seed: Random seed for reproducibility
            fps_override: Override video FPS
            save_samples: Number of comparison samples to save
            sample_interval: Save every Nth frame as sample
            weights: LAB channel weights (L, A, B)
            ssim_weight: SSIM weight for lab+ssim hybrid
        """
        self.video_path = Path(video_path)
        self.image_folder = Path(image_folder)
        self.output_dir = Path(output_dir)
        self.metric = metric
        self.top_n = top_n
        self.threshold = threshold
        self.no_repeat = no_repeat
        self.demo_mode = demo_mode
        self.demo_seconds = demo_seconds
        self.demo_images = demo_images
        self.checkpoint_batch_size = checkpoint_batch_size
        self.seed = seed
        self.fps_override = fps_override
        self.save_samples = save_samples
        self.sample_interval = sample_interval
        self.weights = weights
        self.ssim_weight = ssim_weight

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Initialize video info
        self.video_info = self._get_video_info()

        # Create config for database/search
        self.config = Config(
            img_dir=self.image_folder,
            output_dir=self.output_dir,
            targets=[],  # Not rendering yet
            metric=metric,  # type: ignore[arg-type]
            weights=weights,
            ssim_weight=ssim_weight,
        )

        # Initialize database and search index
        logger.info(f"Loading image database from {self.image_folder}")
        self.db = ImageDatabase(self.config)

        # Apply demo mode image limit
        if self.demo_mode and len(self.db) > self.demo_images:
            logger.info(f"Demo mode: Using {self.demo_images} of {len(self.db)} images")
            # Randomly sample images
            indices = np.random.choice(len(self.db), self.demo_images, replace=False)
            self.db.filenames = [self.db.filenames[i] for i in indices]
            self.db.lab_images = self.db.lab_images[indices]

        logger.info(f"Building search index with metric: {metric}")
        self.search_index = SearchIndex(self.config, self.db)

        # Tracking
        self.used_indices: set[int] = set()
        self.checkpoint_path = self.output_dir / "checkpoint.json"

    def _get_video_info(self) -> VideoInfo:
        """Extract video information."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps_override:
            fps = self.fps_override

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        if self.demo_mode:
            max_frames = int(self.demo_seconds * fps)
            total_frames = min(total_frames, max_frames)

        return VideoInfo(
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
        )

    def _frame_to_lab(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to LAB vector (like ImageDatabase does).

        Args:
            frame: BGR frame

        Returns:
            Flattened LAB vector (64*64*3,)
        """
        # Resize to 64x64 (same as database)
        resized = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)

        # Convert to LAB
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)

        # Flatten
        return lab.reshape(-1).astype(np.float32)

    def process(self) -> dict:
        """Process video and match frames to images.

        Returns:
            Checkpoint dictionary with results
        """
        # Load checkpoint if exists
        if self.checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            with self.checkpoint_path.open() as f:
                checkpoint = json.load(f)
                start_frame = len(checkpoint["frames"])
                logger.info(f"Resuming from frame {start_frame}")
        else:
            checkpoint = {"frames": []}
            start_frame = 0

        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

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
                frame_lab = self._frame_to_lab(frame)

                # Search for nearest neighbors
                distances, indices = self.search_index.search(
                    frame_lab.reshape(1, -1),
                    k=self.top_n + len(self.used_indices) if self.no_repeat else self.top_n
                )

                # Select best match
                selected_idx = self._select_match(indices[0], distances[0])

                # Save result
                result = FrameResult(
                    frame_number=frame_idx,
                    selected_image=self.db.filenames[selected_idx],
                    similarity_score=float(1.0 / (1.0 + distances[0][0])),  # Convert distance to similarity
                )

                checkpoint["frames"].append({
                    "frame_number": result.frame_number,
                    "selected_image": result.selected_image,
                    "similarity_score": result.similarity_score,
                })

                # Mark as used
                if self.no_repeat:
                    self.used_indices.add(selected_idx)

                # Save checkpoint
                if (frame_idx + 1) % self.checkpoint_batch_size == 0:
                    self._save_checkpoint(checkpoint)

        finally:
            cap.release()

        # Final checkpoint save
        self._save_checkpoint(checkpoint)

        return checkpoint

    def _select_match(self, indices: np.ndarray, distances: np.ndarray) -> int:
        """Select best match from candidates.

        Args:
            indices: Array of candidate indices
            distances: Array of distances

        Returns:
            Selected image index
        """
        if self.no_repeat:
            # Filter out used indices
            valid_mask = np.array([idx not in self.used_indices for idx in indices])
            if not valid_mask.any():
                # All used - pick best even if used
                return int(indices[0])

            valid_indices = indices[valid_mask]
            valid_distances = distances[valid_mask]

            # Return best unused
            return int(valid_indices[0])
        else:
            # Return best match
            return int(indices[0])

    def _save_checkpoint(self, checkpoint: dict) -> None:
        """Save checkpoint to disk."""
        with self.checkpoint_path.open("w") as f:
            json.dump(checkpoint, f, indent=2)

    def generate_output(self, checkpoint: dict) -> None:
        """Generate output video from checkpoint.

        Args:
            checkpoint: Checkpoint dictionary with matched frames
        """
        logger.info("Generating output video...")

        # Get video properties
        output_path = self.output_dir / f"{self.video_path.stem}_matched.mp4"

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
                img_path = self.image_folder / frame_data["selected_image"]
                img = cv2.imread(str(img_path))

                if img is not None:
                    # Resize to match video dimensions
                    img_resized = cv2.resize(
                        img,
                        (self.video_info.width, self.video_info.height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    out.write(img_resized)

        finally:
            out.release()

        logger.info(f"Output video saved to {output_path}")
