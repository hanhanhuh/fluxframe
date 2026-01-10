#!/usr/bin/env python3
"""Video rendering module with smart cropping and optional color grading."""

from __future__ import annotations

from contextlib import ExitStack

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from .color_grading import ColorGrader, create_color_grader
from .config import Config
from .database import ImageDatabase


def smart_crop(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Bulletproof cropping that prevents 'All images in a movie should have same size' errors.

    Args:
        img: Source image (BGR)
        target_w: Target width
        target_h: Target height

    Returns:
        Cropped and resized image (target_w x target_h)
    """
    h, w = img.shape[:2]

    # Calculate scale factor
    scale = max(target_w / w, target_h / h)

    # FIX 1: Add rounding buffer (+0.5) so int() doesn't round down (e.g., 1079.9 -> 1079)
    nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Calculate crop offsets (center crop)
    x = (nw - target_w) // 2
    y = (nh - target_h) // 2

    # Ensure non-negative offsets
    x = max(0, x)
    y = max(0, y)

    # Crop
    cropped = resized[y : y + target_h, x : x + target_w]

    # FIX 2: Absolute guarantee (The "Hammer")
    # If due to any math glitch the image is 1921x1080 or 1919x1080,
    # force it back to exact target size
    if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
        cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return cropped


class VideoRenderer:
    """Renders video sequences with optional color grading."""

    def __init__(self, cfg: Config, db: ImageDatabase):
        """Initialize renderer.

        Args:
            cfg: Configuration object
            db: Image database
        """
        self.cfg = cfg
        self.db = db
        self.color_grader = create_color_grader(cfg)

    def render_videos(self, path_indices: list[int]) -> None:
        """Render all target formats simultaneously.

        Args:
            path_indices: List of image indices forming the path
        """
        if not self.cfg.targets:
            print("[Render] No render targets specified, skipping rendering.")
            return

        # Ensure output directory exists
        if not self.cfg.output_dir.exists():
            print(f"[Render] Creating output directory: {self.cfg.output_dir}")
            self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Render] Starting rendering for {len(self.cfg.targets)} formats ({len(path_indices)} frames)...")
        print(f"[Render] Output location: {self.cfg.output_dir.absolute()}")

        if self.color_grader:
            print(
                f"[Render] Color grading enabled: method={self.cfg.color_grading_method}, "
                f"strength={self.cfg.color_grading_strength}"
            )

        # Open all video writers
        with ExitStack() as stack:
            writers = []
            for t in self.cfg.targets:
                out_path = self.cfg.output_dir / t.filename

                print(f"  -> {t.filename} ({t.width}x{t.height})")

                w = stack.enter_context(
                    imageio.get_writer(
                        str(out_path),
                        format="FFMPEG",
                        mode="I",
                        fps=self.cfg.fps,
                        codec="libx264",
                        pixelformat="yuv420p",
                        output_params=["-crf", "18", "-preset", "medium"],
                        macro_block_size=None,
                    )
                )
                writers.append((w, t))

            # Render frames
            prev_frames: dict[str, np.ndarray] = {}  # Track previous frame per target

            for frame_idx, idx in enumerate(tqdm(path_indices, desc="Encoding")):
                fname = self.db.filenames[idx]
                fpath = self.cfg.img_dir / fname

                img_bgr = cv2.imread(str(fpath))
                if img_bgr is None:
                    continue

                for writer, target in writers:
                    # Crop to target resolution
                    crop = smart_crop(img_bgr, target.width, target.height)

                    # Apply color grading (progressive, frame-to-frame)
                    if self.color_grader and frame_idx > 0:
                        prev_frame = prev_frames.get(target.filename)
                        if prev_frame is not None:
                            crop = self.color_grader.match_colors(
                                source=crop, target=prev_frame, prev_frame=prev_frame
                            )

                    # Store for next iteration
                    prev_frames[target.filename] = crop.copy()

                    # Convert to RGB and write
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    writer.append_data(crop_rgb)

        print("[Render] All videos complete.")


def render_videos(cfg: Config, db: ImageDatabase, path_indices: list[int]) -> None:
    """Convenience function for rendering videos.

    Args:
        cfg: Configuration object
        db: Image database
        path_indices: List of image indices forming the path
    """
    renderer = VideoRenderer(cfg, db)
    renderer.render_videos(path_indices)
