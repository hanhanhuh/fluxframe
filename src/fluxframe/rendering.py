#!/usr/bin/env python3
"""Video rendering module with smart cropping and optional color grading."""

from __future__ import annotations

from contextlib import ExitStack

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from .color_grading import ColorGrader
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

        # Create color graders for each method
        self.color_graders: dict[str, ColorGrader] = {}
        for method in cfg.color_grading_methods or []:
            grader_cfg = Config(
                img_dir=cfg.img_dir,
                output_dir=cfg.output_dir,
                color_grading_method=method,
                color_grading_strength=cfg.color_grading_strength,
                enable_color_grading=True,
            )
            self.color_graders[method] = ColorGrader(grader_cfg)

    def render_videos(self, path_indices: list[int]) -> None:
        """Render all target formats simultaneously with multiple color grading variants.

        For each target, generates:
        - Base version (no color grading)
        - One version per color grading method specified in config

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

        # Calculate total number of output videos
        variants_per_target = 1 + len(self.color_graders)  # base + graded versions
        total_outputs = len(self.cfg.targets) * variants_per_target

        print(
            f"[Render] Starting rendering for {len(self.cfg.targets)} formats "
            f"x {variants_per_target} variants = {total_outputs} videos "
            f"({len(path_indices)} frames)..."
        )
        print(f"[Render] Output location: {self.cfg.output_dir.absolute()}")

        if self.color_graders:
            methods_str = ", ".join(self.color_graders.keys())
            strength = self.cfg.color_grading_strength
            print(f"[Render] Color grading methods: {methods_str} (strength={strength})")
            print("[Render] Plus base (non-graded) version for each format")

        # Open all video writers
        with ExitStack() as stack:
            # Structure: [(writer, target, grading_method), ...]  # noqa: ERA001
            # grading_method is None for base version, or method name for graded
            writers = []

            for t in self.cfg.targets:
                # Base version (no color grading)
                base_filename = t.filename
                out_path = self.cfg.output_dir / base_filename
                print(f"  -> {base_filename} ({t.width}x{t.height}) [base]")

                w = stack.enter_context(
                    imageio.get_writer(  # type: ignore[arg-type]
                        str(out_path),
                        format="FFMPEG",  # type: ignore[arg-type]
                        mode="I",
                        fps=self.cfg.fps,
                        codec="libx264",
                        pixelformat="yuv420p",
                        output_params=["-crf", "18", "-preset", "medium"],
                        macro_block_size=None,
                    )
                )
                writers.append((w, t, None))  # None = no color grading

                # Color graded versions
                for method in self.color_graders:
                    # Insert method name before extension
                    stem = t.filename.rsplit(".", 1)[0] if "." in t.filename else t.filename
                    ext = t.filename.rsplit(".", 1)[1] if "." in t.filename else "mp4"
                    graded_filename = f"{stem}_{method}.{ext}"
                    out_path = self.cfg.output_dir / graded_filename
                    print(f"  -> {graded_filename} ({t.width}x{t.height}) [{method}]")

                    w = stack.enter_context(
                        imageio.get_writer(  # type: ignore[arg-type]
                            str(out_path),
                            format="FFMPEG",  # type: ignore[arg-type]
                            mode="I",
                            fps=self.cfg.fps,
                            codec="libx264",
                            pixelformat="yuv420p",
                            output_params=["-crf", "18", "-preset", "medium"],
                            macro_block_size=None,
                        )
                    )
                    writers.append((w, t, method))  # type: ignore[arg-type]

            # Render frames
            # Track previous frames per (target.filename, method) for temporal smoothing
            prev_frames: dict[tuple[str, str | None], np.ndarray] = {}

            for frame_idx, idx in enumerate(tqdm(path_indices, desc="Encoding")):
                fname = self.db.filenames[idx]
                fpath = self.cfg.img_dir / fname

                img_bgr = cv2.imread(str(fpath))
                if img_bgr is None:
                    continue

                for writer, target, method in writers:  # type: ignore
                    # Crop to target resolution
                    crop = smart_crop(img_bgr, target.width, target.height)

                    # Apply color grading if method specified
                    if method is not None and frame_idx > 0:
                        key = (target.filename, method)
                        prev_frame = prev_frames.get(key)
                        if prev_frame is not None:
                            grader = self.color_graders[method]
                            crop = grader.match_colors(
                                source=crop, target=prev_frame, prev_frame=prev_frame
                            )

                    # Store for next iteration
                    key = (target.filename, method)  # type: ignore[assignment]
                    prev_frames[key] = crop.copy()

                    # Convert to RGB and write
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    writer.append_data(crop_rgb)  # type: ignore[attr-defined]

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
