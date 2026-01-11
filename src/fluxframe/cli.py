#!/usr/bin/env python3
"""CLI interface for FluxFrame."""

import argparse
import logging
import sys
from pathlib import Path

from .frame_matching import VideoFrameMatcher


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    parser = argparse.ArgumentParser(
        description="FluxFrame - Smooth video generation and frame matching",
        epilog="Mode auto-detection: Provide --video for frame matching mode, "
               "or --dir only for video generation mode."
    )

    # Common arguments
    parser.add_argument("--dir", type=Path, required=True, help="Source image directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")

    # Video input (optional - triggers frame matching mode)
    parser.add_argument("--video", type=str, default=None,
                       help="Input video file (enables frame matching mode)")

    # Metric configuration (shared)
    metric_group = parser.add_argument_group("metric options")
    metric_group.add_argument(
        "--metric",
        type=str,
        choices=["lab", "ssim", "lab+ssim", "gist"],
        default="lab",
        help="Distance metric (lab=fast color, ssim=structural, gist=scene)",
    )
    metric_group.add_argument(
        "--weights", type=float, nargs=3, default=[1.0, 2.0, 2.0], help="LAB weights (L, A, B)"
    )
    metric_group.add_argument(
        "--ssim-weight", type=float, default=0.5, help="SSIM weight for lab+ssim"
    )
    metric_group.add_argument(
        "--threshold", type=float, default=0.0, help="Similarity threshold"
    )
    metric_group.add_argument(
        "--seed", type=int, default=None, help="Random seed"
    )

    # Output configuration (shared)
    output_group = parser.add_argument_group("output options")
    output_group.add_argument("--fps", type=int, default=30, help="Output FPS")
    output_group.add_argument(
        "--no-repeat", action="store_true", help="Use each image only once"
    )
    output_group.add_argument(
        "--color-grade",
        nargs="*",
        metavar="METHOD",
        choices=["histogram", "color_transfer", "lut"],
        help="Color grading methods. Generates base + one per method.",
    )
    output_group.add_argument(
        "--color-strength", type=float, default=0.7, help="Color strength (0-1)"
    )

    # Legacy support (deprecated)
    output_group.add_argument(
        "--color-method",
        type=str,
        choices=["histogram", "color_transfer", "lut"],
        default="histogram",
        help="[DEPRECATED] Use --color-grade",
    )

    # Generation-specific options
    gen_group = parser.add_argument_group("generation options (without --video)")
    gen_group.add_argument(
        "--formats",
        nargs="+",
        default=["1080x1920:shorts.mp4", "1920x1080:wide.mp4"],
        help="Output formats as WIDTHxHEIGHT:filename",
    )
    gen_group.add_argument("--dur", type=int, default=10, help="Duration in seconds")
    gen_group.add_argument(
        "--start-img", type=str, default=None, help="Starting image path"
    )
    gen_group.add_argument(
        "--smoothing", type=int, default=3, help="Smoothness (1=greedy, 3-5=smooth)"
    )

    # Matching-specific options
    match_group = parser.add_argument_group("matching options (with --video)")
    match_group.add_argument(
        "--fps-override", type=float, default=None, help="Override input FPS"
    )
    match_group.add_argument(
        "--checkpoint-batch", type=int, default=10, help="Save every N frames"
    )
    match_group.add_argument(
        "--skip-output", action="store_true", help="Skip output generation"
    )

    # Debug options
    debug_group = parser.add_argument_group("debug options")
    debug_group.add_argument(
        "--top-n", type=int, default=10, help="Search depth (candidates)"
    )
    debug_group.add_argument(
        "--demo-limit", type=int, default=None, help="Limit for quick testing"
    )
    debug_group.add_argument(
        "--save-samples", type=int, default=0, help="Save comparison samples"
    )
    debug_group.add_argument(
        "--comparison-demo",
        action="store_true",
        help="Generate metric comparison grids",
    )

    args = parser.parse_args()

    # Validate directory exists
    if not args.dir.exists():
        sys.exit(f"Error: Directory not found: {args.dir}")

    # Determine mode based on --video presence
    if args.video:
        # Frame matching mode
        if not Path(args.video).exists():
            sys.exit(f"Error: Video file not found: {args.video}")

        from .config import Config  # noqa: PLC0415

        out_dir = args.out_dir if args.out_dir else Path.cwd() / "output"
        out_dir.mkdir(exist_ok=True, parents=True)

        # Parse color grading methods
        if args.color_grade is not None:
            if len(args.color_grade) == 0:
                # --color-grade without args = all methods
                color_methods = ["histogram", "color_transfer", "lut"]
            else:
                color_methods = args.color_grade
        else:
            color_methods = []

        cfg = Config(
            img_dir=args.dir,
            output_dir=out_dir,
            video_path=Path(args.video),
            targets=[],  # Not generating
            metric=args.metric,
            weights=tuple(args.weights),
            ssim_weight=args.ssim_weight,
            threshold=args.threshold,
            top_n=args.top_n,
            enforce_unique=args.no_repeat,
            demo_mode=args.demo_limit is not None,
            demo_seconds=args.demo_limit or 20,
            demo_images=args.demo_limit or 1000,
            checkpoint_batch_size=args.checkpoint_batch,
            seed=args.seed,
            fps_override=args.fps_override,
            save_samples=args.save_samples,
            sample_interval=1,
            color_grading_methods=color_methods,
            color_grading_strength=args.color_strength,
        )

        try:
            cfg.validate()
        except ValueError as e:
            sys.exit(f"Configuration error: {e}")

        matcher = VideoFrameMatcher(cfg)
        checkpoint = matcher.process()
        if not args.skip_output:
            matcher.generate_output(checkpoint)
    elif args.comparison_demo:
        # Comparison demo mode
        from .demo import run_comparison_demo  # noqa: PLC0415
        run_comparison_demo(args)
    else:
        # Video generation mode
        from .config import Config, RenderTarget  # noqa: PLC0415
        from .database import ImageDatabase  # noqa: PLC0415
        from .pathfinding import find_path  # noqa: PLC0415
        from .rendering import render_videos  # noqa: PLC0415
        from .search import SearchIndex  # noqa: PLC0415

        def parse_target(s: str) -> RenderTarget:
            try:
                res, name = s.split(":")
                w, h = map(int, res.split("x"))
                return RenderTarget(w, h, name)
            except Exception:
                sys.exit(f"Error: Invalid format '{s}'")

        out_dir = args.out_dir if args.out_dir else Path.cwd()
        out_dir.mkdir(exist_ok=True, parents=True)

        # Parse color grading methods
        if args.color_grade is not None:
            if len(args.color_grade) == 0:
                # --color-grade without args = all methods
                color_methods = ["histogram", "color_transfer", "lut"]
            else:
                color_methods = args.color_grade
        else:
            color_methods = []

        cfg = Config(
            img_dir=args.dir, output_dir=out_dir, targets=[parse_target(s) for s in args.formats],
            start_filename=args.start_img, fps=args.fps, duration=args.dur,
            weights=tuple(args.weights), enforce_unique=args.no_repeat,
            smoothing_k=args.smoothing, metric=args.metric, ssim_weight=args.ssim_weight,
            color_grading_methods=color_methods,
            color_grading_strength=args.color_strength,
        )

        try:
            cfg.validate()
        except ValueError as e:
            sys.exit(f"Configuration error: {e}")

        print(f"\nFluxFrame Generator - {cfg.total_frames} frames @ {cfg.fps}fps\n")
        db = ImageDatabase(cfg)
        idx = SearchIndex(cfg, db)
        path = find_path(cfg, db, idx)
        if len(path) < cfg.total_frames:
            print(f"Warning: Only {len(path)} frames found")
        render_videos(cfg, db, path)
        print("\nComplete!")


if __name__ == "__main__":
    main()
