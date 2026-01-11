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

    # Generator-specific arguments
    gen_group = parser.add_argument_group("video generation options")
    gen_group.add_argument("--formats", nargs="+", default=["1080x1920:shorts.mp4", "1920x1080:wide.mp4"],
                          help="Output formats as WIDTHxHEIGHT:filename")
    gen_group.add_argument("--fps", type=int, default=30, help="Frames per second")
    gen_group.add_argument("--dur", type=int, default=10, help="Duration in seconds")
    gen_group.add_argument("--start-img", type=str, default=None, help="Starting image path")
    gen_group.add_argument("--weights", type=float, nargs=3, default=[1.0, 2.0, 2.0],
                          help="LAB channel weights (L, A, B)")
    gen_group.add_argument("--smoothing", type=int, default=3, help="Smoothing window size")
    gen_group.add_argument("--allow-duplicates", action="store_true", help="Allow duplicate frames")
    gen_group.add_argument("--metric", type=str, choices=["lab", "ssim", "lab+ssim", "gist"], default="lab",
                          help="Distance metric (lab=fast color, ssim=structural, gist=scene layout)")
    gen_group.add_argument("--ssim-weight", type=float, default=0.5, help="SSIM weight for lab+ssim hybrid")
    gen_group.add_argument("--color-grade", action="store_true", help="Enable color grading")
    gen_group.add_argument("--color-method", type=str, choices=["histogram", "color_transfer", "lut"],
                          default="histogram", help="Color grading method")
    gen_group.add_argument("--color-strength", type=float, default=0.7, help="Color grading strength (0-1)")

    # Frame matching-specific arguments
    match_group = parser.add_argument_group("frame matching options (requires --video)")
    match_group.add_argument("--top-n", type=int, default=10, help="Number of top similar images")
    match_group.add_argument("--threshold", type=float, default=0.0, help="Minimum similarity threshold")
    match_group.add_argument("--no-repeat", action="store_true", help="Use each image only once")
    match_group.add_argument("--skip-output", action="store_true", help="Skip output generation")
    match_group.add_argument("--demo", action="store_true", help="Demo mode")
    match_group.add_argument("--demo-seconds", type=int, default=20, help="Seconds to process in demo")
    match_group.add_argument("--demo-images", type=int, default=1000, help="Images to use in demo")
    match_group.add_argument("--checkpoint-batch", type=int, default=10, help="Checkpoint frequency")
    match_group.add_argument("--seed", type=int, default=None, help="Random seed")
    match_group.add_argument("--fps-override", type=float, default=None, help="Override FPS")
    match_group.add_argument("--save-samples", type=int, default=0, help="Save comparison samples")
    match_group.add_argument("--sample-interval", type=int, default=1, help="Sample interval")

    # Comparison demo mode
    demo_group = parser.add_argument_group("comparison demo options")
    demo_group.add_argument("--comparison-demo", action="store_true",
                           help="Generate comparison grids for different settings")
    demo_group.add_argument("--demo-frames", type=int, default=5,
                           help="Number of frames/images to use in comparison demo")

    args = parser.parse_args()

    # Validate directory exists
    if not args.dir.exists():
        sys.exit(f"Error: Directory not found: {args.dir}")

    # Determine mode based on --video presence
    if args.video:
        # Frame matching mode
        if not Path(args.video).exists():
            sys.exit(f"Error: Video file not found: {args.video}")

        output_dir = str(args.out_dir) if args.out_dir else "./output"

        matcher = VideoFrameMatcher(
            video_path=args.video,
            image_folder=str(args.dir),
            output_dir=output_dir,
            metric=args.metric,
            top_n=args.top_n,
            threshold=args.threshold,
            no_repeat=args.no_repeat,
            demo_mode=args.demo,
            demo_seconds=args.demo_seconds,
            demo_images=args.demo_images,
            checkpoint_batch_size=args.checkpoint_batch,
            seed=args.seed,
            fps_override=args.fps_override,
            save_samples=args.save_samples,
            sample_interval=args.sample_interval,
            weights=tuple(args.weights),
            ssim_weight=args.ssim_weight,
        )
        checkpoint = matcher.process()
        if not args.skip_output:
            matcher.generate_output(checkpoint)
    elif args.comparison_demo:
        # Comparison demo mode
        from .demo import run_comparison_demo
        run_comparison_demo(args)
    else:
        # Video generation mode
        from .config import Config, RenderTarget
        from .database import ImageDatabase
        from .pathfinding import find_path
        from .rendering import render_videos
        from .search import SearchIndex

        def parse_target(s: str) -> RenderTarget:
            try:
                res, name = s.split(":")
                w, h = map(int, res.split("x"))
                return RenderTarget(w, h, name)
            except Exception:
                sys.exit(f"Error: Invalid format '{s}'")

        out_dir = args.out_dir if args.out_dir else Path.cwd()
        out_dir.mkdir(exist_ok=True, parents=True)

        cfg = Config(
            img_dir=args.dir, output_dir=out_dir, targets=[parse_target(s) for s in args.formats],
            start_filename=args.start_img, fps=args.fps, duration=args.dur,
            weights=tuple(args.weights), enforce_unique=not args.allow_duplicates,
            smoothing_k=args.smoothing, metric=args.metric, ssim_weight=args.ssim_weight,
            enable_color_grading=args.color_grade, color_grading_method=args.color_method,
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
