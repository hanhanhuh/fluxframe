#!/usr/bin/env python3
"""CLI interface for FluxFrame."""

import argparse
import logging
import sys
from pathlib import Path

from .processor import VideoImageMatcher


def add_match_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for match command."""
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("images", type=str, help="Path to folder containing images")
    parser.add_argument("output", type=str, help="Output directory")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top similar images")
    parser.add_argument("--edge-weight", type=float, default=0.33, help="Edge similarity weight")
    parser.add_argument("--texture-weight", type=float, default=0.33, help="Texture similarity weight")
    parser.add_argument("--color-weight", type=float, default=0.34, help="Color similarity weight")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum similarity threshold")
    parser.add_argument("--no-repeat", action="store_true", help="Use each image only once")
    parser.add_argument("--comparison-size", type=int, default=256, help="Resize for comparison")
    parser.add_argument("--skip-output", action="store_true", help="Skip output generation")
    parser.add_argument("--demo", action="store_true", help="Demo mode")
    parser.add_argument("--demo-seconds", type=int, default=20, help="Seconds to process in demo")
    parser.add_argument("--demo-images", type=int, default=1000, help="Images to use in demo")
    parser.add_argument("--checkpoint-batch", type=int, default=10, help="Checkpoint frequency")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--fps-override", type=float, default=None, help="Override FPS")
    parser.add_argument("--feature-method", type=str, default="canny",
                       choices=["canny", "spatial_pyramid", "hog", "mobilenet", "efficientnet",
                               "spatial_color", "gist"],
                       help="Feature extraction method")
    parser.add_argument("--pooling-method", type=str, default="avg", choices=["avg", "gem"],
                       help="Spatial pooling method")
    parser.add_argument("--gem-p", type=float, default=3.0, help="GeM pooling power")
    parser.add_argument("--spatial-grid", type=int, default=2, choices=[2, 3], help="Grid size")
    parser.add_argument("--use-global-pooling", action="store_true", help="Use global pooling")
    parser.add_argument("--force-mobilenet-export", action="store_true", help="Force ONNX export")
    parser.add_argument("--save-samples", type=int, default=0, help="Save comparison samples")
    parser.add_argument("--sample-interval", type=int, default=1, help="Sample interval")


def add_generate_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for generate command."""
    parser.add_argument("--dir", type=Path, required=True, help="Source image directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--formats", nargs="+", default=["1080x1920:shorts.mp4", "1920x1080:wide.mp4"],
                       help="Output formats as WIDTHxHEIGHT:filename")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--dur", type=int, default=10, help="Duration in seconds")
    parser.add_argument("--start-img", type=str, default=None, help="Starting image path")
    parser.add_argument("--weights", type=float, nargs=3, default=[1.0, 2.0, 2.0],
                       help="LAB channel weights")
    parser.add_argument("--smoothing", type=int, default=3, help="Smoothing window size")
    parser.add_argument("--allow-duplicates", action="store_true", help="Allow duplicate frames")
    parser.add_argument("--metric", type=str, choices=["lab", "ssim", "lab+ssim"], default="lab",
                       help="Distance metric")
    parser.add_argument("--ssim-weight", type=float, default=0.5, help="SSIM weight for hybrid")
    parser.add_argument("--color-grade", action="store_true", help="Enable color grading")
    parser.add_argument("--color-method", type=str, choices=["histogram", "color_transfer", "lut"],
                       default="histogram", help="Color grading method")
    parser.add_argument("--color-strength", type=float, default=0.7, help="Color grading strength")


def run_match(args: argparse.Namespace) -> None:
    """Run match command."""
    if not Path(args.video).exists():
        sys.exit(f"Error: Video file not found: {args.video}")
    if not Path(args.images).is_dir():
        sys.exit(f"Error: Image folder not found: {args.images}")

    matcher = VideoImageMatcher(
        video_path=args.video, image_folder=args.images, output_dir=args.output,
        top_n=args.top_n, edge_weight=args.edge_weight, texture_weight=args.texture_weight,
        color_weight=args.color_weight, similarity_threshold=args.threshold,
        no_repeat=args.no_repeat, comparison_size=args.comparison_size,
        demo_mode=args.demo, demo_seconds=args.demo_seconds, demo_images=args.demo_images,
        checkpoint_batch_size=args.checkpoint_batch, seed=args.seed, num_workers=args.num_workers,
        fps_override=args.fps_override, save_samples=args.save_samples,
        sample_interval=args.sample_interval, feature_method=args.feature_method,
        force_mobilenet_export=args.force_mobilenet_export, pooling_method=args.pooling_method,
        gem_p=args.gem_p, spatial_grid=args.spatial_grid, use_global_pooling=args.use_global_pooling,
    )
    checkpoint = matcher.process()
    if not args.skip_output:
        matcher.generate_output(checkpoint)


def run_generate(args: argparse.Namespace) -> None:
    """Run generate command."""
    from .generator.config import Config, RenderTarget
    from .generator.database import ImageDatabase
    from .generator.pathfinding import find_path
    from .generator.rendering import render_videos
    from .generator.search import SearchIndex

    def parse_target(s: str) -> RenderTarget:
        try:
            res, name = s.split(":")
            w, h = map(int, res.split("x"))
            return RenderTarget(w, h, name)
        except Exception:
            sys.exit(f"Error: Invalid format '{s}'")

    if not args.dir.exists():
        sys.exit(f"Error: Directory not found: {args.dir}")

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


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="FluxFrame - Video frame matching and generation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    match_parser = subparsers.add_parser("match", help="Match video frames to similar images")
    add_match_args(match_parser)

    generate_parser = subparsers.add_parser("generate", help="Generate smooth transition videos")
    add_generate_args(generate_parser)

    args = parser.parse_args()

    if args.command == "match":
        run_match(args)
    elif args.command == "generate":
        run_generate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
