#!/usr/bin/env python3
"""CLI interface for video-image-matcher."""

import argparse
import logging
import sys
from pathlib import Path

from .processor import VideoImageMatcher


def main() -> None:
    """CLI entry point for video-image-matcher.

    Parses command-line arguments and runs the matching pipeline.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )

    parser = argparse.ArgumentParser(
        description="Match video frames to similar images from a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("images", type=str, help="Path to folder containing images")
    parser.add_argument("output", type=str, help="Output directory")

    parser.add_argument("--top-n", type=int, default=10,
                       help="Number of top similar images to consider")
    parser.add_argument("--edge-weight", type=float, default=0.33,
                       help="Weight for edge/contour similarity (0-1)")
    parser.add_argument("--texture-weight", type=float, default=0.33,
                       help="Weight for texture similarity (0-1)")
    parser.add_argument("--color-weight", type=float, default=0.34,
                       help="Weight for color similarity (0-1)")
    parser.add_argument("--threshold", type=float, default=0.0,
                       help="Minimum similarity threshold for selection (0-1)")
    parser.add_argument("--no-repeat", action="store_true",
                       help="Use each image only once")
    parser.add_argument("--comparison-size", type=int, default=256,
                       help="Resize images to this size for comparison "
                            "(larger = slower but more accurate)")
    parser.add_argument("--skip-output", action="store_true",
                       help="Skip output generation (only compute matches)")
    parser.add_argument("--demo", action="store_true",
                       help="Demo mode: process only a subset of video and images")
    parser.add_argument("--demo-seconds", type=int, default=20,
                       help="Number of seconds to process in demo mode")
    parser.add_argument("--demo-images", type=int, default=1000,
                       help="Number of images to use in demo mode")
    parser.add_argument("--checkpoint-batch", type=int, default=10,
                       help="Save checkpoint every N frames")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of parallel workers (default: auto-detect CPU count)")
    parser.add_argument("--fps-override", type=float, default=None,
                       help="Override output video FPS (default: use input video FPS)")
    parser.add_argument("--feature-method", type=str, default="canny",
                       choices=["canny", "spatial_pyramid", "hog"],
                       help="Edge/structure feature extraction method. "
                            "canny=fast/no spatial info (original), "
                            "spatial_pyramid=medium/preserves layout, "
                            "hog=slower/best motion preservation")
    parser.add_argument("--save-samples", type=int, default=0,
                       help="Number of frame-match comparison samples to save (0=disabled)")
    parser.add_argument("--sample-interval", type=int, default=1,
                       help="Save every Nth frame as sample (1=every frame)")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    if not Path(args.images).is_dir():
        print(f"Error: Image folder not found: {args.images}")
        sys.exit(1)

    # Create matcher and process
    matcher = VideoImageMatcher(
        video_path=args.video,
        image_folder=args.images,
        output_dir=args.output,
        top_n=args.top_n,
        edge_weight=args.edge_weight,
        texture_weight=args.texture_weight,
        color_weight=args.color_weight,
        similarity_threshold=args.threshold,
        no_repeat=args.no_repeat,
        comparison_size=args.comparison_size,
        demo_mode=args.demo,
        demo_seconds=args.demo_seconds,
        demo_images=args.demo_images,
        checkpoint_batch_size=args.checkpoint_batch,
        seed=args.seed,
        num_workers=args.num_workers,
        fps_override=args.fps_override,
        save_samples=args.save_samples,
        sample_interval=args.sample_interval,
        feature_method=args.feature_method
    )

    # Process frames and find matches
    checkpoint = matcher.process()

    # Generate output
    if not args.skip_output:
        matcher.generate_output(checkpoint)

    print("\nDone!")


if __name__ == "__main__":
    main()
