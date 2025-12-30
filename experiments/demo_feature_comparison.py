#!/usr/bin/env python3
"""Demo script to compare all feature extraction methods on sample video frames.

This script:
1. Extracts random frames from input video
2. Runs all feature methods (canny, spatial_pyramid, hog, mobilenet)
3. Finds top matches for each method
4. Creates side-by-side comparison images showing match quality

Usage:
    python demo_feature_comparison.py <video_path> <image_folder> [--output demo_output]
"""

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.fluxframe.matcher import FeatureMethod, ImageMatcher
from src.fluxframe.processor import VideoImageMatcher


def extract_random_frames(
    video_path: str, num_frames: int = 5, seed: int = 42
) -> list[tuple[int, npt.NDArray[Any]]]:
    """Extract random frames from video.

    Args:
        video_path: Path to video file
        num_frames: Number of random frames to extract
        seed: Random seed for reproducibility

    Returns:
        List of (frame_number, frame_image) tuples
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        raise ValueError(f"Could not read video: {video_path}")

    # Select random frame indices
    np.random.seed(seed)
    frame_indices = np.random.choice(
        total_frames, size=min(num_frames, total_frames), replace=False
    )
    frame_indices.sort()

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append((int(frame_idx), frame))

    cap.release()
    return frames


def create_comparison_image(
    query_frame: npt.NDArray[Any],
    matches: dict[str, list[tuple[str, npt.NDArray[Any], float]]],
    method_order: list[str],
    max_matches: int = 3,
) -> npt.NDArray[Any]:
    """Create side-by-side comparison grid showing query and top matches per method.

    Layout:
    [Query] [Method1 Match1] [Method1 Match2] [Method1 Match3]
    [Query] [Method2 Match1] [Method2 Match2] [Method2 Match3]
    ...

    Args:
        query_frame: Original video frame
        matches: Dict mapping method name to list of (path, image, similarity)
        method_order: Order to display methods
        max_matches: Number of matches to show per method

    Returns:
        Comparison grid image
    """
    # Resize query frame to standard size
    img_h, img_w = 200, 300
    query_resized = cv2.resize(query_frame, (img_w, img_h))

    # Add label
    label_h = 30
    rows = []

    for method in method_order:
        if method not in matches:
            continue

        row_images = []

        # Query image with label
        query_labeled = np.zeros((img_h + label_h, img_w, 3), dtype=np.uint8)
        query_labeled[label_h:, :] = query_resized
        cv2.putText(
            query_labeled,
            "Query Frame",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        row_images.append(query_labeled)

        # Top matches with similarity scores
        for i, (_img_path, img, similarity) in enumerate(matches[method][:max_matches]):
            match_resized = cv2.resize(img, (img_w, img_h))
            match_labeled = np.zeros((img_h + label_h, img_w, 3), dtype=np.uint8)
            match_labeled[label_h:, :] = match_resized

            # Label with method name and similarity
            label = f"{method} #{i+1}: {similarity:.3f}"
            cv2.putText(
                match_labeled,
                label,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
            row_images.append(match_labeled)

        # Concatenate row
        rows.append(np.hstack(row_images))

    # Stack all rows
    return np.vstack(rows)


def main() -> None:
    """Run feature comparison demo."""
    parser = argparse.ArgumentParser(
        description="Compare feature extraction methods on video frames"
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("images", help="Path to folder containing images")
    parser.add_argument(
        "--output", default="demo_output", help="Output directory for comparisons"
    )
    parser.add_argument(
        "--frames", type=int, default=5, help="Number of random frames to extract"
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of top matches to show"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--force-mobilenet-export", action="store_true",
        help="Force re-export of MobileNet ONNX model"
    )
    parser.add_argument(
        "--demo-images", type=int, default=None,
        help="Use only a subset of images (default: use all images)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("Feature Extraction Method Comparison Demo")
    print(f"{'='*80}\n")

    # Extract random frames
    print(f"Extracting {args.frames} random frames from: {args.video}")
    frames = extract_random_frames(args.video, args.frames, args.seed)
    print(f"✓ Extracted {len(frames)} frames\n")

    # Define configurations to test: (name, feature_method, pooling_method, gem_p, spatial_grid, edge_wt, texture_wt, color_wt)
    configs: list[tuple[str, str, str, float, int, float, float, float]] = [
        # Classical methods - test different weight combinations
        ("canny", "canny", "avg", 3.0, 2, 0.33, 0.33, 0.34),
        ("canny_edge", "canny", "avg", 3.0, 2, 0.7, 0.15, 0.15),
        ("spatial_pyr", "spatial_pyramid", "avg", 3.0, 2, 0.33, 0.33, 0.34),
        ("spatial_pyr_struct", "spatial_pyramid", "avg", 3.0, 2, 0.6, 0.2, 0.2),
        ("hog", "hog", "avg", 3.0, 2, 0.33, 0.33, 0.34),
        ("hog_motion", "hog", "avg", 3.0, 2, 0.7, 0.15, 0.15),
    ]

    # Try to add neural methods if available (neural methods use edge_wt=1.0, texture/color not used)
    try:
        ImageMatcher(
            feature_method="mobilenet",
            force_mobilenet_export=args.force_mobilenet_export
        )
        # Add recommended neural configurations
        configs.extend([
            ("mobilenet+avg", "mobilenet", "avg", 3.0, 2, 1.0, 0.0, 0.0),
            ("mobilenet+gem", "mobilenet", "gem", 3.0, 2, 1.0, 0.0, 0.0),
            ("mobilenet+gem3x3", "mobilenet", "gem", 3.0, 3, 1.0, 0.0, 0.0),
        ])
        print("✓ MobileNet available\n")
    except ImportError as e:
        print(f"⚠ MobileNet not available: {e}\n")

    try:
        ImageMatcher(
            feature_method="efficientnet",
            force_mobilenet_export=args.force_mobilenet_export
        )
        # Add EfficientNet configurations
        configs.extend([
            ("efficientnet+gem", "efficientnet", "gem", 3.0, 2, 1.0, 0.0, 0.0),
            ("efficientnet+gem3x3", "efficientnet", "gem", 3.0, 3, 1.0, 0.0, 0.0),
        ])
        print("✓ EfficientNet available\n")
    except ImportError as e:
        print(f"⚠ EfficientNet not available: {e}\n")

    # Build FAISS index for each configuration
    print(f"Building FAISS indices for {len(configs)} configurations...")
    print(f"Image folder: {args.images}\n")

    indices: dict[str, VideoImageMatcher] = {}
    benchmarks: dict[str, dict[str, Any]] = {}

    for name, feature_method, pooling_method, gem_p, spatial_grid, edge_wt, texture_wt, color_wt in tqdm(configs, desc="Building indices"):
        # Create processor with this configuration
        t_start = time.perf_counter()
        processor = VideoImageMatcher(
            video_path=args.video,
            image_folder=args.images,
            output_dir=str(output_dir / name),
            feature_method=feature_method,  # type: ignore
            top_n=args.top_k,
            edge_weight=edge_wt,
            texture_weight=texture_wt,
            color_weight=color_wt,
            force_mobilenet_export=args.force_mobilenet_export,
            pooling_method=pooling_method,  # type: ignore
            gem_p=gem_p,
            spatial_grid=spatial_grid,
        )

        # Build index
        image_files = processor.get_image_files()

        # Use subset if specified
        if args.demo_images is not None and len(image_files) > args.demo_images:
            np.random.seed(args.seed)
            subset_indices = np.random.choice(
                len(image_files), size=args.demo_images, replace=False
            )
            image_files = [image_files[i] for i in sorted(subset_indices)]

        processor._build_faiss_index(image_files)
        t_build = time.perf_counter() - t_start

        indices[name] = processor
        benchmarks[name] = {
            "vector_size": processor.faiss_index.d,
            "num_images": len(image_files),
            "config": f"{feature_method}/{pooling_method}/grid{spatial_grid}",
            "weights": f"e{edge_wt:.2f}/t{texture_wt:.2f}/c{color_wt:.2f}",
            "build_time": t_build,
        }

    print(f"\n✓ Built {len(indices)} indices\n")

    # Process each frame
    config_names = list(indices.keys())
    print(f"Processing {len(frames)} frames with {len(config_names)} configurations...\n")

    # Track per-frame matching times
    match_times: dict[str, list[float]] = {name: [] for name in config_names}

    for frame_idx, (frame_num, frame) in enumerate(frames):
        print(f"\n{'─'*80}")
        print(f"Frame {frame_idx + 1}/{len(frames)} (frame #{frame_num})")
        print(f"{'─'*80}\n")

        matches_per_config: dict[str, list[tuple[str, npt.NDArray[Any], float]]] = {}

        # Test each configuration
        for config_name in config_names:
            processor = indices[config_name]

            # Find matches (needs frame_num and aspect_ratio)
            h, w = frame.shape[:2]
            aspect_ratio = w / h

            t_start = time.perf_counter()
            top_matches = processor.find_top_matches(frame, frame_num, aspect_ratio)
            t_match = time.perf_counter() - t_start
            match_times[config_name].append(t_match)

            # Load matched images
            matches = []
            for img_path, similarity in top_matches[:args.top_k]:
                img = cv2.imread(img_path)
                if img is not None:
                    matches.append((img_path, img, similarity))

            matches_per_config[config_name] = matches

            # Print results
            print(f"{config_name:25s} | Top: {top_matches[0][1]:.3f} | {t_match*1000:6.1f}ms")

        # Create comparison image
        comparison = create_comparison_image(frame, matches_per_config, config_names, args.top_k)

        # Save comparison
        output_path = output_dir / f"frame_{frame_num:06d}_comparison.jpg"
        cv2.imwrite(str(output_path), comparison)
        print(f"\n✓ Saved comparison: {output_path}")

    # Print summary statistics
    print(f"\n\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}\n")

    print(f"{'Configuration':<25} {'Vec Size':<10} {'Build (s)':<12} {'Match (ms)':<12} {'Weights':<20}")
    print(f"{'-'*95}")

    for config_name in config_names:
        vector_size = benchmarks[config_name]["vector_size"]
        build_time = benchmarks[config_name]["build_time"]
        avg_match_time = np.mean(match_times[config_name]) * 1000  # Convert to ms
        weights = benchmarks[config_name]["weights"]
        print(f"{config_name:<25} {vector_size:<10d} {build_time:<12.2f} {avg_match_time:<12.1f} {weights:<20}")

    print(f"\n✓ Demo complete! Check output directory: {output_dir}\n")


if __name__ == "__main__":
    main()
