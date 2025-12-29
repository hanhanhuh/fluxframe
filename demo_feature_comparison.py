#!/usr/bin/env python3
"""Demo script to compare all feature extraction methods on sample video frames.

This script:
1. Extracts random frames from input video
2. Runs all feature methods (canny, spatial_pyramid, hog, mobilenet)
3. Finds top matches for each method
4. Creates side-by-side comparison images
5. Outputs performance benchmarks

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
        for i, (img_path, img, similarity) in enumerate(matches[method][:max_matches]):
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


def benchmark_method(
    matcher: ImageMatcher,
    frame: npt.NDArray[Any],
    num_runs: int = 10,
) -> tuple[float, npt.NDArray[np.float32]]:
    """Benchmark feature extraction time for a method.

    Args:
        matcher: ImageMatcher instance
        frame: Frame to extract features from
        num_runs: Number of runs for averaging

    Returns:
        Tuple of (avg_time_ms, feature_vector)
    """
    # Warmup
    features = matcher.compute_all_features(frame)
    vector = matcher.features_to_vector(features)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        features = matcher.compute_all_features(frame)
        vector = matcher.features_to_vector(features)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times), vector


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

    # Define methods to test
    methods: list[FeatureMethod] = ["canny", "spatial_pyramid", "hog"]

    # Try to add mobilenet if available
    try:
        test_matcher = ImageMatcher(feature_method="mobilenet")
        methods.append("mobilenet")
        print("✓ MobileNet available\n")
    except ImportError as e:
        print(f"⚠ MobileNet not available: {e}\n")

    # Build FAISS index for each method
    print("Building FAISS indices for each method...")
    print(f"Image folder: {args.images}\n")

    indices: dict[str, VideoImageMatcher] = {}
    benchmarks: dict[str, dict[str, Any]] = {}

    for method in tqdm(methods, desc="Building indices"):
        # Create processor with this method
        processor = VideoImageMatcher(
            video_path=args.video,
            image_folder=args.images,
            output_dir=str(output_dir / method),
            feature_method=method,
            top_n=args.top_k,
        )

        # Build index
        image_files = processor.get_image_files()
        processor._build_faiss_index(image_files)

        indices[method] = processor
        benchmarks[method] = {
            "vector_size": processor.faiss_index.d,
            "num_images": len(image_files),
        }

    print(f"\n✓ Built {len(indices)} indices\n")

    # Process each frame
    print(f"Processing {len(frames)} frames with {len(methods)} methods...\n")

    for frame_idx, (frame_num, frame) in enumerate(frames):
        print(f"\n{'─'*80}")
        print(f"Frame {frame_idx + 1}/{len(frames)} (frame #{frame_num})")
        print(f"{'─'*80}\n")

        matches_per_method: dict[str, list[tuple[str, npt.NDArray[Any], float]]] = {}
        timings: dict[str, float] = {}

        # Test each method
        for method in methods:
            processor = indices[method]
            matcher = processor.matcher

            # Benchmark feature extraction
            avg_time, _ = benchmark_method(matcher, frame)
            timings[method] = avg_time

            # Find matches
            top_matches = processor.find_top_matches(frame)

            # Load matched images
            matches = []
            for img_path, similarity in top_matches[:args.top_k]:
                img = cv2.imread(img_path)
                if img is not None:
                    matches.append((img_path, img, similarity))

            matches_per_method[method] = matches

            # Print results
            print(f"{method:20s} | Time: {avg_time:6.2f}ms | Top match: {top_matches[0][1]:.3f}")

        # Create comparison image
        comparison = create_comparison_image(frame, matches_per_method, methods, args.top_k)

        # Save comparison
        output_path = output_dir / f"frame_{frame_num:06d}_comparison.jpg"
        cv2.imwrite(str(output_path), comparison)
        print(f"\n✓ Saved comparison: {output_path}")

    # Print summary statistics
    print(f"\n\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}\n")

    print(f"{'Method':<20} {'Vector Size':<15} {'Avg Time (ms)':<15} {'Images':<10}")
    print(f"{'-'*70}")

    for method in methods:
        vector_size = benchmarks[method]["vector_size"]
        num_images = benchmarks[method]["num_images"]

        # Calculate average time across all frames
        avg_times = []
        for frame_idx, (frame_num, frame) in enumerate(frames):
            processor = indices[method]
            avg_time, _ = benchmark_method(processor.matcher, frame, num_runs=3)
            avg_times.append(avg_time)

        overall_avg = np.mean(avg_times)

        print(f"{method:<20} {vector_size:<15d} {overall_avg:<15.2f} {num_images:<10d}")

    print(f"\n✓ Demo complete! Check output directory: {output_dir}\n")


if __name__ == "__main__":
    main()
