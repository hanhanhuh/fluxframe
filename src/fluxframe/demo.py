#!/usr/bin/env python3
"""Comparison demo mode for visualizing different FluxFrame settings."""
# mypy: disable-error-code="arg-type,operator,attr-defined"

from __future__ import annotations

import argparse

import cv2
import numpy as np

from .color_grading import ColorGrader
from .config import Config
from .database import ImageDatabase
from .metrics import create_metric
from .search import SearchIndex


def create_comparison_grid(
    images: list[np.ndarray],
    labels: list[str],
    cols: int = 3
) -> np.ndarray:
    """Create a grid of images with labels.

    Args:
        images: List of images (BGR format)
        labels: List of labels for each image
        cols: Number of columns in grid

    Returns:
        Grid image with labels
    """
    if not images:
        msg = "No images provided"
        raise ValueError(msg)

    # Calculate grid dimensions
    n_images = len(images)
    rows = (n_images + cols - 1) // cols

    # Get max dimensions
    h, w = images[0].shape[:2]

    # Label height
    label_h = 40
    total_h = h + label_h

    # Create grid canvas
    grid = np.ones((rows * total_h, cols * w, 3), dtype=np.uint8) * 255

    # Place images and labels
    for idx, (img, label) in enumerate(zip(images, labels, strict=False)):
        row = idx // cols
        col = idx % cols

        # Resize image if needed
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))  # noqa: PLW2901

        # Place image
        y_start = row * total_h
        x_start = col * w
        grid[y_start:y_start + h, x_start:x_start + w] = img

        # Add label
        label_img = np.ones((label_h, w, 3), dtype=np.uint8) * 240
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (label_h + text_size[1]) // 2
        cv2.putText(label_img, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        grid[y_start + h:y_start + total_h, x_start:x_start + w] = label_img

    return grid


def run_comparison_demo(args: argparse.Namespace) -> None:
    """Run comparison demo mode.

    Generates comparison grids showing:
    - Different metrics (LAB, SSIM, LAB+SSIM)
    - Different color grading methods (histogram, color_transfer, lut)
    - Different LAB weights

    Args:
        args: Parsed command line arguments
    """
    print("\n=== FluxFrame Comparison Demo ===\n")

    # Load images
    img_dir = args.dir
    image_files = sorted([
        f for f in img_dir.glob("*")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])[:args.demo_frames]

    if len(image_files) < 2:
        print("Error: Need at least 2 images for comparison")
        return

    print(f"Loaded {len(image_files)} images from {img_dir}")

    # Output directory
    out_dir = args.out_dir if args.out_dir else img_dir / "comparison_demo"
    out_dir.mkdir(exist_ok=True, parents=True)

    # === Demo 1: Metric Comparison ===
    print("\n1. Comparing distance metrics...")
    metrics_to_test = ["lab", "ssim", "lab+ssim"]

    for metric_name in metrics_to_test:
        print(f"   - Testing {metric_name}")

        # Create minimal config
        cfg = Config(
            img_dir=img_dir,
            output_dir=out_dir,
            targets=[],
            fps=1,
            duration=len(image_files),
            metric=metric_name,
            weights=(1.0, 2.0, 2.0),
            ssim_weight=0.5
        )

        # Create database and index
        db = ImageDatabase(cfg)
        _ = SearchIndex(cfg, db)  # Build index
        metric_fn = create_metric(cfg)

        # Find nearest neighbors for first image
        query_idx = 0
        distances = []
        for i in range(len(db)):
            if i != query_idx:
                dist = metric_fn(db.lab_images[query_idx], db.lab_images[i])
                distances.append((i, dist))

        # Get top 5 nearest neighbors
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:5]

        # Load and create grid
        images = [cv2.imread(str(img_dir / db.filenames[query_idx]))]
        labels = ["Query"]

        for idx_img, dist in neighbors:
            img = cv2.imread(str(img_dir / db.filenames[idx_img]))
            images.append(img)
            labels.append(f"d={dist:.3f}")

        grid = create_comparison_grid(images, labels, cols=3)
        output_path = out_dir / f"metric_{metric_name}.jpg"
        cv2.imwrite(str(output_path), grid)

    print(f"   Saved metric comparisons to {out_dir}")

    # === Demo 2: Color Grading Comparison ===
    print("\n2. Comparing color grading methods...")

    if len(image_files) >= 2:
        # Load two consecutive images
        img1 = cv2.imread(str(image_files[0]))
        img2 = cv2.imread(str(image_files[1]))

        # Resize to consistent size
        h, w = 720, 1280
        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))

        methods = ["histogram", "color_transfer", "lut"]
        images = [img1, img2]
        labels = ["Source", "Target"]

        for method in methods:
            print(f"   - Testing {method}")

            # Create dummy config
            cfg = Config(
                img_dir=img_dir,
                output_dir=out_dir,
                targets=[],
                fps=1,
                duration=2,
                enable_color_grading=True,
                color_grading_method=method,
                color_grading_strength=0.7
            )

            # Apply color grading
            grader = ColorGrader(cfg)
            graded = grader.match_colors(img1, img2)

            images.append(graded)
            labels.append(f"{method}")

        grid = create_comparison_grid(images, labels, cols=3)
        output_path = out_dir / "color_grading_comparison.jpg"
        cv2.imwrite(str(output_path), grid)
        print(f"   Saved color grading comparison to {output_path}")

    # === Demo 3: LAB Weights Comparison ===
    print("\n3. Comparing LAB channel weights...")

    weight_configs = [
        ((1.0, 1.0, 1.0), "Equal"),
        ((1.0, 2.0, 2.0), "Default (1,2,2)"),
        ((2.0, 1.0, 1.0), "Lightness (2,1,1)"),
        ((1.0, 3.0, 3.0), "Color (1,3,3)"),
    ]

    for weights, label in weight_configs:
        print(f"   - Testing {label}: {weights}")

        cfg = Config(
            img_dir=img_dir,
            output_dir=out_dir,
            targets=[],
            fps=1,
            duration=len(image_files),
            metric="lab",
            weights=weights
        )

        db = ImageDatabase(cfg)
        _ = SearchIndex(cfg, db)  # Build index
        metric_fn = create_metric(cfg)

        # Find nearest neighbors
        query_idx = 0
        distances = []
        for i in range(len(db)):
            if i != query_idx:
                dist = metric_fn(db.lab_images[query_idx], db.lab_images[i])
                distances.append((i, dist))

        distances.sort(key=lambda x: x[1])
        neighbors = distances[:3]

        # Create mini grid for this weight config
        images = []
        for idx_img, _dist in neighbors:
            img = cv2.imread(str(img_dir / db.filenames[idx_img]))
            img = cv2.resize(img, (320, 180))
            images.append(img)

        # Save individual result
        if images:
            combined = np.hstack(images)
            output_path = out_dir / f"weights_{label.replace(' ', '_').replace(',', '_')}.jpg"
            cv2.imwrite(str(output_path), combined)

    print(f"   Saved LAB weight comparisons to {out_dir}")

    print(f"\n✓ Comparison demo complete! Results saved to: {out_dir}\n")
