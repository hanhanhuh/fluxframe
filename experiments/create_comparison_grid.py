#!/usr/bin/env python3
"""Create side-by-side comparison grids from pooling test samples."""

from pathlib import Path

import cv2
import numpy as np

# Output directories
CONFIGS = [
    ("avg_2x2", "Avg 2x2 (baseline)", "output_avg_2x2"),
    ("avg_3x3", "Avg 3x3", "output_avg_3x3"),
    ("avg_global", "Avg Global (1x1)", "output_avg_global"),
    ("gem_2x2", "GeM 2x2", "output_gem_2x2"),
    ("gem_global", "GeM Global (1x1)", "output_gem_global"),
]


def create_comparison_grid(sample_num, output_path):
    """Create a comparison grid for a specific sample number."""

    # Load all samples for this frame
    images = []
    labels = []

    for _config_name, label, output_dir in CONFIGS:
        sample_path = Path(output_dir) / "comparison_samples" / f"sample_{sample_num:06d}.jpg"

        if sample_path.exists():
            img = cv2.imread(str(sample_path))
            if img is not None:
                images.append(img)
                labels.append(label)
            else:
                print(f"Warning: Could not read {sample_path}")
        else:
            print(f"Warning: Sample not found: {sample_path}")

    if not images:
        print(f"No images found for sample {sample_num}")
        return False

    # Ensure all images have the same width
    max_width = max(img.shape[1] for img in images)
    resized_images = []

    for img in images:
        if img.shape[1] != max_width:
            # Resize to match max width while preserving aspect ratio
            h, w = img.shape[:2]
            new_h = int(h * max_width / w)
            img_resized = cv2.resize(img, (max_width, new_h))
            resized_images.append(img_resized)
        else:
            resized_images.append(img)

    # Add labels to images
    labeled_images = []
    for img, label in zip(resized_images, labels, strict=False):
        # Create a copy to avoid modifying original
        img_labeled = img.copy()

        # Add label at the top
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (0, 255, 255)  # Yellow

        # Get text size for background
        (text_w, text_h), _baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw background rectangle
        cv2.rectangle(img_labeled, (5, 5), (text_w + 15, text_h + 15), (0, 0, 0), -1)

        # Draw text
        cv2.putText(
            img_labeled, label, (10, text_h + 10), font, font_scale, color, thickness, cv2.LINE_AA
        )

        labeled_images.append(img_labeled)

    # Stack vertically
    grid = np.vstack(labeled_images)

    # Save comparison grid
    cv2.imwrite(str(output_path), grid)
    print(f"✓ Created comparison grid: {output_path}")

    return True


def main():
    """Create comparison grids for all samples."""

    print("Creating comparison grids...")
    print()

    # Find all sample numbers (check first config directory)
    first_config_dir = Path(CONFIGS[0][2]) / "comparison_samples"

    if not first_config_dir.exists():
        print(f"Error: Sample directory not found: {first_config_dir}")
        print("Run pooling_comparison.sh first!")
        return

    # Get all sample files
    sample_files = sorted(first_config_dir.glob("sample_*.jpg"))

    if not sample_files:
        print(f"No samples found in {first_config_dir}")
        return

    print(f"Found {len(sample_files)} samples")
    print()

    # Create output directory
    comparison_dir = Path("./pooling_comparisons")
    comparison_dir.mkdir(exist_ok=True)

    # Create grid for each sample
    for sample_file in sample_files:
        # Extract frame number
        sample_num = int(sample_file.stem.split("_")[1])

        output_path = comparison_dir / f"comparison_frame_{sample_num:06d}.jpg"
        create_comparison_grid(sample_num, output_path)

    print()
    print("=" * 60)
    print("Comparison grids created!")
    print("=" * 60)
    print(f"Location: {comparison_dir}")
    print()
    print(f"Created {len(sample_files)} comparison images")
    print("Each image shows all 5 pooling methods side-by-side")
    print()
    print("Visual inspection tips:")
    print("  - Check for matching quality across methods")
    print("  - Look for differences in scene/motion preservation")
    print("  - Note if global pooling loses important spatial info")


if __name__ == "__main__":
    main()
