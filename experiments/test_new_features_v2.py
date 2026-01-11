#!/usr/bin/env python3
"""Fixed version: test spatial histograms, LBP, and wavelet features."""

import time
from pathlib import Path

import cv2
import numpy as np
import pywt
from scipy.spatial.distance import cosine
from skimage.feature import local_binary_pattern


def compute_spatial_color_histogram_v2(img, grid_size=4, bins=8):
    """
    FIXED: Reduced bins from 32 to 8.

    4x4 grid × 8×8×8 bins = 16 cells × 512 bins = 8,192 dimensions
    Much more reasonable than 524k!
    """
    h, w = img.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size

    histograms = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = img[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]

            hist = cv2.calcHist(
                [cell], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256]
            )
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)
            histograms.append(hist)

    return np.concatenate(histograms)


def compute_spatial_grayscale_histogram(img, grid_size=4, bins=32):
    """
    Alternative: grayscale histograms (much smaller).

    Captures brightness patterns spatially.
    4x4 grid × 32 bins = 512 dimensions
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell_h, cell_w = h // grid_size, w // grid_size

    histograms = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = gray[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]

            hist = cv2.calcHist([cell], [0], None, [bins], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)
            histograms.append(hist)

    return np.concatenate(histograms)


def compute_lbp_histogram_v2(img, grid_size=4, n_points=8, radius=1):
    """
    FIXED: Compute LBP on grid cells instead of whole image.

    Combines spatial layout with texture patterns.
    Better discriminative power.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell_h, cell_w = h // grid_size, w // grid_size

    histograms = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = gray[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]

            # Compute LBP for this cell
            lbp = local_binary_pattern(cell, n_points, radius, method="uniform")

            # Histogram of LBP codes (uniform patterns have n_points+2 bins)
            n_bins = n_points + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype(np.float32)
            hist = hist / (hist.sum() + 1e-7)
            histograms.append(hist)

    return np.concatenate(histograms)


def compute_wavelet_features(img, wavelet="db4", level=3):
    """Wavelet decomposition - unchanged, already working well."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coeffs = pywt.wavedec2(gray, wavelet, level=level)

    features = []
    approx = coeffs[0]
    features.extend([approx.mean(), approx.std()])

    for detail_level in coeffs[1:]:
        for detail in detail_level:
            features.extend([detail.mean(), detail.std(), np.median(detail)])

    return np.array(features, dtype=np.float32)


# Load test images
print("Loading test images...")
images_dir = Path("/home/birgit/fiftyone/open-images-v7/train/data")
image_files = list(images_dir.glob("*.jpg"))[:20]  # More images for better test

imgs = []
for f in image_files:
    img = cv2.imread(str(f))
    img = cv2.resize(img, (224, 224))
    imgs.append(img)

print(f"Loaded {len(imgs)} images")
print("=" * 70)

# Test all methods
methods = {
    "Spatial Color Histogram (4x4, 8 bins)": compute_spatial_color_histogram_v2,
    "Spatial Grayscale Histogram (4x4, 32 bins)": compute_spatial_grayscale_histogram,
    "Spatial LBP (4x4 grid)": compute_lbp_histogram_v2,
    "Wavelet Features": compute_wavelet_features,
}

results = {}

for method_name, compute_func in methods.items():
    print(f"\n{method_name}")
    print("-" * 70)

    # Compute features
    start = time.time()
    features = [compute_func(img) for img in imgs]
    elapsed = time.time() - start

    print(f"Feature extraction: {elapsed:.3f}s for {len(imgs)} images")
    print(f"Feature dimension: {len(features[0])}")
    print(f"Speed: {len(imgs) / elapsed:.1f} images/sec")

    # Compare image 0 to all others
    similarities = []
    for i in range(1, len(features)):
        sim = 1 - cosine(features[0], features[i])
        similarities.append(sim)

    print("\nSimilarities (first 10):")
    for i, sim in enumerate(similarities[:10], 1):
        print(f"  Image 0 vs {i}: {sim:.3f}")

    avg = np.mean(similarities)
    std = np.std(similarities)
    print(f"\nStatistics (all {len(similarities)} comparisons):")
    print(f"  Average: {avg:.3f}")
    print(f"  Std dev: {std:.3f}")
    print(f"  Range: {min(similarities):.3f} - {max(similarities):.3f}")

    # Good discriminative power if std > 0.02 and range > 0.1
    discriminative = std > 0.02 and (max(similarities) - min(similarities)) > 0.1

    results[method_name] = {
        "speed": len(imgs) / elapsed,
        "dimension": len(features[0]),
        "avg": avg,
        "std": std,
        "range": max(similarities) - min(similarities),
        "discriminative": discriminative,
    }

print("\n" + "=" * 70)
print("\nSUMMARY:")
print("-" * 70)
print(f"{'Method':<45} {'Speed':>10} {'Dims':>8} {'Range':>10} {'Quality':>10}")
print("-" * 70)

for name, res in results.items():
    quality = "✓ Good" if res["discriminative"] else "✗ Poor"
    print(
        f"{name:<45} {res['speed']:>8.1f}/s {res['dimension']:>8d} {res['range']:>10.3f} {quality:>10}"
    )

print("\nRecommendation: Use methods marked '✓ Good'")
