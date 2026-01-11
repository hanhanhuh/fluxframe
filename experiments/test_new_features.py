#!/usr/bin/env python3
"""Test spatial histograms, LBP, and wavelet features for aesthetic matching."""

import time
from pathlib import Path

import cv2
import numpy as np
import pywt
from scipy.spatial.distance import cosine, euclidean
from skimage.feature import local_binary_pattern


def compute_spatial_color_histogram(img, grid_size=4, bins=32):
    """
    Divide image into grid and compute color histogram per cell.

    Captures: "dark patch in top-left matches dark patch in top-left"
    Content-independent spatial color matching.
    """
    h, w = img.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size

    histograms = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Extract cell
            cell = img[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]

            # Compute 3D color histogram for this cell
            hist = cv2.calcHist(
                [cell], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256]
            )
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)  # Normalize
            histograms.append(hist)

    # Concatenate all cell histograms
    return np.concatenate(histograms)


def compute_lbp_features(img, n_points=24, radius=3, n_bins=256):
    """
    Local Binary Patterns - captures micro-texture patterns.

    Captures: Repeating textures, edge patterns, local structures
    Content-independent: brick texture matches any brick, regardless of what building
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

    # Compute histogram of LBP codes
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    return hist / (hist.sum() + 1e-7)


def compute_wavelet_features(img, wavelet="db4", level=3):
    """
    Wavelet decomposition - captures frequency patterns.

    Captures: Visual rhythms, patterns at different scales
    Content-independent: similar frequency patterns match regardless of content
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Multi-level wavelet decomposition
    coeffs = pywt.wavedec2(gray, wavelet, level=level)

    features = []

    # Approximation coefficients (coarse features)
    approx = coeffs[0]
    features.append(approx.mean())
    features.append(approx.std())

    # Detail coefficients at each level (fine features)
    for detail_level in coeffs[1:]:
        for detail in detail_level:  # Horizontal, Vertical, Diagonal
            features.append(detail.mean())
            features.append(detail.std())
            features.append(np.median(detail))

    return np.array(features, dtype=np.float32)


def compare_features(feat1, feat2, method="cosine"):
    """Compare two feature vectors."""
    if method == "cosine":
        return 1 - cosine(feat1, feat2)  # Convert distance to similarity
    if method == "euclidean":
        return 1 / (1 + euclidean(feat1, feat2))  # Convert to similarity
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))


# Load test images
print("Loading test images...")
images_dir = Path("/home/birgit/fiftyone/open-images-v7/train/data")
image_files = list(images_dir.glob("*.jpg"))[:10]

imgs = []
for f in image_files:
    img = cv2.imread(str(f))
    img = cv2.resize(img, (224, 224))
    imgs.append(img)

print(f"Loaded {len(imgs)} images")
print("=" * 70)

# Test all three methods
methods = {
    "Spatial Color Histogram (4x4 grid)": compute_spatial_color_histogram,
    "Local Binary Patterns (LBP)": compute_lbp_features,
    "Wavelet Features": compute_wavelet_features,
}

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
    print("\nSimilarity of Image 0 to others:")
    similarities = []
    for i in range(1, len(features)):
        sim = compare_features(features[0], features[i])
        similarities.append(sim)
        print(f"  Image 0 vs {i}: {sim:.3f}")

    print(f"Average: {np.mean(similarities):.3f}")
    print(f"Range: {min(similarities):.3f} - {max(similarities):.3f}")
    print(f"Std dev: {np.std(similarities):.3f}")

print("\n" + "=" * 70)
print("\nCOMPARISON:")
print("Look for:")
print("  - Good range of similarities (not all ~0.5)")
print("  - Fast extraction speed")
print("  - Reasonable discriminative power")
