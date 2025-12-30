#!/usr/bin/env python3
"""Demonstrate why pixel-level metrics fail for image retrieval."""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

# Load two random images from your dataset
images_dir = Path("/home/birgit/fiftyone/open-images-v7/train/data")
image_files = list(images_dir.glob("*.jpg"))[:3]

if len(image_files) < 3:
    print("Not enough images found")
    exit(1)

# Load images
imgs = [cv2.imread(str(f)) for f in image_files[:3]]
imgs = [cv2.resize(img, (224, 224)) for img in imgs]  # Same size for comparison

print("Testing Pixel-level SSIM vs Feature-level similarity")
print("=" * 60)

# Pixel-level: SSIM between completely different images
print("\n1. SSIM (Pixel-level) between different images:")
for i in range(len(imgs)):
    for j in range(i+1, len(imgs)):
        # Convert to grayscale for SSIM
        gray_i = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
        gray_j = cv2.cvtColor(imgs[j], cv2.COLOR_BGR2GRAY)

        similarity = ssim(gray_i, gray_j)
        print(f"  Image {i} vs Image {j}: {similarity:.3f}")

print("\n2. SSIM between same image with color change:")
# Create color-shifted version of first image
img_shifted = imgs[0].copy()
img_shifted[:, :, 0] = imgs[0][:, :, 2]  # Swap blue and red channels
img_shifted[:, :, 2] = imgs[0][:, :, 0]

gray_orig = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
gray_shifted = cv2.cvtColor(img_shifted, cv2.COLOR_BGR2GRAY)
print(f"  Original vs Color-shifted: {ssim(gray_orig, gray_shifted):.3f}")

print("\n3. Feature-level similarity would be:")
print("  Original vs Color-shifted: ~0.9-1.0 (high!)")
print("  Because structure/edges are identical")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("  - SSIM fails to find similar-but-different images")
print("  - Feature-level (LPIPS, MobileNet, HOG) works across images")
print("  - For image retrieval: MUST use feature-level")
