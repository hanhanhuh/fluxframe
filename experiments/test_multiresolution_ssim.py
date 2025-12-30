#!/usr/bin/env python3
"""Test SSIM at different resolutions."""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

images_dir = Path("/home/birgit/fiftyone/open-images-v7/train/data")
image_files = list(images_dir.glob("*.jpg"))[:5]

imgs = [cv2.imread(str(f)) for f in image_files[:5]]

print("Testing SSIM at different resolutions")
print("=" * 70)

resolutions = [224, 112, 56, 28, 16, 8]

for res in resolutions:
    print(f"\nResolution: {res}x{res}")

    # Resize images
    resized = [cv2.resize(img, (res, res)) for img in imgs]
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in resized]

    # Compare first image to others
    scores = []
    for i in range(1, len(grays)):
        score = ssim(grays[0], grays[i])
        scores.append(score)
        print(f"  Image 0 vs {i}: {score:.3f}")

    avg = np.mean(scores)
    print(f"  Average similarity: {avg:.3f}")
    print(f"  Range: {min(scores):.3f} - {max(scores):.3f}")

print("\n" + "=" * 70)
print("OBSERVATION:")
print("  Lower resolution → Higher SSIM scores")
print("  Captures 'macro' patterns instead of fine details")
print("  Better for finding 'accidental' visual matches")
