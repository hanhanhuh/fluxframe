#!/usr/bin/env python3
"""Create visual comparison of different feature matching methods."""

import numpy as np
import cv2
from pathlib import Path
import sys

# Import our feature functions
sys.path.insert(0, str(Path(__file__).parent))
from test_new_features_v2 import (
    compute_spatial_color_histogram_v2,
    compute_spatial_grayscale_histogram,
    compute_lbp_histogram_v2,
    compute_wavelet_features
)
from scipy.spatial.distance import cosine

def find_best_match(query_features, database_features):
    """Find index of best matching image."""
    best_idx = -1
    best_sim = -1

    for idx, db_feat in enumerate(database_features):
        sim = 1 - cosine(query_features, db_feat)
        if sim > best_sim:
            best_sim = sim
            best_idx = idx

    return best_idx, best_sim

# Load video frame
video_path = Path("/home/birgit/Downloads/Winter Cycling： Vantaa, Finland - 4K 60fps.mp4")
if not video_path.exists():
    print(f"Video not found: {video_path}")
    exit(1)

cap = cv2.VideoCapture(str(video_path))
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # Frame 100
ret, video_frame = cap.read()
cap.release()

if not ret:
    print("Failed to read video frame")
    exit(1)

video_frame = cv2.resize(video_frame, (224, 224))

# Load database images
print("Loading database images...")
images_dir = Path("/home/birgit/fiftyone/open-images-v7/train/data")
image_files = list(images_dir.glob("*.jpg"))[:100]  # Use 100 images

db_images = []
for f in image_files:
    img = cv2.imread(str(f))
    if img is not None:
        img = cv2.resize(img, (224, 224))
        db_images.append(img)

print(f"Loaded {len(db_images)} database images")

# Test different methods
methods = {
    'Spatial Grayscale': compute_spatial_grayscale_histogram,
    'Spatial Color': compute_spatial_color_histogram_v2,
    'Spatial LBP': compute_lbp_histogram_v2,
    'Wavelet': compute_wavelet_features,
}

print("\nFinding best matches for each method...")
print("=" * 70)

results = {}

for method_name, compute_func in methods.items():
    print(f"\n{method_name}:")

    # Compute features
    query_feat = compute_func(video_frame)
    db_feats = [compute_func(img) for img in db_images]

    # Find best match
    best_idx, best_sim = find_best_match(query_feat, db_feats)

    print(f"  Best match: Image {best_idx}")
    print(f"  Similarity: {best_sim:.3f}")

    results[method_name] = {
        'image': db_images[best_idx],
        'similarity': best_sim,
        'index': best_idx
    }

# Create comparison visualization
print("\nCreating comparison image...")

# Calculate layout
n_methods = len(methods)
img_h, img_w = 224, 224
margin = 10
text_height = 60

# Layout: video frame on left, then each method's match
total_width = (n_methods + 1) * (img_w + margin) + margin
total_height = img_h + 2 * margin + text_height

canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

# Add video frame
y_offset = margin + text_height
canvas[y_offset:y_offset+img_h, margin:margin+img_w] = video_frame

# Add label
cv2.putText(canvas, "Video Frame", (margin, margin + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Add matched images
x_offset = margin + img_w + margin

for method_name, result in results.items():
    # Add image
    canvas[y_offset:y_offset+img_h, x_offset:x_offset+img_w] = result['image']

    # Add label with similarity
    label = f"{method_name}"
    sim_text = f"sim: {result['similarity']:.3f}"

    cv2.putText(canvas, label, (x_offset, margin + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, sim_text, (x_offset, margin + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1)

    x_offset += img_w + margin

# Save
output_path = Path("feature_comparison.jpg")
cv2.imwrite(str(output_path), canvas)
print(f"\nSaved comparison to: {output_path}")

# Also create individual comparisons for each method
print("\nCreating individual comparison images...")

for method_name, result in results.items():
    # Side by side: video frame | matched image
    comparison = np.hstack([video_frame, result['image']])

    # Add labels
    h, w = comparison.shape[:2]
    labeled = np.ones((h + 40, w, 3), dtype=np.uint8) * 255
    labeled[40:, :] = comparison

    cv2.putText(labeled, "Video Frame", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(labeled, f"{method_name} Match (sim: {result['similarity']:.3f})",
                (video_frame.shape[1] + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)

    filename = f"comparison_{method_name.lower().replace(' ', '_')}.jpg"
    cv2.imwrite(filename, labeled)
    print(f"  Saved: {filename}")

print("\nDone! Check the generated images.")
