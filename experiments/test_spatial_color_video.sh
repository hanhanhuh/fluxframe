#!/bin/bash
# Test Spatial Color Histogram on full video segment

VIDEO="/home/birgit/Downloads/Winter Cycling： Vantaa, Finland - 4K 60fps.mp4"
IMAGES="/home/birgit/fiftyone/open-images-v7/train/data"

source venv/bin/activate

echo "Testing Spatial Color Histogram on video"
echo "=========================================="
echo ""
echo "Running 10 second test with 1000 images"
echo ""

# We need to add this as a new feature method to fluxframe
# For now, let's create a standalone script

python3 << 'PYTHON_SCRIPT'
import cv2
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
import faiss
from tqdm import tqdm
import json

def compute_spatial_color_histogram(img, grid_size=4, bins=8):
    """Spatial color histogram features."""
    h, w = img.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size

    histograms = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            hist = cv2.calcHist([cell], [0, 1, 2], None,
                               [bins, bins, bins],
                               [0, 256, 0, 256, 0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)
            histograms.append(hist)

    return np.concatenate(histograms).astype(np.float32)

print("Loading video...")
video_path = Path("/home/birgit/Downloads/Winter Cycling： Vantaa, Finland - 4K 60fps.mp4")
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)

# Extract 10 seconds at 25 fps
target_fps = 25
skip = int(fps / target_fps)
max_frames = 10 * target_fps  # 250 frames

frames = []
frame_idx = 0
while len(frames) < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % skip == 0:
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    frame_idx += 1

cap.release()
print(f"Extracted {len(frames)} video frames")

# Load database images
print("Loading database images...")
images_dir = Path("/home/birgit/fiftyone/open-images-v7/train/data")
image_files = list(images_dir.glob("*.jpg"))[:1000]

db_images = []
db_paths = []
for f in tqdm(image_files, desc="Loading images"):
    img = cv2.imread(str(f))
    if img is not None:
        img = cv2.resize(img, (224, 224))
        db_images.append(img)
        db_paths.append(str(f))

print(f"Loaded {len(db_images)} database images")

# Compute features
print("\nComputing database features...")
db_features = []
for img in tqdm(db_images, desc="Database features"):
    feat = compute_spatial_color_histogram(img)
    db_features.append(feat)

db_features = np.array(db_features, dtype=np.float32)

# Build FAISS index
print("Building FAISS index...")
dimension = db_features.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

# Normalize for cosine similarity
faiss.normalize_L2(db_features)
index.add(db_features)

print(f"Index built with {index.ntotal} vectors")

# Match video frames
print("\nMatching video frames...")
matches = []

for frame_num, frame in enumerate(tqdm(frames, desc="Matching frames")):
    # Compute feature
    feat = compute_spatial_color_histogram(frame)
    feat = feat.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(feat)

    # Search
    D, I = index.search(feat, 1)

    best_idx = I[0][0]
    similarity = D[0][0]

    matches.append({
        'frame': frame_num,
        'matched_image': db_paths[best_idx],
        'similarity': float(similarity)
    })

# Save results
output_dir = Path("output_spatial_color_test")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "results.json", "w") as f:
    json.dump(matches, f, indent=2)

# Create video
print("\nCreating output video...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    str(output_dir / "matched_video.mp4"),
    fourcc, 25.0, (224, 224)
)

for match in tqdm(matches, desc="Writing video"):
    img = cv2.imread(match['matched_image'])
    img = cv2.resize(img, (224, 224))
    out.write(img)

out.release()

# Save comparison samples (every 25th frame)
print("Saving comparison samples...")
samples_dir = output_dir / "comparison_samples"
samples_dir.mkdir(exist_ok=True)

for i in range(0, len(frames), 25):
    if i >= len(matches):
        break

    match = matches[i]
    matched_img = cv2.imread(match['matched_image'])
    matched_img = cv2.resize(matched_img, (224, 224))

    # Side by side
    comparison = np.hstack([frames[i], matched_img])

    # Add labels
    h, w = comparison.shape[:2]
    labeled = np.ones((h + 40, w, 3), dtype=np.uint8) * 255
    labeled[40:, :] = comparison

    cv2.putText(labeled, f"Frame {i:06d}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(labeled, f"Match (sim: {match['similarity']:.3f})",
                (frames[i].shape[1] + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)

    cv2.imwrite(str(samples_dir / f"sample_{i:06d}.jpg"), labeled)

print(f"\nDone! Output saved to: {output_dir}")
print(f"  - matched_video.mp4")
print(f"  - comparison_samples/")
print(f"  - results.json")

# Print statistics
similarities = [m['similarity'] for m in matches]
print(f"\nSimilarity statistics:")
print(f"  Average: {np.mean(similarities):.3f}")
print(f"  Min: {np.min(similarities):.3f}")
print(f"  Max: {np.max(similarities):.3f}")
print(f"  Std dev: {np.std(similarities):.3f}")

PYTHON_SCRIPT

echo ""
echo "Test complete!"
echo "Check output_spatial_color_test/ for results"
