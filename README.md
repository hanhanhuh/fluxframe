# FluxFrame

Create temporal video collages by replacing each frame with similar images from large datasets. FluxFrame transforms videos into artistic visual remixes where the original motion and composition remain recognizable, but constructed entirely from different source images.

Optimized for CPU performance with FAISS vector search, making it suitable for matching against 500K+ image pools.

## Features

- **Multiple feature extraction methods**:
  - **Canny**: Fast edge histogram (default)
  - **Spatial Pyramid**: 4x4 grid preserves spatial layout
  - **HOG**: Histogram of Oriented Gradients for best motion preservation
  - **MobileNet**: Neural network features for semantic similarity
- **Multi-metric similarity**: Combines edge, texture (Sobel), and color (HSV) features
- **FAISS vector search**: Fast exact similarity search with IndexFlatIP
- **No-repeat mode**: Guarantees zero duplicates - always picks best unused match
- **Aspect-ratio preserving**: Center cropping, no letterboxing
- **FAISS caching**: Build index once, reuse across runs
- **Cache validation**: SHA256 parameter tracking for automatic invalidation
- **Frame skip control**: Process subset of frames with `--fps-override`
- **Checkpoint resume**: Save progress, resume after interruption
- **Demo mode**: Test on video/image subsets before full run
- **Type-safe**: Full mypy strict mode compliance

## Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv pip install -e .
```

**For MobileNet support (optional):**
```bash
# Install onnxscript (runtime dependency)
uv pip install -e ".[mobilenet]"

# One-time: Install PyTorch CPU-only for ONNX export (~200MB)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Export ONNX model (automatic on first use, or run manually)
python -c "from src.fluxframe.matcher import ImageMatcher; ImageMatcher(feature_method='mobilenet')"

# After ONNX model cached (~/.cache/fluxframe/), uninstall PyTorch
uv pip uninstall torch torchvision
```

Or using pip:

```bash
pip install -e .
```

## Quick Start

**Basic usage:**
```bash
fluxframe <video_file> <image_folder> <output_dir>
```

**Compare all feature methods (demo):**
```bash
python demo_feature_comparison.py <video_file> <image_folder> --frames 5
```
This generates side-by-side comparisons showing how different methods match frames, with performance benchmarks.

### Examples

**Default (Canny features):**
```bash
fluxframe input.mp4 /path/to/images ./output
```

**Best quality (HOG features, no duplicates):**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --feature-method hog \
  --no-repeat
```

**Custom FPS (process fewer frames):**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --fps-override 15
```

**Emphasize color over structure:**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --color-weight 0.8 --edge-weight 0.1 --texture-weight 0.1
```

**With similarity threshold:**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --no-repeat --threshold 0.5
```

**Demo mode (quick test):**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --demo --demo-seconds 30 --demo-images 500
```

**Save comparison samples:**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --save-samples 20 --sample-interval 10
```

### Command-line Arguments

**Required:**
- `video`: Path to input video file
- `images`: Path to folder containing images (flat structure)
- `output`: Output directory

**Optional:**
- `--feature-method`: `canny` (fast), `spatial_pyramid` (balanced), `hog` (best motion), `mobilenet` (semantic) (default: canny)
- `--search-depth`: Number of top matches to find (default: 10)
- `--edge-weight`: Edge similarity weight 0-1 (default: 0.33)*
- `--texture-weight`: Texture similarity weight 0-1 (default: 0.33)*
- `--color-weight`: Color similarity weight 0-1 (default: 0.34)*
- `--threshold`: Minimum similarity threshold 0-1 (default: 0.0)
- `--no-repeat`: Use each image only once (zero duplicates guaranteed)
- `--comparison-size`: Image resize for comparison (default: 256)
- `--fps-override`: Output FPS by skipping input frames (default: input FPS)
- `--skip-output`: Only compute matches, skip video generation
- `--demo`: Process subset for quick testing
- `--demo-seconds`: Seconds to process in demo (default: 20)
- `--demo-images`: Images to use in demo (default: 1000)
- `--checkpoint-batch`: Save progress every N frames (default: 10)
- `--seed`: Random seed for reproducibility
- `--save-samples`: Number of frame-match comparison samples to save (default: 0)
- `--sample-interval`: Save every Nth frame as sample (default: 1)

*Weights are auto-normalized to sum to 1.0 (ignored for mobilenet)

### Feature Methods

| Method | Speed | Quality | Feature Size | Spatial Info |
|--------|-------|---------|--------------|--------------|
| `canny` | ⭐⭐⭐⭐⭐ | ⭐⭐ | 832D | ❌ None |
| `spatial_pyramid` | ⭐⭐⭐⭐ | ⭐⭐⭐ | 1088D | ✅ 4×4 grid |
| `hog` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 1332D* | ✅ Cells |
| `mobilenet` | ⭐⭐⭐ | ⭐⭐⭐⭐ | 192D | ✅ 2×2 grid |

*HOG dimensions vary with `--comparison-size` (684D at 128px, 1332D at 256px, 4896D at 512px)

**Note:**
- MobileNet uses 2×2 spatial pyramid pooling to preserve perspective/layout, ignoring edge/texture/color weights
- Requires one-time setup: `uv pip install -e ".[mobilenet-setup]"` (see Installation section)
- After ONNX export (~10 seconds), PyTorch can be uninstalled - subsequent runs use cached ONNX model

## How It Works

### 1. FAISS Index Building (First Run)
- Computes edge, texture, and color features for all images
- Concatenates weighted features into single vector per image
- Normalizes with L2 (for cosine similarity via inner product)
- Builds FAISS IndexFlatIP (exact search)
- Caches: `faiss_index.bin`, `vectors.npy`, `cache_metadata.json`
- **Subsequent runs**: Instant reload from cache

### 2. Frame Processing
- Reads video metadata (FPS, resolution, frames)
- Processes frames sequentially (memory efficient)
- Applies frame skip if `--fps-override` set
- Checkpoints every N frames for resume

### 3. Similarity Search
**Feature Extraction:**
- Edge: Canny/Spatial Pyramid/HOG
- Texture: Sobel gradient magnitude
- Color: HSV 3D histogram
- Concatenate weighted, normalize L2

**FAISS Search:**
- Query index with search depth k
- With `--no-repeat`: Expand k to `k + len(used)`, post-filter used images
- Returns top k unused matches sorted by similarity

**Combined Score:**
```
similarity = edge_weight × edge_sim + texture_weight × texture_sim + color_weight × color_sim
```

### 4. Selection Strategy

**With `--no-repeat` (recommended)**:
- Always selects **best match** (highest similarity)
- Post-filtering ensures used images excluded
- Guarantees **zero duplicates**
- **Deterministic**: same input = same output

**Without `--no-repeat`**:
- Randomly samples from top k for variety
- Images can be reused
- Non-deterministic

### 5. Checkpointing
- Saves every N frames to `checkpoint.json`
- Stores top matches and selected image per frame
- Resume on interruption or rerun with different output settings

### 6. Output Generation
- Creates video with matched images
- Saves individual frames as JPEGs
- Images cropped/resized to match video dimensions

## Output Structure

```
output_dir/
├── checkpoint.json           # Resume checkpoint
├── results.json             # Final mappings
├── cache_metadata.json      # FAISS cache params
├── faiss_index.bin          # FAISS index
├── vectors.npy              # Feature vectors
├── <video>_matched.mp4      # Output video
├── matched_frames/          # Individual frames
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
└── comparison_samples/      # Frame-match comparisons (if --save-samples used)
    ├── sample_000000.jpg
    ├── sample_000010.jpg
    └── ...
```

## Performance

**Typical (500K images, 30fps video)**:
- Index build (first run): ~45-60 min on 8-core CPU
- Index build (cached): Instant reload
- Per-frame search: ~0.3-0.5 sec
- 5 min video: ~4-7 min total
- 30 min video: ~25-40 min total

**Scaling**:
- 10K images: ~1-2 min build
- 100K images: ~8-12 min build
- 500K images: ~45-60 min build (8 cores)
- Parallel processing uses all CPU cores
- FAISS search: Very fast even with millions of vectors
- Post-filtering: Negligible overhead
- Cache: Instant on subsequent runs

**Memory**:
- ~4 bytes per dimension per image
- Canny (832D): 500K images = ~1.6GB
- Spatial Pyramid (1088D): 500K images = ~2.1GB
- MobileNet (192D): 500K images = ~384MB
- HOG at 256px (1332D): 500K images = ~2.5GB
- HOG at 512px (4896D): 500K images = ~9.3GB

**Tips**:
- Smaller `--comparison-size` (128/256) = faster
- Larger `--comparison-size` (512/1024) = more accurate
- `canny`: Fastest feature method
- `hog`: Best quality, slower
- Cache rebuilds only when parameters change

## Development

Tests:
```bash
pytest tests/ -v
```

Linting:
```bash
ruff check src/ tests/
mypy src/
```

## Resume After Errors

Automatically resumes from `checkpoint.json`. To restart with different parameters:
1. Delete `checkpoint.json`
2. Run with new parameters

**Note**: FAISS cache auto-invalidates when parameters change (weights, feature method, comparison size, image pool).

## Technical Details

**Color**: HSV 3D histograms, correlation metric

**Edges**:
- Canny: Fast, no spatial info
- Spatial Pyramid: 4x4 grid preserves layout
- HOG: Gradient orientation, best for motion

**Texture**: Sobel gradient magnitude (fast alternative to LBP)

**Vector Search**:
- FAISS IndexFlatIP (exact inner product = cosine similarity after L2 norm)
- Post-filter for no-repeat (simple, fast, exact)
- No approximation, always best match

## Sources

- [FAISS Library](https://github.com/facebookresearch/faiss)
- [FAISS Documentation](https://faiss.ai/index.html)
- [OpenCV Histogram Comparison](https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html)
- [Canny Edge Detection](https://medium.com/@abhisheksriram845/canny-edge-detection-explained-and-compared-with-opencv-in-python-57a161b4bd19)
- [Histogram of Oriented Gradients](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html)
