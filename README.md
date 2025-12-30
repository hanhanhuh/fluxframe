# FluxFrame

Create temporal video collages by replacing each frame with similar images from large datasets. FluxFrame transforms videos into artistic visual remixes where the original motion and composition remain recognizable, but constructed entirely from different source images.

Optimized for CPU performance with FAISS vector search, memory-mapped indexes, and optional TurboJPEG (3x faster image loading), making it suitable for matching against 500K+ image pools.

## Features

- **Multiple feature extraction methods**:
  - **Canny**: Fast edge histogram (default)
  - **Spatial Pyramid**: 2×2 or 3×3 grid preserves spatial layout
  - **HOG**: Histogram of Oriented Gradients for motion preservation
  - **Spatial Color**: 4×4 grid with dense color histograms (8192D→256D for large datasets)
  - **MobileNet**: Neural network features (48 channels) for semantic similarity
  - **EfficientNet**: Powerful neural features (112 channels) for higher quality
- **Configurable spatial pooling**: Average or GeM (Generalized Mean) pooling for neural methods
- **Multi-metric similarity**: Combines edge, texture (Sobel), and color (HSV) features
- **FAISS vector search**: Fast exact search with IndexFlatIP (optional IVF for 16x faster search)
- **Performance optimizations**: OpenCV SIMD, memory-mapped indexes, optional TurboJPEG
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

**For neural network support (MobileNet/EfficientNet, optional):**
```bash
# Install onnxscript (runtime dependency)
uv pip install -e ".[mobilenet]"

# One-time: Install PyTorch CPU-only for ONNX export (~200MB)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Export ONNX models (automatic on first use, or run manually)
python -c "from src.fluxframe.matcher import ImageMatcher; ImageMatcher(feature_method='mobilenet')"
python -c "from src.fluxframe.matcher import ImageMatcher; ImageMatcher(feature_method='efficientnet')"

# After ONNX models cached (~/.cache/fluxframe/), uninstall PyTorch
uv pip uninstall torch torchvision
```

**For 3x faster JPEG loading (optional):**
```bash
# Install PyTurboJPEG (automatically uses libjpeg-turbo if available)
uv pip install PyTurboJPEG
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
python demo_feature_comparison.py <video_file> <image_folder> --frames 5 --demo-images 1000
```
This generates side-by-side comparisons showing how different methods match frames, with performance benchmarks. Use `--demo-images` to limit the image pool for faster testing.

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

**Neural network (EfficientNet with GeM pooling, 3×3 grid):**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --feature-method efficientnet \
  --pooling-method gem \
  --gem-p 3.0 \
  --spatial-grid 3
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
- `--feature-method`: `canny` (fast), `spatial_pyramid` (balanced), `hog` (best motion), `spatial_color` (high quality, large datasets), `mobilenet` (neural), `efficientnet` (neural, higher quality) (default: canny)
- `--pooling-method`: `avg` (average pooling, fast), `gem` (Generalized Mean pooling, better quality for neural methods) (default: avg)
- `--gem-p`: GeM pooling power parameter (1=avg, 3-4=optimal, ∞=max) (default: 3.0)
- `--spatial-grid`: Spatial pyramid grid size for neural methods (2=2×2, 3=3×3) (default: 2)
- `--force-mobilenet-export`: Force re-export of ONNX models (regenerates cached models)
- `--search-depth`: Number of top matches to find (default: 10)
- `--use-ivf-index`: Use IndexIVFFlat for 16x faster search (default: false, adds overhead to index building)
- `--ivf-nlist`: Number of IVF clusters (default: auto = √N × 4)
- `--ivf-nprobe`: Number of IVF clusters to search (default: auto = nlist/32)
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

*Weights are auto-normalized to sum to 1.0 (ignored for neural methods which only use edge features)

### Feature Methods

| Method | Speed | Quality | Feature Size | Spatial Info | Config |
|--------|-------|---------|--------------|--------------|--------|
| `canny` | ⭐⭐⭐⭐⭐ | ⭐⭐ | 832D | ❌ None | — |
| `spatial_pyramid` | ⭐⭐⭐⭐ | ⭐⭐⭐ | 1088D (2×2)<br>2448D (3×3) | ✅ Grid | `--spatial-grid` |
| `hog` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 1332D* | ✅ Cells | — |
| `spatial_color` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 256D (8192D→256D)** | ✅ 4×4 Grid | — |
| `mobilenet` | ⭐⭐⭐ | ⭐⭐⭐⭐ | 192D (2×2+avg)<br>432D (3×3+avg)<br>192D (2×2+gem)<br>432D (3×3+gem) | ✅ Grid | `--pooling-method`<br>`--gem-p`<br>`--spatial-grid` |
| `efficientnet` | ⭐⭐ | ⭐⭐⭐⭐⭐ | 448D (2×2+avg)<br>1008D (3×3+avg)<br>448D (2×2+gem)<br>1008D (3×3+gem) | ✅ Grid | `--pooling-method`<br>`--gem-p`<br>`--spatial-grid` |

*HOG dimensions vary with `--comparison-size` (684D at 128px, 1332D at 256px, 4896D at 512px)
**Spatial Color: 4×4 grid × 512 color bins = 8192D, auto-reduced to 256D via random projection

**Recommended configurations:**
- **Fast preview**: `canny` (default)
- **Balanced quality**: `spatial_pyramid --spatial-grid 2`
- **Best motion**: `hog`
- **Large datasets**: `spatial_color` (4×4 color grid, auto-reduces to 256D)
- **Neural fast**: `mobilenet --pooling-method gem --spatial-grid 2`
- **Neural best**: `efficientnet --pooling-method gem --spatial-grid 3 --gem-p 3.0`

**Neural network notes:**
- Use edge features only (ignore edge/texture/color weights)
- Require one-time setup: `uv pip install -e ".[mobilenet]"` + PyTorch CPU (see Installation)
- After ONNX export (~10 seconds per model), PyTorch can be uninstalled
- Use single-process execution (ONNX Runtime handles multi-threading internally)
- **GeM pooling**: Generalizes average (p=1) and max (p=∞) pooling. p=3-4 optimal for image retrieval
- **Spatial grid**: 2×2 faster, 3×3 better preserves layout details
- Classical methods (canny/hog/spatial_pyramid) use multiprocessing for parallelism

## How It Works

### 1. FAISS Index Building (First Run)
- Computes edge, texture, and color features for all images
- Concatenates weighted features into single vector per image
- Normalizes with L2 (for cosine similarity via inner product)
- Builds FAISS IndexFlatIP (exact search, default)
  - Optional: Use `--use-ivf-index` for IndexIVFFlat (16x faster search, adds training overhead)
  - IVF uses k-means clustering with auto-calculated nlist (√N × 4)
  - Searches only nearest nprobe clusters (nlist/32)
  - Falls back to exact search for small datasets (<100 images)
- Caches: `faiss_index.bin`, `vectors.npy`, `cache_metadata.json`
- **Subsequent runs**: Instant reload with memory-mapped indexes

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
├── dim_reducer.pkl          # Dimensionality reducer (spatial_color only)
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

**Optimizations (enabled by default)**:
- **Reduced feature dimensions**: 128 edge + 32 texture + 216 color (55% reduction from 832D)
- **Optimized resize**: INTER_AREA for 10-15% faster downscaling
- **Vectorized normalization**: In-place operations for 5-10% speedup
- **Memory-mapped indexes**: Faster loading, supports larger-than-RAM datasets
- **OpenCV SIMD**: Automatic SSE2/AVX optimizations
- **TurboJPEG** (optional): 3x faster JPEG loading
- **IndexIVFFlat** (opt-in): 16x faster search, slower index building (use `--use-ivf-index`)

**Typical (500K images, 30fps video, 8-core CPU)**:
- Index build (first run): ~25-35 min (30-50% faster with optimizations)
- Index load (cached): <1 sec with memory mapping
- Per-frame search: ~0.2-0.3 sec (exact), ~0.02-0.03 sec (with IVF)
- 5 min video: ~3-5 min total (exact), ~1-2 min (with IVF after initial build)
- 30 min video: ~15-25 min total (exact), ~5-10 min (with IVF after initial build)

**Scaling**:
- 10K images: ~0.5-1 min build
- 100K images: ~5-8 min build
- 500K images: ~25-35 min build (8 cores)
- Parallel processing uses all CPU cores
- FAISS IVF search: Sub-millisecond even with millions of vectors
- Post-filtering: Negligible overhead
- Cache: Instant reload with memory mapping

**Memory**:
- ~4 bytes per dimension per image
- Canny (376D, optimized): 500K images = ~752MB
- Spatial Pyramid 2×2 (1088D): 500K images = ~2.1GB
- Spatial Pyramid 3×3 (2448D): 500K images = ~4.7GB
- Spatial Color (256D, reduced): 500K images = ~512MB (mini-batch: 170MB peak)
- MobileNet 2×2 (192D): 500K images = ~384MB
- MobileNet 3×3 (432D): 500K images = ~864MB
- EfficientNet 2×2 (448D): 500K images = ~896MB
- EfficientNet 3×3 (1008D): 500K images = ~2.0GB
- HOG at 256px (1332D): 500K images = ~2.5GB
- HOG at 512px (4896D): 500K images = ~9.3GB

*Note: Canny method uses optimized feature dimensions (128 edge + 32 texture + 216 color = 376D) for 30-50% faster index building*

**Tips**:
- Smaller `--comparison-size` (128/256) = faster
- Larger `--comparison-size` (512/1024) = more accurate
- `canny`: Fastest feature method
- `hog`: Best quality, slower
- Cache rebuilds only when parameters change
- Install PyTurboJPEG for 3x faster JPEG loading (most image datasets)
- Use `--use-ivf-index` if you search many times vs build once (trade search speed for build time)
- Memory-mapped indexes reduce RAM usage for large datasets

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
- Optional IndexIVFFlat: 16x speedup with 95-99% recall (use `--use-ivf-index`)
- Post-filter for no-repeat (simple, fast, exact)
- Trade-off: IVF faster search but slower index building

## Sources

- [FAISS Library](https://github.com/facebookresearch/faiss)
- [FAISS Documentation](https://faiss.ai/index.html)
- [OpenCV Histogram Comparison](https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html)
- [Canny Edge Detection](https://medium.com/@abhisheksriram845/canny-edge-detection-explained-and-compared-with-opencv-in-python-57a161b4bd19)
- [Histogram of Oriented Gradients](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html)
