# FluxFrame

Create temporal video collages by replacing each frame with similar images from large datasets. FluxFrame transforms videos into artistic visual remixes where the original motion and composition remain recognizable, but constructed entirely from different source images.

Perfect for experimental video art, temporal mosaics, and creative visual storytelling. Optimized for CPU performance with feature pre-computation, making it suitable for matching against 0.5M+ images.

## Features

- **Multiple feature extraction methods**:
  - **Canny**: Fast edge histogram (default, no spatial info)
  - **Spatial Pyramid**: 4x4 grid of edge histograms (preserves spatial layout)
  - **HOG**: Histogram of Oriented Gradients (best motion preservation for temporal mosaics)
- **Multi-metric similarity**: Combines edge/structure, texture (Sobel gradients), and color (HSV) features
- **Weighted metrics**: Configurable weights for each similarity component
- **Parallel processing**: Multiprocessing support for fast similarity computation across CPU cores
- **Aspect-ratio preserving**: Uses center cropping instead of adding bars
- **CPU-optimized**: Fast comparison using OpenCV with efficient histogram-based methods
- **Feature pre-computation**: Computes image features once and caches them for reuse (100x+ speedup for large datasets)
- **Aspect-ratio independent caching**: Caches features for multiple aspect ratios, works with different videos without recomputation
- **Memory efficient**: Processes video frame-by-frame without loading all frames into memory
- **Batched checkpointing**: Saves progress every N frames (configurable) for efficient resume
- **FPS override**: Optional custom frame rate for output video
- **Reproducible**: Optional random seed for consistent results
- **Top-N selection**: Finds top N most similar images and randomly selects one
- **No-repeat mode**: Optional constraint to use each image only once
- **Similarity threshold**: Optional minimum similarity for selection
- **Demo mode**: Test on subset of video/images for quick validation
- **Dual output**: Generates both video and individual frame images
- **Type-safe**: Full mypy strict mode compliance with comprehensive type hints

## Installation

Install the package in development mode:

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install -r requirements.txt
```

## Package Structure

```
fluxframe/
├── src/fluxframe/
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── processor.py        # Main processing pipeline
│   ├── matcher.py          # Image similarity computation
│   └── models.py           # Pydantic data models
├── tests/
│   ├── test_matcher.py     # Unit tests
│   └── test_e2e.py         # End-to-end tests
└── pyproject.toml          # Package configuration
```

## Usage

Basic usage via command-line:

```bash
fluxframe <video_file> <image_folder> <output_dir>
```

Or as a Python module:

```bash
python -m fluxframe.cli <video_file> <image_folder> <output_dir>
```

### Examples

**Simple matching with default settings (Canny features):**
```bash
fluxframe input.mp4 /path/to/open-images ./output
```

**Temporal mosaic with motion preservation (HOG features):**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --feature-method hog
```

**Balanced speed/quality (Spatial Pyramid features):**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --feature-method spatial_pyramid
```

**Fast processing with parallelization:**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --feature-method canny --num-workers 16
```

**Custom FPS for output video:**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --fps-override 30
```

**Emphasize edge/structure similarity:**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --edge-weight 0.6 --texture-weight 0.2 --color-weight 0.2
```

**Emphasize color similarity:**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --edge-weight 0.1 --texture-weight 0.1 --color-weight 0.8
```

**Use each image only once with similarity threshold:**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --no-repeat --threshold 0.5 --top-n 20
```

**High-quality comparison (slower but more accurate):**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --comparison-size 512 --feature-method hog
```

**Only compute matches (skip video generation):**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --skip-output
```

**Demo mode (test with small subset):**
```bash
fluxframe input.mp4 /path/to/images ./output \
  --demo --demo-seconds 30 --demo-images 500
```

### Command-line Arguments

**Required:**
- `video`: Path to input video file
- `images`: Path to folder containing images (flat structure, no recursive search)
- `output`: Output directory

**Optional:**
- `--feature-method`: Feature extraction method - `canny` (fast), `spatial_pyramid` (balanced), `hog` (best motion) (default: canny)
- `--top-n`: Number of top similar images to consider (default: 10)
- `--edge-weight`: Weight for edge/structure similarity 0-1 (default: 0.33)*
- `--texture-weight`: Weight for texture similarity 0-1 (default: 0.33)*
- `--color-weight`: Weight for color similarity 0-1 (default: 0.34)*
- `--threshold`: Minimum similarity threshold for selection 0-1 (default: 0.0)
- `--no-repeat`: Use each image only once
- `--comparison-size`: Resize images to this size for comparison (default: 256, larger = slower but more accurate)
- `--num-workers`: Number of parallel workers (default: auto-detect CPU count)
- `--fps-override`: Override output video FPS (default: use input video FPS)
- `--skip-output`: Skip output generation (only compute matches)
- `--demo`: Enable demo mode (process only subset of video and images)
- `--demo-seconds`: Number of seconds to process in demo mode (default: 20)
- `--demo-images`: Number of images to use in demo mode (default: 1000)
- `--checkpoint-batch`: Save checkpoint every N frames (default: 10)
- `--seed`: Random seed for reproducibility (default: None)

*Note: Weights are automatically normalized to sum to 1.0, so you can use any values (e.g., `2:1:1` becomes `0.5:0.25:0.25`).

### Feature Methods Comparison

| Method | Speed | Motion Preservation | Feature Size | Best For |
|--------|-------|---------------------|--------------|----------|
| `canny` | ⭐⭐⭐⭐⭐ | ❌ Poor | 256 values | Fast processing, static scenes |
| `spatial_pyramid` | ⭐⭐⭐⭐ | ⭐⭐⭐ Good | 512 values | Balanced speed/quality |
| `hog` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ Excellent | ~1296 values | Temporal mosaics, motion preservation |

## Programmatic Usage

You can also use the package programmatically:

```python
from fluxframe.processor import VideoImageMatcher

# Initialize matcher
matcher = VideoImageMatcher(
    video_path="input.mp4",
    image_folder="/path/to/images",
    output_dir="./output",
    top_n=10,
    edge_weight=0.33,
    texture_weight=0.33,
    color_weight=0.34,
    similarity_threshold=0.5,
    no_repeat=True,
    seed=42
)

# Process video frames
checkpoint = matcher.process()

# Generate output video
matcher.generate_output(checkpoint)
```

## How It Works

### 1. Pre-computation (First Run Only)
The package pre-computes feature vectors for all images in the dataset:
- Crops images to common aspect ratios (16:9, 4:3, 1:1, 9:16, 3:4)
- Computes edge, texture, and color features for each crop
- Saves to `image_features.pkl` for instant reuse
- **This happens once** - subsequent runs with different videos reuse the cache

### 2. Frame-by-Frame Processing
Processes video frames one at a time without loading entire video into memory:
- Reads video metadata (FPS, resolution, total frames)
- Opens video stream and processes each frame sequentially
- Releases memory after each frame

### 3. Similarity Computation
For each video frame:

**Color Similarity:**
- Converts images to HSV color space
- Computes 3D color histogram (8x8x8 bins)
- Uses histogram correlation for comparison

**Edge/Contour Similarity:**
- Applies Canny edge detection
- Computes edge histogram
- Uses histogram correlation for comparison

**Texture Similarity:**
- Computes Sobel gradients (approximation of Local Binary Patterns)
- Analyzes gradient magnitude as texture descriptor
- Uses histogram correlation for comparison

**Combined Score:**
```
similarity = edge_weight * edge_sim + texture_weight * texture_sim + color_weight * color_sim
```

### 3. Aspect Ratio Handling
- Images are center-cropped to match video aspect ratio
- No bars are added (preserves image quality)
- Comparison is done on resized images for speed

### 4. Top-N Selection
- Computes similarity for all images in the folder
- Selects top N most similar images
- Filters by similarity threshold (if specified)
- Randomly picks one from the top matches
- Optionally excludes already-used images

### 5. Checkpointing
- Saves results after each frame to `checkpoint.json`
- Stores top matches and selected image for each frame
- Can resume processing if interrupted
- Enables parameter experimentation without recomputing

### 6. Output Generation
- Creates output video with matched images
- Saves individual frames as images named by frame number
- Images are cropped and resized to match video dimensions

## Output Structure

```
output_dir/
├── checkpoint.json           # Processing checkpoint
├── results.json             # Final results
├── <video_name>_matched.mp4 # Output video
└── matched_frames/          # Individual frame images
    ├── frame_000000.jpg
    ├── frame_000001.jpg
    └── ...
```

## Development

Run tests:

```bash
pytest tests/ -v
```

Run linters:

```bash
ruff check src/ tests/
mypy src/video_image_matcher --ignore-missing-imports
```

## Resuming After Errors

The package automatically saves progress. If interrupted, simply run the same command again - it will resume from the last checkpoint.

To restart with different parameters (e.g., different weights):
1. Delete `checkpoint.json` in the output directory
2. Run with new parameters

## Performance Tips

- **Faster processing**: Use smaller `--comparison-size` (e.g., 128 or 256)
- **Better quality**: Use larger `--comparison-size` (e.g., 512 or 1024)
- **Large datasets**: The script processes all images for each frame - consider filtering your image dataset first
- **CPU usage**: OpenCV operations are optimized for CPU and use SIMD instructions when available

## Technical Details

The package uses CPU-efficient algorithms based on research:

**Color Comparison:**
- HSV color histograms with correlation metric (OpenCV)
- Higher correlation = more similar colors

**Edge Detection:**
- Canny edge detection for robust contour extraction
- Histogram-based comparison for edge distribution

**Texture Analysis:**
- Sobel gradient magnitude as texture descriptor
- Fast alternative to Local Binary Patterns

All comparisons use normalized histograms and correlation metrics for consistent 0-1 similarity scores.

## Sources

Research for this implementation:
- [Image Comparison Using OpenCV and Python](https://www.pythontutorials.net/blog/compare-similarity-of-images-using-opencv-with-python/)
- [OpenCV Histogram Comparison](https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html)
- [Image Hashing with OpenCV](https://github.com/JohannesBuchner/imagehash)
- [Canny Edge Detection with OpenCV](https://medium.com/@abhisheksriram845/canny-edge-detection-explained-and-compared-with-opencv-in-python-57a161b4bd19)
- [OpenCV img_hash module](https://docs.opencv.org/3.4/d4/d93/group__img__hash.html)
