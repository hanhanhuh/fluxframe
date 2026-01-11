"""Benchmark different neural network optimization options (FAST version).

Tests:
1. Layer depth (layer 3 vs 4)
2. Pooling methods (avg vs gem, different grid sizes)
3. Comparison sizes (224 vs 128 vs 96)
4. Batching (single vs batched inference)
"""

import time
from pathlib import Path

import cv2
import numpy as np

# Test parameters - FAST
NUM_IMAGES = 20  # Reduced for speed
NUM_WARMUP = 3  # Minimal warmup


def benchmark_layer_depth():
    """Compare MobileNet layer 3 vs layer 4."""
    print("\n" + "=" * 80)
    print("BENCHMARK 1: Layer Depth (Layer 3 vs Layer 4)")
    print("=" * 80)

    try:
        import torch
        from torchvision import models
    except ImportError:
        print("⚠️  PyTorch not available, skipping")
        return

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.eval()

    layer3 = torch.nn.Sequential(*list(model.features[:3]))
    layer4 = torch.nn.Sequential(*list(model.features[:4]))

    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"\nLayer 3 output: {layer3(dummy_input).shape} (48 ch @ 28x28)")
    print(f"Layer 4 output: {layer4(dummy_input).shape} (48 ch @ 14x14)")

    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = layer3(dummy_input)
        start = time.perf_counter()
        for _ in range(NUM_IMAGES):
            _ = layer3(dummy_input)
        layer3_time = (time.perf_counter() - start) / NUM_IMAGES * 1000

        for _ in range(NUM_WARMUP):
            _ = layer4(dummy_input)
        start = time.perf_counter()
        for _ in range(NUM_IMAGES):
            _ = layer4(dummy_input)
        layer4_time = (time.perf_counter() - start) / NUM_IMAGES * 1000

    print(f"\nLayer 3: {layer3_time:.2f} ms/image (4x more spatial values)")
    print(f"Layer 4: {layer4_time:.2f} ms/image (current)")
    print(f"Speedup: {layer3_time / layer4_time:.2f}x slower with layer 3")


def benchmark_pooling_methods():
    """Compare different pooling strategies."""
    print("\n" + "=" * 80)
    print("BENCHMARK 2: Pooling Methods (on CPU)")
    print("=" * 80)

    features = np.random.randn(1, 48, 14, 14).astype(np.float32)

    def avg_pool_spatial(feats, grid=2):
        _, _c, h, w = feats.shape
        grid_h, grid_w = h // grid, w // grid
        result = []
        for i in range(grid):
            for j in range(grid):
                cell = feats[0, :, i * grid_h : (i + 1) * grid_h, j * grid_w : (j + 1) * grid_w]
                result.append(cell.mean(axis=(1, 2)))
        return np.concatenate(result)

    def gem_pool_spatial(feats, grid=2, p=3.0, eps=1e-6):
        _, _c, h, w = feats.shape
        grid_h, grid_w = h // grid, w // grid
        result = []
        for i in range(grid):
            for j in range(grid):
                cell = feats[0, :, i * grid_h : (i + 1) * grid_h, j * grid_w : (j + 1) * grid_w]
                cell_clamped = np.maximum(cell, eps)
                gem = np.power(np.mean(np.power(cell_clamped, p), axis=(1, 2)), 1.0 / p)
                result.append(gem)
        return np.concatenate(result)

    def global_avg_pool(feats):
        return feats[0].mean(axis=(1, 2))

    configs = [
        ("Global avg (1x1)", lambda: global_avg_pool(features), 48),
        ("Avg 2x2 grid (current)", lambda: avg_pool_spatial(features, 2), 192),
        ("GeM 2x2 grid", lambda: gem_pool_spatial(features, 2), 192),
        ("Avg 3x3 grid", lambda: avg_pool_spatial(features, 3), 432),
    ]

    print("\nTiming pooling operations...\n")

    for name, func, dim in configs:
        for _ in range(NUM_WARMUP):
            _ = func()
        start = time.perf_counter()
        for _ in range(NUM_IMAGES * 10):  # Pool is very fast, do more iterations
            _ = func()
        elapsed = (time.perf_counter() - start) / (NUM_IMAGES * 10) * 1000

        print(f"{name:25s} {dim:4d}D  {elapsed:.4f} ms")


def benchmark_comparison_sizes():
    """Note: ONNX models have fixed input size, so testing resize impact."""
    print("\n" + "=" * 80)
    print("BENCHMARK 3: Input Resize Impact (Preprocessing)")
    print("=" * 80)

    # Test resize from different source sizes to 224x224
    test_img_720p = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    test_img_480p = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_img_360p = np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8)

    configs = [
        ("720p → 224px", test_img_720p, 224),
        ("480p → 224px", test_img_480p, 224),
        ("360p → 224px", test_img_360p, 224),
        ("720p → 160px", test_img_720p, 160),
        ("720p → 128px", test_img_720p, 128),
    ]

    print("\nTiming resize + normalize...\n")

    for name, img, target_size in configs:
        for _ in range(NUM_WARMUP):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (target_size, target_size))
            normalized = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            _ = (normalized - mean) / std

        start = time.perf_counter()
        for _ in range(NUM_IMAGES):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (target_size, target_size))
            normalized = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            _ = (normalized - mean) / std
        elapsed = (time.perf_counter() - start) / NUM_IMAGES * 1000

        print(f"{name:20s}: {elapsed:.2f} ms")


def benchmark_batching():
    """Compare single vs batched inference."""
    print("\n" + "=" * 80)
    print("BENCHMARK 4: Batching on CPU")
    print("=" * 80)

    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  ONNX Runtime not available, skipping")
        return

    cache_dir = Path.home() / ".cache" / "fluxframe"
    onnx_path = cache_dir / "mobilenetv3_small_block4.onnx"

    if not onnx_path.exists():
        print("⚠️  ONNX model not found, skipping")
        return

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.intra_op_num_threads = 0
    sess_options.inter_op_num_threads = 1

    session = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    batch_sizes = [1, 2, 4, 8, 16]
    total_images = 64

    print(f"\nProcessing {total_images} images...\n")

    baseline_time = None
    for batch_size in batch_sizes:
        dummy_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
        num_batches = total_images // batch_size

        for _ in range(NUM_WARMUP):
            _ = session.run([output_name], {input_name: dummy_input})

        start = time.perf_counter()
        for _ in range(num_batches):
            _ = session.run([output_name], {input_name: dummy_input})
        total_time = time.perf_counter() - start
        time_per_image = (total_time / total_images) * 1000

        if baseline_time is None:
            baseline_time = time_per_image

        speedup = baseline_time / time_per_image
        print(f"Batch {batch_size:2d}: {time_per_image:.2f} ms/image  ({speedup:.2f}x speedup)")


def benchmark_full_pipeline():
    """Benchmark realistic end-to-end feature extraction."""
    print("\n" + "=" * 80)
    print("BENCHMARK 5: Full Pipeline (realistic)")
    print("=" * 80)

    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  ONNX Runtime not available, skipping")
        return

    cache_dir = Path.home() / ".cache" / "fluxframe"
    onnx_path = cache_dir / "mobilenetv3_small_block4.onnx"

    if not onnx_path.exists():
        print("⚠️  ONNX model not found, skipping")
        return

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.intra_op_num_threads = 0
    sess_options.inter_op_num_threads = 1

    session = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Simulate video frame
    test_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    def process_single(img, target_size=224):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (target_size, target_size))
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        features = session.run([output_name], {input_name: input_tensor})[0]
        return features[0].mean(axis=(1, 2))

    configs = [
        ("224px + 2x2 grid", 224),
        ("160px + 2x2 grid", 160),
        ("128px + global", 128),
        ("96px + global", 96),
    ]

    print("\nEnd-to-end timing (preprocessing + inference + pooling)...\n")

    baseline = None
    for name, size in configs:
        for _ in range(NUM_WARMUP):
            _ = process_single(test_img, size)

        start = time.perf_counter()
        for _ in range(NUM_IMAGES):
            _ = process_single(test_img, size)
        elapsed = (time.perf_counter() - start) / NUM_IMAGES * 1000

        if baseline is None:
            baseline = elapsed

        speedup = baseline / elapsed
        print(f"{name:20s}: {elapsed:.2f} ms/frame  ({speedup:.2f}x vs current)")


if __name__ == "__main__":
    print("Neural Network Optimization Benchmark (FAST)")
    print("=" * 80)

    benchmark_layer_depth()
    benchmark_pooling_methods()
    benchmark_comparison_sizes()
    benchmark_batching()
    benchmark_full_pipeline()

    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print("""
Expected outcomes:

1. **Layer 3 vs 4**: Layer 3 slightly slower, same feature dim after pooling
   → Stick with Layer 4 (current choice)

2. **Pooling**: Global (1x1) is ~50x faster than spatial grids
   → Try global pooling for ~2x total speedup

3. **Input Size**: Smaller = much faster
   → 128px: ~2.5x speedup, 96px: ~4x speedup

4. **Batching**: Modest gains on CPU
   → Batch 4-8: ~20-30% speedup, worth implementing

5. **Quick Win Combo**:
   - 128px input + global pooling: ~3-4x speedup total
   - 96px input + global pooling: ~5-6x speedup total
   - Add batching: additional 20-30% on top

This could bring MobileNet close to HOG speed!
    """)
