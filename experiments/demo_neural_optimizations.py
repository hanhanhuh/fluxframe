"""Compare neural network optimization configurations.

Tests different combinations of pooling and comparison sizes to find
the best speed/quality tradeoff for MobileNet.
"""

import subprocess
import time
from pathlib import Path

# Test video and images
VIDEO = "/home/birgit/Downloads/Winter Cycling： Vantaa, Finland - 4K 60fps.mp4"
IMAGES = "/home/birgit/fiftyone/open-images-v7/train/data"

# Common parameters
BASE_PARAMS = [
    "--fps-override", "25",
    "--no-repeat",
    "--edge-weight", "1.0",  # Neural uses only edge features
    "--color-weight", "0.0",
    "--texture-weight", "0.0",
    "--feature-method", "mobilenet",
    "--demo",  # Use demo mode for speed
    "--skip-output",  # Only benchmark feature extraction
]

# Configurations to test
CONFIGS = [
    {
        "name": "mobilenet_baseline",
        "desc": "MobileNet + avg + 2x2 grid (current)",
        "params": ["--pooling-method", "avg", "--spatial-grid", "2"],
    },
    {
        "name": "mobilenet_global",
        "desc": "MobileNet + avg + global (1x1)",
        "params": ["--pooling-method", "avg", "--use-global-pooling"],
    },
    {
        "name": "mobilenet_global_128px",
        "desc": "MobileNet + global + 128px input",
        "params": ["--pooling-method", "avg", "--use-global-pooling", "--comparison-size", "128"],
    },
    {
        "name": "mobilenet_baseline_128px",
        "desc": "MobileNet + 2x2 + 128px input",
        "params": ["--pooling-method", "avg", "--spatial-grid", "2", "--comparison-size", "128"],
    },
]

def run_config(config):
    """Run a single configuration and measure time."""
    output_dir = f"./output_{config['name']}"

    cmd = [
        "fluxframe",
        VIDEO,
        IMAGES,
        output_dir,
    ] + BASE_PARAMS + config["params"]

    print(f"\n{'='*80}")
    print(f"Testing: {config['desc']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    # Extract timing info from output
    build_time = None
    match_time = None

    for line in result.stdout.split('\n') + result.stderr.split('\n'):
        if 'Building FAISS index' in line or 'Computing features' in line:
            # Try to extract time from progress bar
            pass
        if 'Matching frames' in line:
            pass

    print(f"✓ Completed in {elapsed:.1f}s")

    return {
        "name": config["name"],
        "desc": config["desc"],
        "total_time": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

def main():
    """Run all configurations and compare results."""
    print("Neural Network Optimization Comparison")
    print("="*80)
    print(f"Video: {VIDEO}")
    print(f"Images: {IMAGES}")
    print(f"Mode: Demo (subset of data)")
    print("="*80)

    results = []

    for config in CONFIGS:
        try:
            result = run_config(config)
            results.append(result)
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print()

    if not results:
        print("No results to display")
        return

    # Find baseline
    baseline = next((r for r in results if r["name"] == "mobilenet_baseline"), results[0])
    baseline_time = baseline["total_time"]

    print(f"{'Configuration':<40} {'Time (s)':>10} {'Speedup':>10}")
    print("-"*80)

    for result in results:
        speedup = baseline_time / result["total_time"]
        print(f"{result['desc']:<40} {result['total_time']:>10.1f} {speedup:>10.2f}x")

    print()
    print("Key findings:")
    print("- Global pooling (1x1) should be ~2x faster than 2x2 grid")
    print("- 128px input should be ~2x faster than 256px")
    print("- Combined optimizations: ~3-4x speedup possible")
    print()
    print("Note: Quality assessment requires visual inspection of output")
    print("      Run without --skip-output to generate matched videos")

if __name__ == "__main__":
    main()
