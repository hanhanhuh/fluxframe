#!/usr/bin/env python3
"""Benchmark batching speedup for ONNX inference."""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Find MobileNet model
model_path = Path.home() / ".cache/fluxframe/mobilenetv3_small_block4.onnx"
if not model_path.exists():
    print(f"Error: {model_path} not found")
    print("Run fluxframe once to generate the model")
    sys.exit(1)

# Load model
session = ort.InferenceSession(str(model_path))
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# Create dummy images
def preprocess_image(img):
    """Preprocess single image."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    normalized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    return np.transpose(normalized, (2, 0, 1))


# Generate test images
print("Generating test images...")
test_images = []
for i in range(100):
    # Create random colored image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_images.append(preprocess_image(img))

print(f"Generated {len(test_images)} test images")
print()

# Benchmark different batch sizes
batch_sizes = [1, 2, 4, 8, 16]

for batch_size in batch_sizes:
    print(f"Batch size: {batch_size}")

    # Prepare batches
    num_batches = len(test_images) // batch_size

    start = time.time()
    for i in range(num_batches):
        batch = test_images[i * batch_size : (i + 1) * batch_size]
        input_tensor = np.stack(batch, axis=0)  # [batch, 3, 224, 224]

        # Run inference
        _ = session.run([output_name], {input_name: input_tensor})

    elapsed = time.time() - start
    images_per_sec = (num_batches * batch_size) / elapsed
    speedup = images_per_sec / (len(test_images) / (time.time() - start + 1))  # rough

    print(f"  Time: {elapsed:.2f}s for {num_batches * batch_size} images")
    print(f"  Throughput: {images_per_sec:.1f} images/sec")
    if batch_size == 1:
        baseline_throughput = images_per_sec
    else:
        speedup = images_per_sec / baseline_throughput
        print(f"  Speedup vs batch=1: {speedup:.2f}x")
    print()
