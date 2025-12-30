#!/bin/bash
# Quick test: Does smaller input size hurt quality?
# Compare same video/images with different input sizes

VIDEO="/home/birgit/Downloads/Winter Cycling： Vantaa, Finland - 4K 60fps.mp4"
IMAGES="/home/birgit/fiftyone/open-images-v7/train/data"

source venv/bin/activate

echo "Testing input size impact on quality"
echo "Using GeM + global pooling for all tests"
echo ""

# Test 224px (current)
echo "1/3: Testing 224px input (current baseline)"
time fluxframe "$VIDEO" "$IMAGES" ./output_224px \
  --fps-override 25 --no-repeat --feature-method mobilenet \
  --pooling-method gem --use-global-pooling \
  --demo --demo-seconds 10 --save-samples 5

echo ""

# Test 128px
echo "2/3: Testing 128px input (~2x faster preprocessing)"
time fluxframe "$VIDEO" "$IMAGES" ./output_128px \
  --fps-override 25 --no-repeat --feature-method mobilenet \
  --pooling-method gem --use-global-pooling \
  --comparison-size 128 \
  --demo --demo-seconds 10 --save-samples 5

echo ""

# Test 96px
echo "3/3: Testing 96px input (~4x faster preprocessing)"
time fluxframe "$VIDEO" "$IMAGES" ./output_96px \
  --fps-override 25 --no-repeat --feature-method mobilenet \
  --pooling-method gem --use-global-pooling \
  --comparison-size 96 \
  --demo --demo-seconds 10 --save-samples 5

echo ""
echo "=================================="
echo "Compare outputs:"
echo "  output_224px/comparison_samples/"
echo "  output_128px/comparison_samples/"
echo "  output_96px/comparison_samples/"
echo ""
echo "Check if matches are similar quality"
