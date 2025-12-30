#!/bin/bash
# FAST test: use only 100 images to test quality difference

VIDEO="/home/birgit/Downloads/Winter Cycling： Vantaa, Finland - 4K 60fps.mp4"

# Create tiny test image folder
TEST_IMAGES="/tmp/test_images_100"
mkdir -p "$TEST_IMAGES"

echo "Creating test set of 100 random images..."
find /home/birgit/fiftyone/open-images-v7/train/data -name "*.jpg" | head -100 | while read img; do
    cp "$img" "$TEST_IMAGES/"
done

source venv/bin/activate

echo ""
echo "Testing 3 input sizes on SAME 100 images"
echo "========================================"

# Test 224px
echo "1/3: 224px (current)"
time fluxframe "$VIDEO" "$TEST_IMAGES" ./test_224px \
  --fps-override 25 --feature-method mobilenet \
  --pooling-method gem --use-global-pooling \
  --demo --demo-seconds 5 --save-samples 3 --no-repeat

echo ""

# Test 128px
echo "2/3: 128px"
time fluxframe "$VIDEO" "$TEST_IMAGES" ./test_128px \
  --fps-override 25 --feature-method mobilenet \
  --pooling-method gem --use-global-pooling \
  --comparison-size 128 \
  --demo --demo-seconds 5 --save-samples 3 --no-repeat

echo ""

# Test 96px
echo "3/3: 96px"
time fluxframe "$VIDEO" "$TEST_IMAGES" ./test_96px \
  --fps-override 25 --feature-method mobilenet \
  --pooling-method gem --use-global-pooling \
  --comparison-size 96 \
  --demo --demo-seconds 5 --save-samples 3 --no-repeat

echo ""
echo "========================================"
echo "Compare samples in:"
echo "  test_224px/comparison_samples/"
echo "  test_128px/comparison_samples/"
echo "  test_96px/comparison_samples/"
