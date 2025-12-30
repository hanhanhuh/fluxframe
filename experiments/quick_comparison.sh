#!/bin/bash
# Quick comparison of optimized MobileNet configurations

VIDEO="/home/birgit/Downloads/Winter Cycling： Vantaa, Finland - 4K 60fps.mp4"
IMAGES="/home/birgit/fiftyone/open-images-v7/train/data"

source venv/bin/activate

echo "=================================="
echo "Quick Neural Optimization Tests"
echo "=================================="
echo ""

# Test 1: Current baseline (avg 2x2)
echo "1. MobileNet + avg + 2x2 grid (baseline)"
time fluxframe "$VIDEO" "$IMAGES" ./output_mobilenet_avg_2x2 \
  --fps-override 25 --no-repeat --edge-weight 1.0 \
  --feature-method mobilenet --pooling-method avg --spatial-grid 2 \
  --demo --demo-seconds 10

echo ""
echo "=================================="
echo ""

# Test 2: GeM 2x2 (better quality per user)
echo "2. MobileNet + GeM + 2x2 grid (better quality)"
time fluxframe "$VIDEO" "$IMAGES" ./output_mobilenet_gem_2x2 \
  --fps-override 25 --no-repeat --edge-weight 1.0 \
  --feature-method mobilenet --pooling-method gem --spatial-grid 2 \
  --demo --demo-seconds 10

echo ""
echo "=================================="
echo ""

# Test 3: Global pooling for speed (avg)
echo "3. MobileNet + avg + global (5x faster pooling)"
time fluxframe "$VIDEO" "$IMAGES" ./output_mobilenet_avg_global \
  --fps-override 25 --no-repeat --edge-weight 1.0 \
  --feature-method mobilenet --pooling-method avg --use-global-pooling \
  --demo --demo-seconds 10

echo ""
echo "=================================="
echo ""

# Test 4: Global pooling with GeM (speed + quality compromise)
echo "4. MobileNet + GeM + global (speed + quality)"
time fluxframe "$VIDEO" "$IMAGES" ./output_mobilenet_gem_global \
  --fps-override 25 --no-repeat --edge-weight 1.0 \
  --feature-method mobilenet --pooling-method gem --use-global-pooling \
  --demo --demo-seconds 10

echo ""
echo "=================================="
echo "All tests complete!"
echo "=================================="
echo ""
echo "Check output directories for visual comparison:"
echo "  - output_mobilenet_avg_2x2/     (baseline)"
echo "  - output_mobilenet_gem_2x2/     (better quality, slower)"
echo "  - output_mobilenet_avg_global/  (faster)"
echo "  - output_mobilenet_gem_global/  (balanced)"
