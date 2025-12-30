#!/bin/bash
# Comprehensive pooling comparison with visual samples

VIDEO="/home/birgit/Downloads/Winter Cycling： Vantaa, Finland - 4K 60fps.mp4"
IMAGES="/home/birgit/fiftyone/open-images-v7/train/data"

source venv/bin/activate

echo "=================================="
echo "Pooling Method Comparison"
echo "=================================="
echo "Testing all pooling configurations"
echo "Saving comparison samples for visual inspection"
echo ""

# Common parameters
COMMON="--fps-override 25 --no-repeat --edge-weight 1.0 --feature-method mobilenet --demo --demo-seconds 10 --save-samples 10 --sample-interval 25"

# Test 1: Avg 2x2 (baseline)
echo "1/5: Avg + 2x2 grid (baseline)"
time fluxframe "$VIDEO" "$IMAGES" ./output_avg_2x2 \
  $COMMON --pooling-method avg --spatial-grid 2

echo ""

# Test 2: Avg 3x3
echo "2/5: Avg + 3x3 grid"
time fluxframe "$VIDEO" "$IMAGES" ./output_avg_3x3 \
  $COMMON --pooling-method avg --spatial-grid 3

echo ""

# Test 3: Avg global (no spatial grid)
echo "3/5: Avg + global (1x1, fastest)"
time fluxframe "$VIDEO" "$IMAGES" ./output_avg_global \
  $COMMON --pooling-method avg --use-global-pooling

echo ""

# Test 4: GeM 2x2
echo "4/5: GeM + 2x2 grid"
time fluxframe "$VIDEO" "$IMAGES" ./output_gem_2x2 \
  $COMMON --pooling-method gem --spatial-grid 2

echo ""

# Test 5: GeM global
echo "5/5: GeM + global (1x1)"
time fluxframe "$VIDEO" "$IMAGES" ./output_gem_global \
  $COMMON --pooling-method gem --use-global-pooling

echo ""
echo "=================================="
echo "All tests complete!"
echo "=================================="
echo ""
echo "Comparison samples saved in:"
echo "  - output_avg_2x2/comparison_samples/"
echo "  - output_avg_3x3/comparison_samples/"
echo "  - output_avg_global/comparison_samples/"
echo "  - output_gem_2x2/comparison_samples/"
echo "  - output_gem_global/comparison_samples/"
echo ""
echo "Creating side-by-side comparison images..."
