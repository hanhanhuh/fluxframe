"""
Unit and integration tests for video_image_matcher.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fluxframe import ImageMatcher, VideoImageMatcher


class TestImageMatcher:
    """Test the ImageMatcher class for similarity computation."""

    def test_aspect_ratio_crop_wider_image(self):
        """Test cropping a wide image to narrower aspect ratio."""
        matcher = ImageMatcher()

        # Create a 200x100 image (aspect ratio 2.0)
        img = np.zeros((100, 200, 3), dtype=np.uint8)

        # Crop to aspect ratio 1.5 (narrower)
        cropped = matcher.aspect_ratio_crop(img, 1.5)

        # Expected: height stays 100, width becomes 150
        assert cropped.shape[0] == 100
        assert cropped.shape[1] == 150

    def test_aspect_ratio_crop_taller_image(self):
        """Test cropping a tall image to wider aspect ratio."""
        matcher = ImageMatcher()

        # Create a 100x200 image (aspect ratio 0.5)
        img = np.zeros((200, 100, 3), dtype=np.uint8)

        # Crop to aspect ratio 1.0 (wider)
        cropped = matcher.aspect_ratio_crop(img, 1.0)

        # Expected: width stays 100, height becomes 100
        assert cropped.shape[0] == 100
        assert cropped.shape[1] == 100

    def test_aspect_ratio_crop_no_change(self):
        """Test that similar aspect ratios don't change the image."""
        matcher = ImageMatcher()

        # Create a 100x100 image
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Crop to nearly same aspect ratio
        cropped = matcher.aspect_ratio_crop(img, 1.005)

        # Should be unchanged
        assert cropped.shape == img.shape

    def test_compute_edge_features_shape(self):
        """Test that edge features have correct shape."""
        matcher = ImageMatcher()

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        features = matcher.compute_edge_features(img)

        # Should return 256-bin histogram
        assert features.shape == (256,)
        assert np.all(features >= 0) and np.all(features <= 1)

    def test_compute_texture_features_shape(self):
        """Test that texture features have correct shape."""
        matcher = ImageMatcher()

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        features = matcher.compute_texture_features(img)

        # Should return 64-bin histogram
        assert features.shape == (64,)
        assert np.all(features >= 0) and np.all(features <= 1)

    def test_compute_color_features_shape(self):
        """Test that color features have correct shape."""
        matcher = ImageMatcher()

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        features = matcher.compute_color_features(img)

        # Should return 8*8*8 = 512-bin histogram
        assert features.shape == (512,)
        assert np.all(features >= 0) and np.all(features <= 1)

    def test_compute_all_features(self):
        """Test that compute_all_features returns all three feature types."""
        matcher = ImageMatcher()

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        features = matcher.compute_all_features(img)

        assert 'edge' in features
        assert 'texture' in features
        assert 'color' in features
        assert features['edge'].shape == (256,)
        assert features['texture'].shape == (64,)
        assert features['color'].shape == (512,)

    def test_similarity_identical_images(self):
        """Test that identical images have maximum similarity."""
        matcher = ImageMatcher()

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        similarity = matcher.compute_similarity(img, img)

        # Should be very close to 1.0 (allowing for small numerical errors)
        assert similarity > 0.99

    def test_similarity_completely_different(self):
        """Test that very different images have lower similarity than identical."""
        matcher = ImageMatcher()

        # All black vs all white
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

        different_similarity = matcher.compute_similarity(img1, img2)
        identical_similarity = matcher.compute_similarity(img1, img1)

        # Different images should have lower similarity than identical
        assert different_similarity < identical_similarity

    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1."""
        matcher = ImageMatcher(edge_weight=2.0, texture_weight=3.0, color_weight=5.0)

        # Should sum to 1.0
        total = matcher.edge_weight + matcher.texture_weight + matcher.color_weight
        assert abs(total - 1.0) < 1e-6

    def test_compare_histograms_identical(self):
        """Test histogram comparison with identical histograms."""
        matcher = ImageMatcher()

        # OpenCV requires float32 type for compareHist
        hist1 = np.random.rand(256).astype(np.float32)
        hist1 = hist1 / hist1.sum()  # Normalize

        similarity = matcher.compare_histograms(hist1, hist1)

        # Should be very close to 1.0
        assert similarity > 0.99


class TestVideoImageMatcherInit:
    """Test VideoImageMatcher initialization and setup."""

    def test_initialization_creates_output_dir(self):
        """Test that initialization creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            # Create dummy paths
            video_path.touch()
            images_path.mkdir()

            _ = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path)
            )

            assert output_path.exists()
            assert output_path.is_dir()

    def test_seed_sets_reproducibility(self):
        """Test that seed parameter ensures reproducible random choices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path1 = Path(tmpdir) / "output1"
            output_path2 = Path(tmpdir) / "output2"

            video_path.touch()
            images_path.mkdir()

            # Create test images for FAISS index
            for i in range(3):
                img_path = images_path / f"img{i+1}.jpg"
                import cv2
                import numpy as np
                img = np.ones((100, 100, 3), dtype=np.uint8) * (i * 50)
                cv2.imwrite(str(img_path), img)

            # Create two matchers with same seed
            matcher1 = VideoImageMatcher(
                str(video_path), str(images_path), str(output_path1), seed=42
            )
            matcher2 = VideoImageMatcher(
                str(video_path), str(images_path), str(output_path2), seed=42
            )

            # Build indices
            image_files = matcher1.get_image_files()
            matcher1._build_faiss_index(image_files)
            matcher2._build_faiss_index(image_files)

            # Test that random selection is same
            top_matches = [
                (str(images_path / "img1.jpg"), 0.9),
                (str(images_path / "img2.jpg"), 0.8),
                (str(images_path / "img3.jpg"), 0.7)
            ]

            selected1 = matcher1.select_match(top_matches)
            # Reset random state
            matcher2 = VideoImageMatcher(
                str(video_path), str(images_path), str(output_path2), seed=42
            )
            matcher2._build_faiss_index(image_files)
            selected2 = matcher2.select_match(top_matches)

            assert selected1 == selected2

    def test_checkpoint_paths_created(self):
        """Test that checkpoint paths are set correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path)
            )

            assert matcher.checkpoint_path == output_path / 'checkpoint.json'
            assert matcher.results_path == output_path / 'results.json'
            # FAISS cache files
            assert matcher.cache_metadata_path == output_path / 'cache_metadata.json'
            assert matcher.faiss_index_path == output_path / 'faiss_index.bin'
            assert matcher.vectors_path == output_path / 'vectors.npy'


class TestCheckpointing:
    """Test checkpoint save/load functionality."""

    def test_save_and_load_checkpoint(self):
        """Test that checkpoints can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path)
            )

            # Create and save checkpoint
            # Note: JSON converts tuples to lists
            checkpoint = {
                'frame_000000': {
                    'top_matches': [['img1.jpg', 0.9], ['img2.jpg', 0.8]],
                    'selected': 'img1.jpg'
                }
            }
            matcher.save_checkpoint(checkpoint)

            # Load it back
            loaded = matcher.load_checkpoint()

            assert loaded == checkpoint
            assert loaded['frame_000000']['selected'] == 'img1.jpg'
            assert loaded['frame_000000']['top_matches'][0] == ['img1.jpg', 0.9]

    def test_load_nonexistent_checkpoint(self):
        """Test loading checkpoint when none exists returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path)
            )

            loaded = matcher.load_checkpoint()
            assert loaded == {}


class TestImageSelection:
    """Test image selection logic."""

    def test_select_match_basic(self):
        """Test basic random selection from top matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create test images
            import cv2
            import numpy as np
            for i in range(3):
                img = np.ones((100, 100, 3), dtype=np.uint8) * (i * 50)
                cv2.imwrite(str(images_path / f"img{i+1}.jpg"), img)

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                seed=42
            )

            # Build index
            matcher._build_faiss_index(matcher.get_image_files())

            top_matches = [
                (str(images_path / 'img1.jpg'), 0.9),
                (str(images_path / 'img2.jpg'), 0.8),
                (str(images_path / 'img3.jpg'), 0.7)
            ]

            selected = matcher.select_match(top_matches)

            # Should be one of the options
            assert selected in [str(images_path / f'img{i}.jpg') for i in range(1, 4)]

    def test_select_match_with_threshold(self):
        """Test selection respects similarity threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create test images
            import cv2
            import numpy as np
            for i in range(3):
                img = np.ones((100, 100, 3), dtype=np.uint8) * (i * 50)
                cv2.imwrite(str(images_path / f"img{i+1}.jpg"), img)

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                similarity_threshold=0.85
            )

            # Build index
            matcher._build_faiss_index(matcher.get_image_files())

            top_matches = [
                (str(images_path / 'img1.jpg'), 0.9),   # Above threshold
                (str(images_path / 'img2.jpg'), 0.8),   # Below threshold
                (str(images_path / 'img3.jpg'), 0.7)    # Below threshold
            ]

            selected = matcher.select_match(top_matches)

            # Should only select img1
            assert selected == str(images_path / 'img1.jpg')

    def test_select_match_all_below_threshold(self):
        """Test selection when all matches are below threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create test images
            import cv2
            import numpy as np
            for i in range(3):
                img = np.ones((100, 100, 3), dtype=np.uint8) * (i * 50)
                cv2.imwrite(str(images_path / f"img{i+1}.jpg"), img)

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                similarity_threshold=0.95
            )

            # Build index
            matcher._build_faiss_index(matcher.get_image_files())

            top_matches = [
                (str(images_path / 'img1.jpg'), 0.9),
                (str(images_path / 'img2.jpg'), 0.8),
                (str(images_path / 'img3.jpg'), 0.7)
            ]

            selected = matcher.select_match(top_matches)

            # Should still return a match (fallback behavior)
            assert selected is not None
            assert isinstance(selected, str)

    def test_select_match_no_repeat(self):
        """Test no-repeat mode excludes used images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create test images
            import cv2
            import numpy as np
            for i in range(3):
                img = np.ones((100, 100, 3), dtype=np.uint8) * (i * 50)
                cv2.imwrite(str(images_path / f"img{i+1}.jpg"), img)

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                no_repeat=True,
                seed=42
            )

            # Build index
            matcher._build_faiss_index(matcher.get_image_files())

            top_matches = [
                (str(images_path / 'img1.jpg'), 0.9),
                (str(images_path / 'img2.jpg'), 0.8),
                (str(images_path / 'img3.jpg'), 0.7)
            ]

            # With no_repeat mode, always picks the best (first) match
            # Filtering happens in find_top_matches, not in select_match
            selected1 = matcher.select_match(top_matches)
            assert selected1 == str(images_path / 'img1.jpg')  # Best match

            # Calling again with same list picks same best match
            # (In real usage, find_top_matches would filter this out)
            selected2 = matcher.select_match(top_matches)
            assert selected2 == str(images_path / 'img1.jpg')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
