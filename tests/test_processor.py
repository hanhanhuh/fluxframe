"""
Comprehensive tests for FAISS-based VideoImageMatcher processor.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from fluxframe import VideoImageMatcher


def create_test_image(path: Path, color: tuple[int, int, int] = (128, 128, 128)) -> None:
    """Create a test image with given color.

    Args:
        path: Path to save image.
        color: BGR color tuple.
    """
    img = np.ones((100, 100, 3), dtype=np.uint8)
    img[:] = color
    cv2.imwrite(str(path), img)


def create_test_video(path: Path, fps: float = 30.0, num_frames: int = 10) -> None:
    """Create a test video file.

    Args:
        path: Path to save video.
        fps: Frames per second.
        num_frames: Number of frames to generate.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, fps, (640, 480))

    for i in range(num_frames):
        # Create frame with varying brightness
        brightness = int(255 * (i / num_frames))
        frame = np.ones((480, 640, 3), dtype=np.uint8) * brightness
        writer.write(frame)

    writer.release()


class TestFAISSCaching:
    """Test FAISS index caching and validation."""

    def test_cache_key_generation(self):
        """Test that cache keys are consistent and deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            matcher1 = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                comparison_size=256
            )

            matcher2 = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                comparison_size=256
            )

            # Create test images
            img1 = images_path / "img1.jpg"
            create_test_image(img1)

            files = [img1]

            # Same parameters should generate same cache key
            key1 = matcher1._generate_cache_key(files)
            key2 = matcher2._generate_cache_key(files)

            assert key1 == key2

    def test_cache_key_changes_with_parameters(self):
        """Test that cache key changes when parameters change."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create test image
            img1 = images_path / "img1.jpg"
            create_test_image(img1)
            files = [img1]

            # Different comparison_size
            matcher1 = VideoImageMatcher(
                str(video_path), str(images_path), str(output_path),
                comparison_size=256
            )
            matcher2 = VideoImageMatcher(
                str(video_path), str(images_path), str(output_path),
                comparison_size=128
            )

            key1 = matcher1._generate_cache_key(files)
            key2 = matcher2._generate_cache_key(files)

            assert key1 != key2

    def test_cache_validation_detects_missing_files(self):
        """Test that cache validation fails when cache files don't exist."""
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

            # Create test image
            img1 = images_path / "img1.jpg"
            create_test_image(img1)
            files = [img1]

            # Cache doesn't exist yet
            assert not matcher._validate_cache(files)

    def test_faiss_index_build_and_load(self):
        """Test that FAISS index can be built and loaded from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create test images
            for i in range(5):
                img_path = images_path / f"img{i}.jpg"
                create_test_image(img_path, color=(i * 50, i * 50, i * 50))

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path)
            )

            image_files = matcher.get_image_files()
            assert len(image_files) == 5

            # Build index
            matcher._build_faiss_index(image_files)

            assert matcher.faiss_index is not None
            assert len(matcher.image_paths) == 5
            assert matcher.vectors is not None
            assert matcher.vectors.shape[0] == 5

            # Verify cache files exist
            assert matcher.cache_metadata_path.exists()
            assert matcher.faiss_index_path.exists()
            assert matcher.vectors_path.exists()

            # Create new matcher and load from cache
            matcher2 = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path)
            )

            matcher2._build_faiss_index(image_files)

            # Should have loaded from cache
            assert matcher2.faiss_index is not None
            assert len(matcher2.image_paths) == 5
            assert matcher2.vectors is not None


class TestAdaptiveSearch:
    """Test adaptive search depth mechanism."""

    def test_search_depth_initialization(self):
        """Test that search depth is initialized correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                top_n=10
            )

            assert matcher.current_search_k == 10
            assert matcher.max_search_k == 50  # min(10 * 5, 100)
            assert matcher.consecutive_failures == 0
            assert matcher.consecutive_successes == 0

    def test_search_depth_expansion_on_failure(self):
        """Test that search depth expands on failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                top_n=10
            )

            initial_k = matcher.current_search_k

            # Trigger failure
            matcher._adjust_search_depth(success=False)

            # Should increase by 50%
            assert matcher.current_search_k > initial_k
            assert matcher.consecutive_failures == 1
            assert matcher.consecutive_successes == 0

    def test_search_depth_reduction_on_success(self):
        """Test that search depth reduces on success streaks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                top_n=10
            )

            # Artificially increase search depth
            matcher.current_search_k = 30

            # Need 3 successes to trigger reduction
            matcher._adjust_search_depth(success=True)
            assert matcher.current_search_k == 30  # Not yet

            matcher._adjust_search_depth(success=True)
            assert matcher.current_search_k == 30  # Not yet

            matcher._adjust_search_depth(success=True)
            assert matcher.current_search_k < 30  # Should reduce now

    def test_search_depth_caps_at_max(self):
        """Test that search depth doesn't exceed max."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                top_n=10
            )

            # Trigger many failures
            for _ in range(20):
                matcher._adjust_search_depth(success=False)

            # Should cap at max_search_k
            assert matcher.current_search_k <= matcher.max_search_k


class TestFrameRateControl:
    """Test frame rate control and skip interval."""

    def test_fps_override_skip_interval(self):
        """Test that skip interval is calculated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            # Create test video at 30 fps
            create_test_video(video_path, fps=30.0, num_frames=30)
            images_path.mkdir()

            # Create test images
            for i in range(10):
                create_test_image(images_path / f"img{i}.jpg")

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                fps_override=10.0  # Should skip every 3rd frame
            )

            # Get video info to trigger skip interval calculation
            video_info = matcher.get_video_info()
            matcher.input_fps = video_info.fps

            # Calculate skip interval
            if matcher.fps_override is not None:
                matcher.input_fps_skip_interval = max(
                    1, round(matcher.input_fps / matcher.fps_override)
                )

            # 30 fps / 10 fps = 3
            assert matcher.input_fps_skip_interval == 3

    def test_no_fps_override(self):
        """Test that skip interval is 1 when no fps_override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            create_test_video(video_path, fps=30.0, num_frames=30)
            images_path.mkdir()

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                fps_override=None
            )

            assert matcher.input_fps_skip_interval == 1


class TestIntelligentFallback:
    """Test intelligent multi-level fallback selection."""

    def test_select_match_threshold_compliant(self):
        """Test selection when matches are above threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create some test images
            for i in range(5):
                create_test_image(images_path / f"img{i}.jpg")

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                similarity_threshold=0.7,
                seed=42
            )

            # Build minimal index
            matcher._build_faiss_index(matcher.get_image_files())

            top_matches = [
                (str(images_path / "img1.jpg"), 0.9),
                (str(images_path / "img2.jpg"), 0.8),
                (str(images_path / "img3.jpg"), 0.6),
            ]

            selected = matcher.select_match(top_matches)

            # Should be one of the threshold-compliant matches
            assert selected in [str(images_path / "img1.jpg"), str(images_path / "img2.jpg")]

    def test_select_match_below_threshold_fallback(self):
        """Test fallback when all matches are below threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create some test images
            for i in range(5):
                create_test_image(images_path / f"img{i}.jpg")

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                similarity_threshold=0.95,  # Very high threshold
                seed=42
            )

            # Build minimal index
            matcher._build_faiss_index(matcher.get_image_files())

            top_matches = [
                (str(images_path / "img1.jpg"), 0.8),
                (str(images_path / "img2.jpg"), 0.7),
                (str(images_path / "img3.jpg"), 0.6),
            ]

            selected = matcher.select_match(top_matches)

            # Should still return a match (best available)
            assert selected in [str(images_path / f"img{i}.jpg") for i in range(1, 4)]

    def test_select_match_no_repeat_filtering(self):
        """Test that no_repeat filters out used images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create some test images
            for i in range(5):
                create_test_image(images_path / f"img{i}.jpg")

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                no_repeat=True,
                similarity_threshold=0.0,
                seed=42
            )

            # Build minimal index
            matcher._build_faiss_index(matcher.get_image_files())

            top_matches = [
                (str(images_path / "img1.jpg"), 0.9),
                (str(images_path / "img2.jpg"), 0.8),
            ]

            # Select first time
            selected1 = matcher.select_match(top_matches)
            assert selected1 in [str(images_path / "img1.jpg"), str(images_path / "img2.jpg")]

            # Select second time - should get different image
            selected2 = matcher.select_match(top_matches)
            assert selected2 != selected1

    def test_select_match_always_returns_valid_path(self):
        """Test that select_match NEVER returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create some test images
            for i in range(5):
                create_test_image(images_path / f"img{i}.jpg")

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                similarity_threshold=0.99,  # Very high
                seed=42
            )

            # Build minimal index
            matcher._build_faiss_index(matcher.get_image_files())

            # Empty top matches
            top_matches: list[tuple[str, float]] = []

            selected = matcher.select_match(top_matches)

            # Should still return a valid path (from dataset fallback)
            assert selected is not None
            assert isinstance(selected, str)
            assert Path(selected).exists()


class TestDemoMode:
    """Test demo mode functionality."""

    def test_demo_mode_limits_images(self):
        """Test that demo mode limits number of images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create many test images
            for i in range(50):
                create_test_image(images_path / f"img{i}.jpg")

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                demo_mode=True,
                demo_images=10
            )

            image_files = matcher.get_image_files()

            # Should be limited to demo_images
            assert len(image_files) == 10

    def test_demo_mode_respected_in_output(self):
        """Test that demo mode settings are respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                demo_mode=True,
                demo_seconds=5,
                demo_images=10
            )

            assert matcher.demo_mode is True
            assert matcher.demo_seconds == 5
            assert matcher.demo_images == 10


class TestCheckpointIntegrity:
    """Test checkpoint integrity and null prevention."""

    def test_checkpoint_no_null_selections(self):
        """Test that checkpoint verification detects null selections."""
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

            # Create checkpoint with null selection
            checkpoint = {
                "frame_000000": {"top_matches": [], "selected": None},
                "frame_000001": {"top_matches": [], "selected": "img1.jpg"},
            }

            matcher.save_checkpoint(checkpoint)
            loaded = matcher.load_checkpoint()

            # Count nulls
            null_count = sum(1 for v in loaded.values() if v.get("selected") is None)

            assert null_count == 1  # Should detect the one null


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
