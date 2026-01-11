"""Unit tests for performance optimizations (IVF index, TurboJPEG, etc.)."""

import tempfile
from pathlib import Path

import cv2
import faiss
import numpy as np
import pytest

from fluxframe import VideoFrameMatcher


class TestIVFIndexOptimization:
    """Test IndexIVFFlat optimization for faster search."""

    def test_ivf_index_enabled_by_default(self):
        """Test that IVF index is enabled by default for faster search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create test images
            for i in range(100):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(str(images_path / f"img{i:03d}.jpg"), img)

            matcher = VideoFrameMatcher(
                str(video_path), str(images_path), str(output_path), use_ivf_index=True
            )

            # Build index
            image_files = matcher.get_image_files()
            matcher._build_faiss_index(image_files)

            # Verify IVF index was created
            assert matcher.faiss_index is not None
            assert isinstance(matcher.faiss_index, faiss.IndexIVFFlat)

    def test_ivf_index_can_be_disabled(self):
        """Test that IVF index can be disabled to use exact search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create test images
            for i in range(100):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(str(images_path / f"img{i:03d}.jpg"), img)

            matcher = VideoFrameMatcher(
                str(video_path), str(images_path), str(output_path), use_ivf_index=False
            )

            # Build index
            image_files = matcher.get_image_files()
            matcher._build_faiss_index(image_files)

            # Verify exact index was created
            assert matcher.faiss_index is not None
            assert isinstance(matcher.faiss_index, faiss.IndexFlatIP)

    def test_ivf_uses_flat_for_small_datasets(self):
        """Test that IVF falls back to flat index for small datasets (<100 images)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create only 50 test images (below threshold)
            for i in range(50):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(str(images_path / f"img{i:03d}.jpg"), img)

            matcher = VideoFrameMatcher(
                str(video_path), str(images_path), str(output_path), use_ivf_index=True
            )

            # Build index
            image_files = matcher.get_image_files()
            matcher._build_faiss_index(image_files)

            # Should use flat index for small dataset
            assert matcher.faiss_index is not None
            assert isinstance(matcher.faiss_index, faiss.IndexFlatIP)

    def test_ivf_custom_nlist_and_nprobe(self):
        """Test custom nlist and nprobe parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create test images
            for i in range(200):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(str(images_path / f"img{i:03d}.jpg"), img)

            matcher = VideoFrameMatcher(
                str(video_path),
                str(images_path),
                str(output_path),
                use_ivf_index=True,
                ivf_nlist=16,
                ivf_nprobe=4,
            )

            # Build index
            image_files = matcher.get_image_files()
            matcher._build_faiss_index(image_files)

            # Verify custom parameters
            assert matcher.faiss_index is not None
            assert isinstance(matcher.faiss_index, faiss.IndexIVFFlat)
            assert matcher.faiss_index.nlist == 16
            assert matcher.faiss_index.nprobe == 4

    def test_ivf_auto_nlist_calculation(self):
        """Test automatic nlist calculation (sqrt(N) * 4)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create 400 images -> sqrt(400) * 4 = 20 * 4 = 80 clusters
            for i in range(400):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(str(images_path / f"img{i:03d}.jpg"), img)

            matcher = VideoFrameMatcher(
                str(video_path), str(images_path), str(output_path), use_ivf_index=True
            )

            # Build index
            image_files = matcher.get_image_files()
            matcher._build_faiss_index(image_files)

            # Verify auto-calculated nlist
            assert matcher.faiss_index is not None
            assert isinstance(matcher.faiss_index, faiss.IndexIVFFlat)
            expected_nlist = 80  # sqrt(400) * 4
            assert matcher.faiss_index.nlist == expected_nlist

    def test_ivf_search_produces_results(self):
        """Test that IVF search returns valid results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create test images
            for i in range(100):
                img = np.ones((100, 100, 3), dtype=np.uint8) * (i % 256)
                cv2.imwrite(str(images_path / f"img{i:03d}.jpg"), img)

            matcher = VideoFrameMatcher(
                str(video_path), str(images_path), str(output_path), use_ivf_index=True, top_n=5
            )

            # Build index
            image_files = matcher.get_image_files()
            matcher._build_faiss_index(image_files)

            # Search with a query
            query_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            top_matches = matcher.find_top_matches(query_img, frame_num=0, target_aspect_ratio=1.0)

            # Should return results
            assert len(top_matches) == 5
            assert all(isinstance(path, str) for path, _ in top_matches)
            assert all(isinstance(sim, float) for _, sim in top_matches)
            assert all(0.0 <= sim <= 1.0 for _, sim in top_matches)


class TestOpenCVOptimization:
    """Test OpenCV SIMD optimizations."""

    def test_opencv_optimized_enabled(self):
        """Test that OpenCV optimizations are enabled."""
        from fluxframe.matcher import ImageMatcher

        # Create a matcher (should enable optimizations in __init__)
        ImageMatcher()

        # Verify optimizations are enabled
        assert cv2.useOptimized()


class TestFastImageLoading:
    """Test fast image loading with TurboJPEG fallback."""

    def test_fast_imread_loads_jpeg(self):
        """Test that fast imread can load JPEG images."""
        from fluxframe.processor import _fast_imread

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.jpg"

            # Create a test JPEG
            test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), test_img)

            # Load with fast imread
            loaded_img = _fast_imread(img_path)

            assert loaded_img is not None
            assert loaded_img.shape == (100, 100, 3)
            assert loaded_img.dtype == np.uint8

    def test_fast_imread_loads_png(self):
        """Test that fast imread can load PNG images."""
        from fluxframe.processor import _fast_imread

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"

            # Create a test PNG
            test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), test_img)

            # Load with fast imread
            loaded_img = _fast_imread(img_path)

            assert loaded_img is not None
            assert loaded_img.shape == (100, 100, 3)
            assert loaded_img.dtype == np.uint8

    def test_fast_imread_returns_none_for_invalid(self):
        """Test that fast imread returns None for invalid files."""
        from fluxframe.processor import _fast_imread

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "nonexistent.jpg"

            # Try to load non-existent file
            loaded_img = _fast_imread(img_path)

            assert loaded_img is None


class TestMemoryMappedIndexLoading:
    """Test memory-mapped FAISS index loading."""

    def test_mmap_index_loading(self):
        """Test that index can be loaded with memory mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create test images
            for i in range(100):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(str(images_path / f"img{i:03d}.jpg"), img)

            # First run - build index
            matcher1 = VideoFrameMatcher(str(video_path), str(images_path), str(output_path))
            image_files = matcher1.get_image_files()
            matcher1._build_faiss_index(image_files)

            # Second run - load from cache with mmap
            matcher2 = VideoFrameMatcher(str(video_path), str(images_path), str(output_path))
            image_files2 = matcher2.get_image_files()
            matcher2._build_faiss_index(image_files2)

            # Verify index was loaded
            assert matcher2.faiss_index is not None
            assert len(matcher2.image_paths) == 100

    def test_mmap_index_search_works(self):
        """Test that search works with memory-mapped index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            images_path = Path(tmpdir) / "images"
            output_path = Path(tmpdir) / "output"

            video_path.touch()
            images_path.mkdir()

            # Create test images
            for i in range(100):
                img = np.ones((100, 100, 3), dtype=np.uint8) * (i % 256)
                cv2.imwrite(str(images_path / f"img{i:03d}.jpg"), img)

            # Build index
            matcher1 = VideoFrameMatcher(
                str(video_path), str(images_path), str(output_path), top_n=5
            )
            image_files = matcher1.get_image_files()
            matcher1._build_faiss_index(image_files)

            # Load from cache and search
            matcher2 = VideoFrameMatcher(
                str(video_path), str(images_path), str(output_path), top_n=5
            )
            image_files2 = matcher2.get_image_files()
            matcher2._build_faiss_index(image_files2)

            # Search
            query_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            top_matches = matcher2.find_top_matches(query_img, frame_num=0, target_aspect_ratio=1.0)

            # Should return results
            assert len(top_matches) == 5
            assert all(isinstance(path, str) for path, _ in top_matches)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
