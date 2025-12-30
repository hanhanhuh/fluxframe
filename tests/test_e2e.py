"""
Integration tests with actual video and image generation.
Tests the full pipeline end-to-end with realistic data.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from fluxframe import VideoImageMatcher


def create_test_video(path: Path, num_frames: int = 10, fps: int = 30,
                     width: int = 320, height: int = 240) -> None:
    """Create a test video with different colored frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create frames with different colors
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Create a gradient or pattern for each frame
        if i % 3 == 0:
            # Red frame
            frame[:, :, 2] = 200
        elif i % 3 == 1:
            # Green frame
            frame[:, :, 1] = 200
        else:
            # Blue frame
            frame[:, :, 0] = 200

        # Add some noise/texture
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        video_writer.write(frame)

    video_writer.release()


def create_test_images(folder: Path, num_images: int = 20) -> None:
    """Create test images with various colors and patterns."""
    folder.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        img = np.zeros((240, 320, 3), dtype=np.uint8)

        if i < 7:
            # Red-ish images
            img[:, :, 2] = np.random.randint(150, 255)
            img[:, :, 1] = np.random.randint(0, 100)
            img[:, :, 0] = np.random.randint(0, 100)
        elif i < 14:
            # Green-ish images
            img[:, :, 1] = np.random.randint(150, 255)
            img[:, :, 2] = np.random.randint(0, 100)
            img[:, :, 0] = np.random.randint(0, 100)
        else:
            # Blue-ish images
            img[:, :, 0] = np.random.randint(150, 255)
            img[:, :, 1] = np.random.randint(0, 100)
            img[:, :, 2] = np.random.randint(0, 100)

        # Add texture
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)

        cv2.imwrite(str(folder / f"img_{i:03d}.jpg"), img)


class TestFullPipeline:
    """Integration tests for the full video-image matching pipeline."""

    def test_end_to_end_processing(self):
        """Test complete pipeline from video to matched output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Setup paths
            video_path = tmpdir_path / "test_video.mp4"
            images_folder = tmpdir_path / "images"
            output_folder = tmpdir_path / "output"

            # Create test data
            create_test_video(video_path, num_frames=5)
            create_test_images(images_folder, num_images=15)

            # Run matcher
            matcher = VideoImageMatcher(
                str(video_path),
                str(images_folder),
                str(output_folder),
                top_n=5,
                seed=42
            )

            checkpoint = matcher.process()

            # Verify checkpoint structure
            assert len(checkpoint) == 5  # 5 frames processed
            for i in range(5):
                frame_key = f"frame_{i:06d}"
                assert frame_key in checkpoint
                assert "top_matches" in checkpoint[frame_key]
                assert "selected" in checkpoint[frame_key]
                assert len(checkpoint[frame_key]["top_matches"]) == 5

            # Verify checkpoint file exists
            assert (output_folder / "checkpoint.json").exists()
            assert (output_folder / "results.json").exists()
            # FAISS cache files
            assert (output_folder / "cache_metadata.json").exists()
            assert (output_folder / "faiss_index.bin").exists()
            assert (output_folder / "vectors.npy").exists()

    def test_color_similarity_matching(self):
        """Test that similar colors produce higher similarity scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            video_path = tmpdir_path / "test_video.mp4"
            images_folder = tmpdir_path / "images"
            output_folder = tmpdir_path / "output"

            # Create video with red frames
            create_test_video(video_path, num_frames=3)

            # Create images - some red, some not
            create_test_images(images_folder, num_images=20)

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_folder),
                str(output_folder),
                top_n=10,
                seed=42
            )

            checkpoint = matcher.process()

            # Check first frame (red)
            frame_0_matches = checkpoint["frame_000000"]["top_matches"]

            # Red images are img_000 to img_006
            # Check that most top matches are red images
            red_count = sum(1 for path, _ in frame_0_matches[:5]
                          if any(f"img_{i:03d}" in path for i in range(7)))

            # At least 3 out of top 5 should be red (similar color)
            assert red_count >= 3, f"Expected mostly red images in top matches, got {red_count}/5"

    def test_output_generation(self):
        """Test that output video and images are generated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            video_path = tmpdir_path / "test_video.mp4"
            images_folder = tmpdir_path / "images"
            output_folder = tmpdir_path / "output"

            create_test_video(video_path, num_frames=3)
            create_test_images(images_folder, num_images=10)

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_folder),
                str(output_folder),
                seed=42
            )

            checkpoint = matcher.process()
            matcher.generate_output(checkpoint)

            # Check output video exists
            output_video = output_folder / "test_video_matched.mp4"
            assert output_video.exists()

            # Verify video can be opened and has correct number of frames
            cap = cv2.VideoCapture(str(output_video))
            assert cap.isOpened()

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                assert frame.shape == (240, 320, 3)  # Check dimensions

            cap.release()
            assert frame_count == 3

            # Check output images folder
            output_images_dir = output_folder / "matched_frames"
            assert output_images_dir.exists()

            # Check individual frame images
            for i in range(3):
                frame_img = output_images_dir / f"frame_{i:06d}.jpg"
                assert frame_img.exists()

                # Verify image can be loaded
                img = cv2.imread(str(frame_img))
                assert img is not None
                assert img.shape == (240, 320, 3)

    def test_checkpoint_resume(self):
        """Test that processing can resume from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            video_path = tmpdir_path / "test_video.mp4"
            images_folder = tmpdir_path / "images"
            output_folder = tmpdir_path / "output"

            create_test_video(video_path, num_frames=5)
            create_test_images(images_folder, num_images=10)

            # First run - process only first 2 frames manually
            matcher1 = VideoImageMatcher(
                str(video_path),
                str(images_folder),
                str(output_folder),
                seed=42
            )

            # Manually create partial checkpoint
            partial_checkpoint = {}

            # Get image files and build FAISS index
            image_files = matcher1.get_image_files()
            matcher1._build_faiss_index(image_files)

            # Process just first 2 frames
            video_info = matcher1.get_video_info()
            cap = cv2.VideoCapture(str(video_path))

            for i in range(2):
                ret, frame = cap.read()
                if ret:
                    frame_key = f"frame_{i:06d}"
                    aspect_ratio = video_info.width / video_info.height
                    top_matches = matcher1.find_top_matches(frame, i, aspect_ratio)
                    selected = matcher1.select_match(top_matches)
                    partial_checkpoint[frame_key] = {
                        "top_matches": top_matches,
                        "selected": selected
                    }

            cap.release()
            matcher1.save_checkpoint(partial_checkpoint)

            # Second run - should resume and complete
            matcher2 = VideoImageMatcher(
                str(video_path),
                str(images_folder),
                str(output_folder),
                seed=42
            )

            checkpoint = matcher2.process()

            # Should have all 5 frames
            assert len(checkpoint) == 5

            # First 2 frames should match from resume
            for i in range(2):
                frame_key = f"frame_{i:06d}"
                expected = partial_checkpoint[frame_key]["selected"]
                assert checkpoint[frame_key]["selected"] == expected

    def test_no_repeat_mode(self):
        """Test that no-repeat mode tries to avoid reusing images but falls back when necessary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            video_path = tmpdir_path / "test_video.mp4"
            images_folder = tmpdir_path / "images"
            output_folder = tmpdir_path / "output"

            # Create fewer frames than images to test no-repeat works
            create_test_video(video_path, num_frames=4)
            create_test_images(images_folder, num_images=10)

            matcher = VideoImageMatcher(
                str(video_path),
                str(images_folder),
                str(output_folder),
                no_repeat=True,
                top_n=3,
                seed=42
            )

            checkpoint = matcher.process()

            # Collect all selected images
            selected_images = [data["selected"] for data in checkpoint.values()
                             if data["selected"] is not None]

            # When frames < images, no image should be reused
            assert len(selected_images) == len(set(selected_images)), \
                "No-repeat mode failed: images were reused when there were enough available"

            # Should have selected exactly 4 unique images
            assert len(selected_images) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
