"""Checkpoint management for video frame matching."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint save/load/validation for frame matching."""

    def __init__(self, checkpoint_path: Path):
        """Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to checkpoint JSON file
        """
        self.checkpoint_path = checkpoint_path

    def load(self) -> dict[str, Any] | None:
        """Load checkpoint from disk.

        Returns:
            Checkpoint dictionary if exists, None otherwise
        """
        if not self.checkpoint_path.exists():
            return None

        try:
            with self.checkpoint_path.open() as f:
                checkpoint: dict[str, Any] = json.load(f)
                logger.info(f"Loaded checkpoint with {len(checkpoint.get('frames', []))} frames")
                return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def save(self, checkpoint: dict[str, Any]) -> None:
        """Save checkpoint to disk.

        Args:
            checkpoint: Checkpoint dictionary to save
        """
        try:
            with self.checkpoint_path.open("w") as f:
                json.dump(checkpoint, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def validate_integrity(self, checkpoint: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate checkpoint integrity.

        Args:
            checkpoint: Checkpoint dictionary to validate

        Returns:
            Tuple of (is_valid, list of issues found)
        """
        issues = []

        if "frames" not in checkpoint:
            issues.append("Missing 'frames' key")
            return False, issues

        frames = checkpoint["frames"]
        if not isinstance(frames, list):
            issues.append("'frames' is not a list")
            return False, issues

        # Check each frame
        for idx, frame_data in enumerate(frames):
            if not isinstance(frame_data, dict):
                issues.append(f"Frame {idx} is not a dict")
                continue

            if "frame_number" not in frame_data:
                issues.append(f"Frame {idx} missing 'frame_number'")

            if "selected_image" not in frame_data:
                issues.append(f"Frame {idx} missing 'selected_image'")
            elif frame_data["selected_image"] is None:
                issues.append(f"Frame {idx} has null selected_image")

            if "similarity_score" not in frame_data:
                issues.append(f"Frame {idx} missing 'similarity_score'")

        return len(issues) == 0, issues
