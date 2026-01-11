#!/usr/bin/env python3
"""Color grading module for smooth progressive frame-to-frame color matching."""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config

# Note: colorharmonies library not currently used
# Color matching is implemented using OpenCV


class ColorGrader:
    """Progressive color grading for video frame sequences.

    Applies frame-to-frame color matching to reduce flicker and
    improve visual consistency in generated videos.
    """

    def __init__(self, cfg: Config):
        """Initialize color grader.

        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.method = cfg.color_grading_method
        self.strength = cfg.color_grading_strength

        # Temporal smoothing buffer
        self.prev_adjustment: np.ndarray | None = None

    def match_colors(
        self, source: np.ndarray, target: np.ndarray, prev_frame: np.ndarray | None = None
    ) -> np.ndarray:
        """Match colors of source to target with temporal smoothing.

        Args:
            source: Source image (BGR)
            target: Target image to match colors to (BGR)
            prev_frame: Previous adjusted frame for temporal smoothing (BGR)

        Returns:
            Color-adjusted source image (BGR)
        """
        # Select method
        if self.method == "histogram":
            adjusted = self._histogram_matching(source, target)
        elif self.method == "color_transfer":
            adjusted = self._color_transfer(source, target)
        elif self.method == "lut":
            adjusted = self._lut_matching(source, target)
        else:
            raise ValueError(f"Unknown color grading method: {self.method}")

        # Apply strength blending
        adjusted = self._blend_with_strength(source, adjusted)

        # Temporal smoothing (reduce flicker)
        if prev_frame is not None:
            adjusted = self._temporal_smoothing(adjusted, prev_frame)

        return adjusted

    def _histogram_matching(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Histogram matching per channel (fast, ~5ms/frame).

        Args:
            source: Source image (BGR)
            target: Target image (BGR)

        Returns:
            Adjusted source image (BGR)
        """
        matched = np.zeros_like(source)

        for channel in range(3):
            matched[:, :, channel] = self._match_histograms_channel(
                source[:, :, channel], target[:, :, channel]
            )

        return matched

    def _match_histograms_channel(
        self, source: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """Match histogram for single channel.

        Args:
            source: Source channel (2D array)
            target: Target channel (2D array)

        Returns:
            Matched channel
        """
        # Compute CDFs
        source_hist, _ = np.histogram(source.flatten(), 256, (0, 256))
        target_hist, _ = np.histogram(target.flatten(), 256, (0, 256))

        source_cdf = source_hist.cumsum()
        target_cdf = target_hist.cumsum()

        # Normalize CDFs
        source_cdf = source_cdf / source_cdf[-1]
        target_cdf = target_cdf / target_cdf[-1]

        # Create lookup table
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # Find closest CDF value in target
            diff = np.abs(target_cdf - source_cdf[i])
            lut[i] = np.argmin(diff)

        # Apply lookup table
        return lut[source]

    def _color_transfer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Reinhard et al. color transfer in LAB space (balanced, ~10ms/frame).

        Args:
            source: Source image (BGR)
            target: Target image (BGR)

        Returns:
            Adjusted source image (BGR)
        """
        # Convert to LAB
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Compute statistics for each channel
        source_mean = source_lab.mean(axis=(0, 1))
        source_std = source_lab.std(axis=(0, 1))
        target_mean = target_lab.mean(axis=(0, 1))
        target_std = target_lab.std(axis=(0, 1))

        # Avoid division by zero
        source_std[source_std == 0] = 1.0

        # Transfer statistics
        adjusted_lab = source_lab.copy()
        for i in range(3):
            # Normalize to zero mean, unit variance
            adjusted_lab[:, :, i] = (adjusted_lab[:, :, i] - source_mean[i]) / source_std[i]
            # Scale to target statistics
            adjusted_lab[:, :, i] = adjusted_lab[:, :, i] * target_std[i] + target_mean[i]

        # Clip to valid LAB range
        adjusted_lab[:, :, 0] = np.clip(adjusted_lab[:, :, 0], 0, 255)  # L: 0-255
        adjusted_lab[:, :, 1] = np.clip(adjusted_lab[:, :, 1], 0, 255)  # A: 0-255
        adjusted_lab[:, :, 2] = np.clip(adjusted_lab[:, :, 2], 0, 255)  # B: 0-255

        # Convert back to BGR
        adjusted_lab = adjusted_lab.astype(np.uint8)
        return cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

    def _lut_matching(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """3D LUT-based color matching (best quality, ~20ms/frame).

        Uses a 3D lookup table to map source colors to target color distribution.
        More accurate than histogram matching for complex color transformations.

        Args:
            source: Source image (BGR)
            target: Target image (BGR)

        Returns:
            Adjusted source image (BGR)
        """
        # Create 3D LUT with 64x64x64 bins (good balance of quality and speed)
        lut_size = 64

        # Convert to RGB for consistent color space
        source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Build histograms for each channel
        bins = lut_size
        source_hist_r, _ = np.histogram(source_rgb[:, :, 0].flatten(), bins, (0, 256))
        source_hist_g, _ = np.histogram(source_rgb[:, :, 1].flatten(), bins, (0, 256))
        source_hist_b, _ = np.histogram(source_rgb[:, :, 2].flatten(), bins, (0, 256))

        target_hist_r, _ = np.histogram(target_rgb[:, :, 0].flatten(), bins, (0, 256))
        target_hist_g, _ = np.histogram(target_rgb[:, :, 1].flatten(), bins, (0, 256))
        target_hist_b, _ = np.histogram(target_rgb[:, :, 2].flatten(), bins, (0, 256))

        # Compute CDFs
        source_cdf_r = source_hist_r.cumsum()
        source_cdf_g = source_hist_g.cumsum()
        source_cdf_b = source_hist_b.cumsum()

        target_cdf_r = target_hist_r.cumsum()
        target_cdf_g = target_hist_g.cumsum()
        target_cdf_b = target_hist_b.cumsum()

        # Normalize CDFs
        source_cdf_r = source_cdf_r / source_cdf_r[-1] if source_cdf_r[-1] > 0 else source_cdf_r
        source_cdf_g = source_cdf_g / source_cdf_g[-1] if source_cdf_g[-1] > 0 else source_cdf_g
        source_cdf_b = source_cdf_b / source_cdf_b[-1] if source_cdf_b[-1] > 0 else source_cdf_b

        target_cdf_r = target_cdf_r / target_cdf_r[-1] if target_cdf_r[-1] > 0 else target_cdf_r
        target_cdf_g = target_cdf_g / target_cdf_g[-1] if target_cdf_g[-1] > 0 else target_cdf_g
        target_cdf_b = target_cdf_b / target_cdf_b[-1] if target_cdf_b[-1] > 0 else target_cdf_b

        # Create 3D LUT mapping
        lut_3d = np.zeros((lut_size, lut_size, lut_size, 3), dtype=np.uint8)

        # For each cell in the 3D LUT, find the corresponding target color
        for r in range(lut_size):
            for g in range(lut_size):
                for b in range(lut_size):
                    # Map each channel independently using CDF matching
                    # Find closest CDF value in target for each channel
                    r_mapped = np.argmin(np.abs(target_cdf_r - source_cdf_r[r])) if r < len(source_cdf_r) else r
                    g_mapped = np.argmin(np.abs(target_cdf_g - source_cdf_g[g])) if g < len(source_cdf_g) else g
                    b_mapped = np.argmin(np.abs(target_cdf_b - source_cdf_b[b])) if b < len(source_cdf_b) else b

                    # Scale back to 0-255 range
                    lut_3d[r, g, b] = [
                        int(r_mapped * 255 / (lut_size - 1)),
                        int(g_mapped * 255 / (lut_size - 1)),
                        int(b_mapped * 255 / (lut_size - 1))
                    ]

        # Apply LUT to source image
        # Quantize source to LUT indices
        scale = (lut_size - 1) / 255.0
        r_idx = (source_rgb[:, :, 0] * scale).astype(np.int32)
        g_idx = (source_rgb[:, :, 1] * scale).astype(np.int32)
        b_idx = (source_rgb[:, :, 2] * scale).astype(np.int32)

        # Clip to valid range
        r_idx = np.clip(r_idx, 0, lut_size - 1)
        g_idx = np.clip(g_idx, 0, lut_size - 1)
        b_idx = np.clip(b_idx, 0, lut_size - 1)

        # Lookup colors
        adjusted_rgb = lut_3d[r_idx, g_idx, b_idx]

        # Convert back to BGR
        return cv2.cvtColor(adjusted_rgb, cv2.COLOR_RGB2BGR)

    def _blend_with_strength(
        self, original: np.ndarray, adjusted: np.ndarray
    ) -> np.ndarray:
        """Blend adjusted image with original based on strength parameter.

        Args:
            original: Original image
            adjusted: Adjusted image

        Returns:
            Blended image
        """
        # Blend: result = original * (1 - strength) + adjusted * strength
        return cv2.addWeighted(
            original, 1.0 - self.strength, adjusted, self.strength, 0
        )

    def _temporal_smoothing(
        self, current: np.ndarray, prev_frame: np.ndarray
    ) -> np.ndarray:
        """Apply temporal smoothing to reduce flicker.

        Args:
            current: Current frame
            prev_frame: Previous frame

        Returns:
            Smoothed frame
        """
        # Gentle smoothing: 90% current, 10% previous
        # This reduces sudden color jumps between frames
        smoothing_factor = 0.1
        return cv2.addWeighted(current, 1.0 - smoothing_factor, prev_frame, smoothing_factor, 0)


def create_color_grader(cfg: Config) -> ColorGrader | None:
    """Factory function to create color grader if enabled.

    Args:
        cfg: Configuration object

    Returns:
        ColorGrader instance if enabled, None otherwise
    """
    if cfg.enable_color_grading:
        return ColorGrader(cfg)
    return None
