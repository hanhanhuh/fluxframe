"""Image similarity matching using multiple metrics (color, edges, texture)."""


from typing import Any, Literal

import cv2
import numpy as np
import numpy.typing as npt

try:
    from skimage.feature import hog
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Constants for magic values
_COLOR_CHANNELS = 3
_GRAYSCALE_CHANNELS = 2
_ASPECT_RATIO_TOLERANCE = 0.01

FeatureMethod = Literal["canny", "hog", "spatial_pyramid"]


class ImageMatcher:
    """Image similarity matching using weighted combination of edge, texture, and color features.

    This class computes similarity between images using three metrics:
    - Edge similarity via Canny edge detection
    - Texture similarity via Sobel gradient magnitude
    - Color similarity via HSV histograms

    Weights are automatically normalized to sum to 1.0.
    """

    def __init__(
        self,
        edge_weight: float = 0.33,
        texture_weight: float = 0.33,
        color_weight: float = 0.34,
        feature_method: FeatureMethod = "canny"
    ):
        """Initialize matcher with metric weights.

        Args:
            edge_weight: Weight for edge/contour similarity (0-1). Default 0.33.
            texture_weight: Weight for texture similarity (0-1). Default 0.33.
            color_weight: Weight for color histogram similarity (0-1). Default 0.34.
            feature_method: Method for edge/structure features. Default "canny".
                - "canny": Fast, no spatial info (current method)
                - "spatial_pyramid": 4x4 grid of Canny histograms (preserves layout)
                - "hog": Histogram of Oriented Gradients (best motion preservation)

        Note:
            Weights are automatically normalized to sum to 1.0 regardless of input values.
        """
        self.edge_weight = edge_weight
        self.texture_weight = texture_weight
        self.color_weight = color_weight
        self.feature_method = feature_method

        # Normalize weights
        total = edge_weight + texture_weight + color_weight
        self.edge_weight /= total
        self.texture_weight /= total
        self.color_weight /= total

    def aspect_ratio_crop(self, img: npt.NDArray[Any], target_ratio: float) -> npt.NDArray[Any]:
        """Crop image to target aspect ratio using center crop (no padding bars).

        Args:
            img: Input image as numpy array (H, W, C) or (H, W).
            target_ratio: Target aspect ratio as width/height.

        Returns:
            Center-cropped image with target aspect ratio.
        """
        h, w = img.shape[:2]
        current_ratio = w / h

        if abs(current_ratio - target_ratio) < _ASPECT_RATIO_TOLERANCE:
            return img

        if current_ratio > target_ratio:
            # Image is too wide, crop width
            new_w = int(h * target_ratio)
            start_x = (w - new_w) // 2
            return img[:, start_x:start_x + new_w]
        # Image is too tall, crop height
        new_h = int(w / target_ratio)
        start_y = (h - new_h) // 2
        return img[start_y:start_y + new_h, :]

    def compute_edge_features(self, img: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Compute edge/structure features based on selected method.

        Args:
            img: Input image (BGR or grayscale).

        Returns:
            Feature vector (size depends on method).
        """
        if self.feature_method == "canny":
            return self._compute_canny_features(img)
        if self.feature_method == "spatial_pyramid":
            return self._compute_spatial_pyramid_features(img)
        return self._compute_hog_features(img)

    def _compute_canny_features(self, img: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Compute Canny edge histogram (original method).

        Args:
            img: Input image (BGR or grayscale).

        Returns:
            Normalized edge histogram (256 bins).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == _COLOR_CHANNELS else img

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Compute histogram of edges
        hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
        return cv2.normalize(hist, hist).flatten()

    def _compute_spatial_pyramid_features(self, img: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Compute spatial pyramid of Canny edge histograms.

        Divides image into 4x4 grid and computes edge histogram for each cell.
        Preserves spatial layout information.

        Args:
            img: Input image (BGR or grayscale).

        Returns:
            Concatenated edge histograms from all grid cells (16 cells * 32 bins = 512 values).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == _COLOR_CHANNELS else img

        h, w = gray.shape[:2]
        grid_size = 4  # 4x4 grid
        cell_h, cell_w = h // grid_size, w // grid_size

        features: list[npt.NDArray[Any]] = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Extract cell
                cell = gray[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]

                # Canny edge detection on cell
                edges = cv2.Canny(cell, 50, 150)

                # Histogram with fewer bins per cell
                hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                features.append(hist)

        concatenated: npt.NDArray[Any] = np.concatenate(features)
        return concatenated.astype(np.float32)

    def _compute_hog_features(self, img: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Compute Histogram of Oriented Gradients features.

        HOG captures both gradient direction and spatial layout,
        providing excellent motion preservation for temporal mosaics.

        Args:
            img: Input image (BGR or grayscale).

        Returns:
            HOG feature vector (~1296 values with current settings).
        """
        if not HAS_SKIMAGE:
            msg = (
                "scikit-image is required for HOG features. "
                "Install with: pip install scikit-image"
            )
            raise ImportError(msg)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == _COLOR_CHANNELS else img

        # Fast HOG settings optimized for speed while preserving motion
        features: npt.NDArray[Any] = hog(
            gray,
            orientations=9,           # 9 gradient orientation bins
            pixels_per_cell=(32, 32), # Larger cells = faster (vs 8x8 or 16x16)
            cells_per_block=(2, 2),   # Normalization blocks
            block_norm="L2-Hys",      # Histogram normalization method
            feature_vector=True       # Return as 1D array
        )

        return features.astype(np.float32)

    def compute_texture_features(self, img: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Compute texture features using Sobel gradient magnitude.

        Uses gradient magnitude as CPU-efficient texture descriptor.

        Args:
            img: Input image (BGR or grayscale).

        Returns:
            Normalized texture histogram (64 bins).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == _COLOR_CHANNELS else img

        # Compute gradients for texture
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Normalize and convert to uint8
        magnitude_uint8 = np.zeros_like(magnitude, dtype=np.uint8)
        cv2.normalize(magnitude, magnitude_uint8, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Compute histogram
        hist = cv2.calcHist([magnitude_uint8], [0], None, [64], [0, 256])
        return cv2.normalize(hist, hist).flatten()

    def compute_color_features(self, img: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Compute color histogram features in HSV space.

        Args:
            img: Input image (BGR).

        Returns:
            Normalized 3D color histogram (8x8x8 bins = 512 values).
        """
        local_img = img
        if len(local_img.shape) == _GRAYSCALE_CHANNELS:
            # Grayscale image, convert to BGR
            local_img = cv2.cvtColor(local_img, cv2.COLOR_GRAY2BGR)

        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(local_img, cv2.COLOR_BGR2HSV)

        # Compute 3D histogram (H, S, V)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                           [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def compare_histograms(
        self, hist1: npt.NDArray[Any], hist2: npt.NDArray[Any]
    ) -> float:
        """Compare two histograms using correlation method.

        Args:
            hist1: First normalized histogram.
            hist2: Second normalized histogram.

        Returns:
            Similarity score (0-1, higher is more similar).
        """
        # Using correlation method (higher = more similar)
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        # Normalize to 0-1 range (correlation can be negative)
        return (similarity + 1) / 2

    def compute_all_features(
        self, img: npt.NDArray[Any]
    ) -> dict[str, npt.NDArray[Any]]:
        """Compute all feature vectors for an image.

        Args:
            img: Input image (BGR or grayscale).

        Returns:
            Dictionary with 'edge', 'texture', 'color' histogram features.
        """
        return {
            "edge": self.compute_edge_features(img),
            "texture": self.compute_texture_features(img),
            "color": self.compute_color_features(img)
        }

    def compute_similarity(self, img1: npt.NDArray[Any], img2: npt.NDArray[Any]) -> float:
        """Compute weighted similarity between two images.

        Args:
            img1: First image (BGR or grayscale).
            img2: Second image (BGR or grayscale).

        Returns:
            Weighted similarity score (0-1, higher is more similar).
        """
        # Ensure images are same size
        local_img2 = img2
        if img1.shape != local_img2.shape:
            local_img2 = cv2.resize(local_img2, (img1.shape[1], img1.shape[0]))

        # Compute features
        edge1 = self.compute_edge_features(img1)
        edge2 = self.compute_edge_features(local_img2)
        edge_sim = self.compare_histograms(edge1, edge2)

        texture1 = self.compute_texture_features(img1)
        texture2 = self.compute_texture_features(local_img2)
        texture_sim = self.compare_histograms(texture1, texture2)

        color1 = self.compute_color_features(img1)
        color2 = self.compute_color_features(local_img2)
        color_sim = self.compare_histograms(color1, color2)

        # Weighted combination
        return (self.edge_weight * edge_sim +
                self.texture_weight * texture_sim +
                self.color_weight * color_sim)

    def compute_similarity_from_features(
        self,
        features1: dict[str, npt.NDArray[Any]],
        features2: dict[str, npt.NDArray[Any]]
    ) -> float:
        """Compute weighted similarity from pre-computed features.

        Args:
            features1: First image features dict with 'edge', 'texture', 'color' keys.
            features2: Second image features dict with 'edge', 'texture', 'color' keys.

        Returns:
            Weighted similarity score (0-1, higher is more similar).
        """
        edge_sim = self.compare_histograms(features1["edge"], features2["edge"])
        texture_sim = self.compare_histograms(features1["texture"], features2["texture"])
        color_sim = self.compare_histograms(features1["color"], features2["color"])

        # Weighted combination
        return (self.edge_weight * edge_sim +
                self.texture_weight * texture_sim +
                self.color_weight * color_sim)
