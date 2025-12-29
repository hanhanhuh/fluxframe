"""Image similarity matching using multiple metrics (color, edges, texture)."""


from typing import Any, Literal, TypedDict

import cv2
import numpy as np
import numpy.typing as npt

try:
    from skimage.feature import hog
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# Constants for magic values
_COLOR_CHANNELS = 3
_GRAYSCALE_CHANNELS = 2
_ASPECT_RATIO_TOLERANCE = 0.01

FeatureMethod = Literal["canny", "hog", "spatial_pyramid", "mobilenet"]


class ClassicalFeatures(TypedDict):
    """Feature dictionary for classical (non-neural) methods.

    Contains separate histograms for edge, texture, and color features.
    Used by: canny, hog, spatial_pyramid methods.
    """
    edge: npt.NDArray[np.float32]
    texture: npt.NDArray[np.float32]
    color: npt.NDArray[np.float32]


class NeuralFeatures(TypedDict):
    """Feature dictionary for neural network methods.

    Contains only edge features (neural networks encode all information in one vector).
    Used by: mobilenet method.
    """
    edge: npt.NDArray[np.float32]


FeatureDict = ClassicalFeatures | NeuralFeatures


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
                - "mobilenet": MobileNetV3-Small early layers (neural, ~3-5ms)

        Note:
            Weights are automatically normalized to sum to 1.0 regardless of input values.
            For "mobilenet" mode, weights are ignored (uses cosine similarity directly).
        """
        self.edge_weight = edge_weight
        self.texture_weight = texture_weight
        self.color_weight = color_weight
        self.feature_method = feature_method

        # Normalize weights (only used for non-neural methods)
        total = edge_weight + texture_weight + color_weight
        self.edge_weight /= total
        self.texture_weight /= total
        self.color_weight /= total

        # Initialize MobileNet model if needed
        self._mobilenet_model = None
        self._mobilenet_transform = None
        if feature_method == "mobilenet":
            self._init_mobilenet()

    def _init_mobilenet(self) -> None:
        """Initialize MobileNetV3-Small ONNX model (truncated for speed).

        Downloads pre-exported ONNX model on first use, caches for subsequent runs.
        """
        if not HAS_ONNX:
            msg = (
                "ONNX Runtime is required for MobileNet features. "
                "Install with: pip install onnxruntime"
            )
            raise ImportError(msg)

        # ONNX model will be downloaded/cached here
        # For now, we'll export from PyTorch on-the-fly (requires torch temporarily)
        # TODO: Host pre-exported ONNX model online for direct download
        import pathlib
        cache_dir = pathlib.Path.home() / ".cache" / "fluxframe"
        cache_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = cache_dir / "mobilenetv3_small_block4.onnx"

        if not onnx_path.exists():
            # Export model on first run (requires torch temporarily)
            try:
                import torch
                import torchvision.models as models
            except ImportError as e:
                msg = (
                    "First-time setup requires PyTorch to export ONNX model.\n"
                    "Install CPU-only version (faster, smaller):\n"
                    "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
                    "After first run, only onnxruntime is needed."
                )
                raise ImportError(msg) from e

            # Load and truncate model
            model = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            )
            model.eval()
            truncated = torch.nn.Sequential(*list(model.features[:4]))

            # Export to ONNX (use temp file to avoid partial writes)
            import tempfile
            temp_path = onnx_path.with_suffix(".onnx.tmp")
            try:
                dummy_input = torch.randn(1, 3, 224, 224)
                torch.onnx.export(
                    truncated,
                    dummy_input,
                    str(temp_path),
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                    opset_version=14,
                    verbose=False,
                )
                # Atomic rename only if export succeeded
                temp_path.rename(onnx_path)
            except Exception:
                # Clean up partial file on failure
                if temp_path.exists():
                    temp_path.unlink()
                raise

        # Load ONNX model with CPU execution provider
        self._mobilenet_model = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )

        # Store input/output names
        self._onnx_input_name = self._mobilenet_model.get_inputs()[0].name
        self._onnx_output_name = self._mobilenet_model.get_outputs()[0].name

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

    def _ensure_grayscale(self, img: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Convert image to grayscale if needed.

        Args:
            img: Input image (BGR or grayscale).

        Returns:
            Grayscale image.
        """
        if len(img.shape) == _COLOR_CHANNELS:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

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
        if self.feature_method == "mobilenet":
            return self._compute_mobilenet_features(img)
        return self._compute_hog_features(img)

    def _compute_canny_features(self, img: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Compute Canny edge histogram (original method).

        Args:
            img: Input image (BGR or grayscale).

        Returns:
            Normalized edge histogram (256 bins).
        """
        gray = self._ensure_grayscale(img)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Compute histogram using NumPy (faster for 1D histograms)
        hist, _ = np.histogram(edges, bins=256, range=(0, 256))
        hist = hist.astype(np.float32)

        # Normalize
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum

        return hist

    def _compute_spatial_pyramid_features(self, img: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Compute spatial pyramid of Canny edge histograms.

        Divides image into 4x4 grid and computes edge histogram for each cell.
        Preserves spatial layout information.

        Args:
            img: Input image (BGR or grayscale).

        Returns:
            Concatenated edge histograms from all grid cells (16 cells * 32 bins = 512 values).
        """
        gray = self._ensure_grayscale(img)

        h, w = gray.shape[:2]
        grid_size = 4  # 4x4 grid
        cell_h, cell_w = h // grid_size, w // grid_size
        bins_per_cell = 32

        # Pre-allocate feature array (faster than list append + concatenate)
        features = np.zeros(grid_size * grid_size * bins_per_cell, dtype=np.float32)

        for i in range(grid_size):
            for j in range(grid_size):
                # Extract cell (using array view, no copy)
                cell = gray[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]

                # Canny edge detection on cell
                edges = cv2.Canny(cell, 50, 150)

                # Compute histogram using NumPy
                hist, _ = np.histogram(edges, bins=bins_per_cell, range=(0, 256))

                # Normalize and store
                hist_sum = hist.sum()
                if hist_sum > 0:
                    hist = hist / hist_sum

                # Store in pre-allocated array
                cell_idx = i * grid_size + j
                features[cell_idx * bins_per_cell:(cell_idx + 1) * bins_per_cell] = hist

        return features

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

        gray = self._ensure_grayscale(img)

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

    def _compute_mobilenet_features(self, img: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Compute MobileNetV3-Small features with spatial pyramid pooling.

        Preserves spatial layout (top/bottom/left/right) for perspective/contour matching.
        Uses 2×2 spatial grid to maintain "where" information (e.g., horizon at bottom vs top).

        Args:
            img: Input image (BGR).

        Returns:
            Spatial pyramid pooled feature vector (2×2 grid × 48 channels = 192D).
        """
        if self._mobilenet_model is None:
            msg = "MobileNet model not initialized"
            raise RuntimeError(msg)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 224×224 (ImageNet input size)
        resized = cv2.resize(rgb, (224, 224))

        # Normalize with ImageNet mean/std
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std

        # Convert to NCHW format (batch, channels, height, width)
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension

        # Run ONNX inference
        features = self._mobilenet_model.run(
            [self._onnx_output_name],
            {self._onnx_input_name: input_tensor}
        )[0]  # Shape: [1, 48, 14, 14]

        # Apply 2×2 spatial pyramid pooling
        # Preserves layout: top-left, top-right, bottom-left, bottom-right
        _, channels, h, w = features.shape
        grid_h, grid_w = h // 2, w // 2

        pooled_features = []
        for i in range(2):
            for j in range(2):
                # Extract spatial cell
                cell = features[0, :, i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                # Average pool within cell
                cell_pooled = cell.mean(axis=(1, 2))  # [48]
                pooled_features.append(cell_pooled)

        # Concatenate all cells: 4 cells × 48 channels = 192D
        result = np.concatenate(pooled_features, axis=0)

        return result.astype(np.float32)

    def compute_texture_features(self, img: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Compute texture features using Sobel gradient magnitude.

        Uses gradient magnitude as CPU-efficient texture descriptor.

        Args:
            img: Input image (BGR or grayscale).

        Returns:
            Normalized texture histogram (64 bins).
        """
        gray = self._ensure_grayscale(img)

        # Compute gradients using CV_32F (faster than CV_64F)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Use cv2.magnitude (optimized C++ implementation, faster than np.sqrt)
        magnitude = cv2.magnitude(sobelx, sobely)

        # Compute histogram directly from float magnitude (no intermediate uint8 conversion)
        # Normalize magnitude to 0-255 range for histogram
        max_mag = magnitude.max()
        if max_mag > 0:
            magnitude_norm = (magnitude / max_mag) * 255
        else:
            magnitude_norm = magnitude

        # Use NumPy histogram (faster for 1D)
        hist, _ = np.histogram(magnitude_norm, bins=64, range=(0, 256))
        hist = hist.astype(np.float32)

        # Normalize
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum

        return hist

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

    def _cosine_similarity(self, feat1: npt.NDArray[Any], feat2: npt.NDArray[Any]) -> float:
        """Compute cosine similarity between two feature vectors.

        Args:
            feat1: First feature vector.
            feat2: Second feature vector.

        Returns:
            Cosine similarity (0-1, higher is more similar).
        """
        dot_product = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity is in [-1, 1], normalize to [0, 1]
        return (dot_product / (norm1 * norm2) + 1) / 2

    def compute_all_features(
        self, img: npt.NDArray[Any]
    ) -> FeatureDict:
        """Compute all feature vectors for an image.

        Optimized to cache grayscale conversion when computing edge and texture features.

        Args:
            img: Input image (BGR or grayscale).

        Returns:
            Dictionary with 'edge', 'texture', 'color' histogram features.
        """
        # MobileNet mode: only compute neural features
        if self.feature_method == "mobilenet":
            return {"edge": self._compute_mobilenet_features(img)}

        # Classical methods: compute all features
        # Cache grayscale conversion for edge and texture features
        # (color features use HSV, so don't benefit from this cache)
        gray = self._ensure_grayscale(img)

        # Compute edge features (already grayscale)
        if self.feature_method == "canny":
            edge_features = self._compute_canny_features(gray)
        elif self.feature_method == "spatial_pyramid":
            edge_features = self._compute_spatial_pyramid_features(gray)
        else:
            edge_features = self._compute_hog_features(gray)

        # Compute texture features (already grayscale)
        texture_features = self.compute_texture_features(gray)

        # Compute color features (needs original BGR image)
        color_features = self.compute_color_features(img)

        return {
            "edge": edge_features,
            "texture": texture_features,
            "color": color_features
        }

    def features_to_vector(self, features: FeatureDict) -> npt.NDArray[np.float32]:
        """Convert feature dict to single vector for FAISS indexing.

        Args:
            features: Feature dictionary from compute_all_features.

        Returns:
            Single concatenated and weighted feature vector.
        """
        # MobileNet: return features directly (no weighting/concatenation)
        if self.feature_method == "mobilenet":
            return features["edge"].astype(np.float32)

        # Classical methods: weighted concatenation
        edge_vec = features["edge"] * self.edge_weight
        texture_vec = features["texture"] * self.texture_weight
        color_vec = features["color"] * self.color_weight

        return np.concatenate([edge_vec, texture_vec, color_vec]).astype(np.float32)

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

        # MobileNet mode: use cosine similarity only
        if self.feature_method == "mobilenet":
            feat1 = self._compute_mobilenet_features(img1)
            feat2 = self._compute_mobilenet_features(local_img2)
            return self._cosine_similarity(feat1, feat2)

        # Classical methods: weighted combination
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
        features1: FeatureDict,
        features2: FeatureDict
    ) -> float:
        """Compute weighted similarity from pre-computed features.

        Args:
            features1: First image features dict from compute_all_features.
            features2: Second image features dict from compute_all_features.

        Returns:
            Weighted similarity score (0-1, higher is more similar).
        """
        # MobileNet mode: use cosine similarity
        if self.feature_method == "mobilenet":
            return self._cosine_similarity(features1["edge"], features2["edge"])

        # Classical methods: weighted combination of histogram comparisons
        edge_sim = self.compare_histograms(features1["edge"], features2["edge"])
        texture_sim = self.compare_histograms(features1["texture"], features2["texture"])
        color_sim = self.compare_histograms(features1["color"], features2["color"])

        return (self.edge_weight * edge_sim +
                self.texture_weight * texture_sim +
                self.color_weight * color_sim)
