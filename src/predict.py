"""Inference module for the NEU Surface Defect CNN classifier.

Loads the trained NEU_CNN_96 model and provides a clean prediction interface
that accepts PIL Images or numpy arrays and returns structured results.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from PIL import Image

from src.utils import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "defect_cnn_model.keras"
_DEFAULT_CLASS_NAMES_PATH = _PROJECT_ROOT / "models" / "class_names.json"

# Lazy-import tensorflow so the module can be imported without TF installed
# (tests that mock the model still work).
_tf = None


def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf  # noqa: PLC0415
        _tf = tf
    return _tf


class DefectPredictor:
    """Loads and runs the NEU Surface Defect CNN model for single-image inference.

    The model expects 96×96 single-channel (grayscale) images normalised to
    the [0, 1] range. Any PIL Image or numpy array will be automatically
    resized and converted before inference.

    Args:
        model_path: Path to the saved ``defect_cnn_model.keras`` file.
            Defaults to ``models/defect_cnn_model.keras`` relative to the
            project root.
        class_names_path: Path to ``class_names.json``. Defaults to
            ``models/class_names.json`` relative to the project root.

    Raises:
        FileNotFoundError: If either the model file or class-names file is
            missing from disk.
        RuntimeError: If TensorFlow fails to load the model.

    Example::

        predictor = DefectPredictor()
        result = predictor.predict(Image.open("surface.jpg"))
        print(result["predicted_class"], result["confidence"])
    """

    def __init__(
        self,
        model_path: Union[str, Path] = _DEFAULT_MODEL_PATH,
        class_names_path: Union[str, Path] = _DEFAULT_CLASS_NAMES_PATH,
    ) -> None:
        self._model_path = Path(model_path)
        self._class_names_path = Path(class_names_path)
        self._model = None
        self._class_names: List[str] = []
        self._img_size: int = 96
        self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the Keras model and class-name metadata from disk.

        Raises:
            FileNotFoundError: If model or class-names file is absent.
            RuntimeError: If TensorFlow raises during model load.
        """
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self._model_path}\n"
                "Run the export cell in N1364759_AAI_Shah.ipynb first."
            )
        if not self._class_names_path.exists():
            raise FileNotFoundError(
                f"Class-names file not found: {self._class_names_path}\n"
                "Run the export cell in N1364759_AAI_Shah.ipynb first."
            )

        logger.info("Loading class names from %s", self._class_names_path)
        with self._class_names_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        self._class_names = meta["class_names"]
        self._img_size = int(meta.get("img_size", 96))
        logger.info("Classes: %s", self._class_names)

        logger.info("Loading model from %s", self._model_path)
        try:
            tf = _get_tf()
            self._model = tf.keras.models.load_model(str(self._model_path))
            logger.info("Model loaded. Input shape: %s", self._model.input_shape)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model from {self._model_path}: {exc}"
            ) from exc

    def _preprocess(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Convert any image input to the model's expected tensor format.

        Args:
            image: A PIL Image (any mode) or a numpy array of shape
                ``(H, W)``, ``(H, W, 1)``, or ``(H, W, 3)``.

        Returns:
            Float32 numpy array of shape ``(1, img_size, img_size, 1)``
            with values in ``[0, 1]``.

        Raises:
            TypeError: If *image* is neither a PIL Image nor a numpy array.
            ValueError: If the numpy array has an unsupported number of
                dimensions.
        """
        if isinstance(image, Image.Image):
            img_gray = image.convert("L")
            img_arr = np.array(img_gray, dtype=np.float32)
        elif isinstance(image, np.ndarray):
            arr = image.astype(np.float32)
            if arr.ndim == 2:
                img_arr = arr
            elif arr.ndim == 3 and arr.shape[2] == 1:
                img_arr = arr[:, :, 0]
            elif arr.ndim == 3 and arr.shape[2] == 3:
                # Weighted RGB→grayscale (same as PIL's 'L' mode)
                img_arr = (
                    0.299 * arr[:, :, 0]
                    + 0.587 * arr[:, :, 1]
                    + 0.114 * arr[:, :, 2]
                )
            else:
                raise ValueError(
                    f"Unsupported numpy array shape: {image.shape}. "
                    "Expected (H, W), (H, W, 1), or (H, W, 3)."
                )
        else:
            raise TypeError(
                f"Expected PIL.Image or numpy.ndarray, got {type(image).__name__}."
            )

        # Normalize to [0, 1] if the array appears to be in [0, 255] range
        if img_arr.max() > 1.0:
            img_arr = img_arr / 255.0

        # Resize to model's expected spatial resolution
        tf = _get_tf()
        resized = tf.image.resize(
            img_arr[..., np.newaxis], [self._img_size, self._img_size]
        ).numpy()

        # Shape: (1, img_size, img_size, 1)
        return resized[np.newaxis].astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self, image: Union[Image.Image, np.ndarray]
    ) -> Dict[str, object]:
        """Run defect classification on a single image.

        Args:
            image: Input image as a PIL Image or numpy array. Any spatial
                size is accepted — it will be resized automatically.

        Returns:
            A dictionary with the following keys:

            * ``predicted_class`` (str): Human-readable defect class name,
              e.g. ``"crazing"``.
            * ``confidence`` (float): Probability of the predicted class,
              in ``[0, 1]``.
            * ``all_probabilities`` (dict[str, float]): Mapping from every
              class name to its softmax probability.
            * ``inference_time_ms`` (float): Wall-clock inference duration
              in milliseconds.

        Raises:
            TypeError: If *image* is the wrong type.
            ValueError: If the array shape is unsupported.
            RuntimeError: If the model has not been loaded successfully.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded. Check initialisation errors.")

        tensor = self._preprocess(image)

        t0 = time.perf_counter()
        probs = self._model(tensor, training=False).numpy()[0]
        inference_ms = (time.perf_counter() - t0) * 1_000

        pred_idx = int(np.argmax(probs))
        predicted_class = self._class_names[pred_idx]
        confidence = float(probs[pred_idx])
        all_probabilities = {
            name: float(probs[i]) for i, name in enumerate(self._class_names)
        }

        logger.debug(
            "Prediction: %s (%.2f%%) in %.1f ms",
            predicted_class,
            confidence * 100,
            inference_ms,
        )

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "inference_time_ms": round(inference_ms, 2),
        }

    @property
    def class_names(self) -> List[str]:
        """Ordered list of defect class names."""
        return list(self._class_names)

    @property
    def img_size(self) -> int:
        """Spatial resolution the model expects (pixels per side)."""
        return self._img_size
