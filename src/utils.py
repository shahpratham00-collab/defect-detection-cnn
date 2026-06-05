"""Utility functions for image loading, preprocessing, and visualisation.

All functions operate with pathlib.Path and follow Google-style docstrings.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger for the given module name.

    The logger writes to stdout with level INFO by default. A second call
    with the same *name* returns the cached logger without adding duplicate
    handlers.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.

    Example::

        logger = get_logger(__name__)
        logger.info("Model loaded successfully.")
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Pin all random-number generators for reproducible results.

    Sets Python's built-in ``random``, ``numpy``, and (if available)
    TensorFlow seeds.

    Args:
        seed: Integer seed value. Defaults to ``42``.

    Example::

        set_seed(0)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf  # noqa: PLC0415
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Image loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (96, 96),
) -> np.ndarray:
    """Load an image from disk, convert to grayscale, and resize it.

    The returned array is a float32 tensor normalised to ``[0, 1]`` with
    shape ``(H, W, 1)`` suitable for direct model input after adding the
    batch dimension.

    Args:
        image_path: Absolute or relative path to a JPEG, PNG, or BMP file.
        target_size: ``(height, width)`` to resize the image to.
            Defaults to ``(96, 96)`` to match the trained model.

    Returns:
        Float32 numpy array of shape ``(target_size[0], target_size[1], 1)``
        with values in ``[0, 1]``.

    Raises:
        FileNotFoundError: If *image_path* does not exist.
        ValueError: If the file cannot be read as an image by Pillow.

    Example::

        arr = load_and_preprocess_image("surface.jpg")
        batch = arr[np.newaxis]  # shape: (1, 96, 96, 1)
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        img = Image.open(path).convert("L")
    except Exception as exc:
        raise ValueError(f"Could not open image at {path}: {exc}") from exc

    img_resized = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return arr[:, :, np.newaxis]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_prediction(
    image: Union[Image.Image, np.ndarray],
    prediction_dict: Dict[str, object],
) -> matplotlib.figure.Figure:
    """Create a two-panel matplotlib figure showing the image and probabilities.

    Left panel: the input image (displayed in grayscale).
    Right panel: horizontal bar chart of all class probabilities, with the
    predicted class highlighted.

    Args:
        image: The source image as a PIL Image or numpy array.
        prediction_dict: Output from :meth:`DefectPredictor.predict`, which
            must contain keys ``predicted_class``, ``confidence``,
            ``all_probabilities``, and ``inference_time_ms``.

    Returns:
        A :class:`matplotlib.figure.Figure` with two axes.

    Raises:
        KeyError: If *prediction_dict* is missing required keys.

    Example::

        fig = visualise_prediction(img, predictor.predict(img))
        fig.savefig("result.png", dpi=150)
    """
    predicted_class: str = prediction_dict["predicted_class"]
    confidence: float = prediction_dict["confidence"]
    all_probs: Dict[str, float] = prediction_dict["all_probabilities"]
    inference_ms: float = prediction_dict["inference_time_ms"]

    # Convert image to displayable numpy array
    if isinstance(image, Image.Image):
        display_arr = np.array(image.convert("L"))
    elif isinstance(image, np.ndarray):
        arr = image.squeeze()
        display_arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    else:
        raise TypeError(f"Expected PIL Image or numpy array, got {type(image).__name__}.")

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#1e1e2e")

    # ── Left panel: image ────────────────────────────────────────────────────
    ax_img.imshow(display_arr, cmap="gray", vmin=0, vmax=255)
    ax_img.set_title(
        f"Prediction: {predicted_class.replace('_', ' ').title()}\n"
        f"Confidence: {confidence * 100:.1f}%  |  {inference_ms:.1f} ms",
        color="white",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax_img.axis("off")
    ax_img.set_facecolor("#1e1e2e")

    # ── Right panel: probability bars ────────────────────────────────────────
    classes = list(all_probs.keys())
    probs = [all_probs[c] for c in classes]
    labels = [c.replace("_", " ").title() for c in classes]
    colors = [
        "#4ade80" if c == predicted_class else "#60a5fa" for c in classes
    ]

    bars = ax_bar.barh(labels, probs, color=colors, edgecolor="none", height=0.55)
    for bar, val in zip(bars, probs):
        ax_bar.text(
            min(val + 0.01, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            color="white",
        )

    ax_bar.set_xlim(0, 1.12)
    ax_bar.set_xlabel("Probability", color="white", fontsize=10)
    ax_bar.set_title("Class Probabilities", color="white", fontsize=11, fontweight="bold")
    ax_bar.tick_params(colors="white")
    ax_bar.spines[:].set_color("#444466")
    ax_bar.set_facecolor("#1e1e2e")
    for label in ax_bar.get_yticklabels():
        label.set_color("white")

    plt.tight_layout(pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def get_confidence_color(confidence: float) -> str:
    """Return a CSS colour string based on prediction confidence level.

    The thresholds match the Streamlit app's progress-bar colouring:

    * ``≥ 0.80`` → green  (``"#4ade80"``)
    * ``0.60 – 0.80`` → amber  (``"#fbbf24"``)
    * ``< 0.60``  → red  (``"#f87171"``)

    Args:
        confidence: Predicted class probability, expected in ``[0, 1]``.

    Returns:
        Hex colour string.

    Raises:
        ValueError: If *confidence* is outside ``[0, 1]``.

    Example::

        color = get_confidence_color(0.92)  # returns "#4ade80"
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"Confidence must be in [0, 1], got {confidence:.4f}."
        )
    if confidence >= 0.80:
        return "#4ade80"
    if confidence >= 0.60:
        return "#fbbf24"
    return "#f87171"
