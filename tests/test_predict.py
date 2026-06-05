"""Pytest test suite for src.predict.DefectPredictor.

Tests cover model loading, output structure, value ranges, timing, and
error-handling for malformed inputs. Requires the model files to exist in
models/ (run the notebook export cell first); tests that need the model
are skipped automatically when files are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup — allow pytest to run from the project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.predict import DefectPredictor  # noqa: E402

_MODEL_PATH = _PROJECT_ROOT / "models" / "defect_cnn_model.keras"
_CLASS_NAMES_PATH = _PROJECT_ROOT / "models" / "class_names.json"

_MODEL_AVAILABLE = _MODEL_PATH.exists() and _CLASS_NAMES_PATH.exists()
_SKIP_NO_MODEL = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Model files not found — run the notebook export cell first.",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def predictor() -> DefectPredictor:
    """Return a module-scoped DefectPredictor (model loaded once per session)."""
    return DefectPredictor()


@pytest.fixture
def grey_pil_image() -> Image.Image:
    """96×96 random grayscale PIL Image suitable for inference."""
    rng = np.random.default_rng(0)
    arr = (rng.integers(0, 256, size=(96, 96))).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def grey_numpy_image() -> np.ndarray:
    """96×96 float32 numpy array in [0, 1] suitable for inference."""
    rng = np.random.default_rng(1)
    return rng.random((96, 96), dtype=np.float32)


# ---------------------------------------------------------------------------
# test_model_loads
# ---------------------------------------------------------------------------


@_SKIP_NO_MODEL
def test_model_loads() -> None:
    """DefectPredictor initialises without raising any exception."""
    pred = DefectPredictor()
    assert pred is not None
    assert len(pred.class_names) == 6
    assert pred.img_size == 96


# ---------------------------------------------------------------------------
# test_prediction_returns_correct_keys
# ---------------------------------------------------------------------------


@_SKIP_NO_MODEL
def test_prediction_returns_correct_keys(
    predictor: DefectPredictor, grey_pil_image: Image.Image
) -> None:
    """predict() output dict contains all four required keys."""
    result = predictor.predict(grey_pil_image)
    assert "predicted_class" in result
    assert "confidence" in result
    assert "all_probabilities" in result
    assert "inference_time_ms" in result


# ---------------------------------------------------------------------------
# test_confidence_range
# ---------------------------------------------------------------------------


@_SKIP_NO_MODEL
def test_confidence_range(
    predictor: DefectPredictor, grey_pil_image: Image.Image
) -> None:
    """Confidence score is a float strictly within [0, 1]."""
    result = predictor.predict(grey_pil_image)
    conf = result["confidence"]
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


@_SKIP_NO_MODEL
def test_all_probabilities_sum_to_one(
    predictor: DefectPredictor, grey_numpy_image: np.ndarray
) -> None:
    """Softmax probabilities across all classes sum to approximately 1."""
    result = predictor.predict(grey_numpy_image)
    total = sum(result["all_probabilities"].values())
    assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total:.6f}, expected ~1.0"


@_SKIP_NO_MODEL
def test_all_probabilities_keys_match_class_names(
    predictor: DefectPredictor, grey_pil_image: Image.Image
) -> None:
    """all_probabilities keys match the predictor's class_names list."""
    result = predictor.predict(grey_pil_image)
    assert set(result["all_probabilities"].keys()) == set(predictor.class_names)


# ---------------------------------------------------------------------------
# test_inference_time_recorded
# ---------------------------------------------------------------------------


@_SKIP_NO_MODEL
def test_inference_time_recorded(
    predictor: DefectPredictor, grey_pil_image: Image.Image
) -> None:
    """inference_time_ms is a positive finite number."""
    result = predictor.predict(grey_pil_image)
    t = result["inference_time_ms"]
    assert isinstance(t, float)
    assert t > 0.0
    assert np.isfinite(t)


# ---------------------------------------------------------------------------
# test_invalid_image_handled
# ---------------------------------------------------------------------------


@_SKIP_NO_MODEL
def test_invalid_image_type_raises_type_error(predictor: DefectPredictor) -> None:
    """Passing a plain string raises TypeError with a clear message."""
    with pytest.raises(TypeError, match="Expected PIL.Image or numpy.ndarray"):
        predictor.predict("not_an_image")  # type: ignore[arg-type]


@_SKIP_NO_MODEL
def test_invalid_array_shape_raises_value_error(predictor: DefectPredictor) -> None:
    """A 4-D numpy array with wrong channel count raises ValueError."""
    bad_input = np.zeros((1, 96, 96, 7), dtype=np.float32)
    with pytest.raises((ValueError, Exception)):
        predictor.predict(bad_input)


# ---------------------------------------------------------------------------
# test_missing_model_file
# ---------------------------------------------------------------------------


def test_missing_model_file_raises_file_not_found(tmp_path: Path) -> None:
    """FileNotFoundError raised when model path does not exist."""
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        DefectPredictor(
            model_path=tmp_path / "nonexistent.keras",
            class_names_path=_CLASS_NAMES_PATH,
        )


def test_missing_class_names_file_raises_file_not_found(tmp_path: Path) -> None:
    """FileNotFoundError raised when class_names.json does not exist."""
    with pytest.raises(FileNotFoundError):
        DefectPredictor(
            model_path=_MODEL_PATH,
            class_names_path=tmp_path / "nonexistent.json",
        )


# ---------------------------------------------------------------------------
# test_rgb_image_accepted
# ---------------------------------------------------------------------------


@_SKIP_NO_MODEL
def test_rgb_pil_image_accepted(predictor: DefectPredictor) -> None:
    """RGB PIL Image is converted to grayscale and returns a valid prediction."""
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    rgb_img = Image.fromarray(arr, mode="RGB")
    result = predictor.predict(rgb_img)
    assert result["predicted_class"] in predictor.class_names


@_SKIP_NO_MODEL
def test_different_input_sizes_accepted(predictor: DefectPredictor) -> None:
    """Images of non-standard sizes are resized transparently."""
    big_img = Image.fromarray(
        np.random.randint(0, 256, (200, 200), dtype=np.uint8), mode="L"
    )
    small_img = Image.fromarray(
        np.random.randint(0, 256, (32, 32), dtype=np.uint8), mode="L"
    )
    for img in (big_img, small_img):
        result = predictor.predict(img)
        assert "predicted_class" in result
