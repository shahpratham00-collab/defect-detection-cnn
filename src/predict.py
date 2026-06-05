"""Defect prediction module using ONNX Runtime for cross-platform deployment."""

import json
import time
import logging
import pathlib
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

CLASS_NAMES_PATH = pathlib.Path(__file__).parent.parent / "models" / "class_names.json"
MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "defect_cnn_model.onnx"


class DefectPredictor:
    """Loads ONNX model and runs defect classification inference."""

    def __init__(self):
        """Initialise predictor by loading class names and ONNX model."""
        if not CLASS_NAMES_PATH.exists():
            raise FileNotFoundError(f"Class names not found: {CLASS_NAMES_PATH}")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

        with open(CLASS_NAMES_PATH) as f:
            config = json.load(f)

        self.class_names = config["class_names"]
        self.img_size = config["img_size"]
        logger.info(f"Classes: {self.class_names}")

        import onnxruntime as ort
        self.session = ort.InferenceSession(str(MODEL_PATH))
        self.input_name = self.session.get_inputs()[0].name
        logger.info("ONNX model loaded successfully")

    def predict(self, image) -> dict:
        """Run inference on a PIL Image or numpy array.

        Args:
            image: PIL Image or numpy array of the defect surface.

        Returns:
            dict with predicted_class, confidence, all_probabilities,
            inference_time_ms.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        img = image.convert("L").resize((self.img_size, self.img_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.reshape(1, self.img_size, self.img_size, 1)

        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: arr})
        elapsed = (time.perf_counter() - start) * 1000

        probs = outputs[0][0]
        pred_idx = int(np.argmax(probs))

        return {
            "predicted_class": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "all_probabilities": {
                name: float(prob)
                for name, prob in zip(self.class_names, probs)
            },
            "inference_time_ms": round(elapsed, 2),
        }
