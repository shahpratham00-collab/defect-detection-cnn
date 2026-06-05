"""Defect prediction module using ONNX Runtime — loads model from HuggingFace Hub."""

import json
import time
import logging
import pathlib
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

REPO_ID = "Shahpratham00/defect-detection-cnn"
MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"


def _ensure_models():
    """Download model files from HuggingFace Hub if not present locally."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "defect_cnn_model.onnx"
    names_path = MODEL_DIR / "class_names.json"
    if not model_path.exists():
        logger.info("Downloading ONNX model from HuggingFace...")
        hf_hub_download(repo_id=REPO_ID, filename="defect_cnn_model.onnx", local_dir=MODEL_DIR)
    if not names_path.exists():
        logger.info("Downloading class names from HuggingFace...")
        hf_hub_download(repo_id=REPO_ID, filename="class_names.json", local_dir=MODEL_DIR)
    return model_path, names_path


class DefectPredictor:
    """Loads ONNX model and runs defect classification inference."""

    def __init__(self):
        model_path, names_path = _ensure_models()
        with open(names_path) as f:
            config = json.load(f)
        self.class_names = config["class_names"]
        self.img_size = config["img_size"]
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        logger.info("ONNX model loaded successfully")

    def predict(self, image) -> dict:
        """Run inference on a PIL Image or numpy array."""
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
