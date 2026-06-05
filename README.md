# Industrial Surface Defect Detection — CNN Classifier

A production-ready computer vision system that classifies steel surface defects in real time using a Convolutional Neural Network trained on the **NEU Surface Defect Database**. The model achieves **95.28% test accuracy** and **0.9528 macro F1-score** across all six defect categories, outperforming classical machine-learning baselines (MLP: 65.83%, Stacking Ensemble: 86.94%) by significant margins. The project ships a Streamlit web application for zero-code interactive inference and a clean Python API for integration into automated inspection pipelines.

---

## Key Result

| Metric | Value |
|---|---|
| **Test Accuracy** | **95.28%** |
| **Macro F1-Score** | **0.9528** |
| **Macro Precision** | ~0.953 |
| **Macro Recall** | ~0.953 |
| Dataset | NEU-DET (1,800 images, 6 classes) |
| Baseline (MLP) | 65.83% |
| Baseline (HOG + Ensemble) | 86.94% |
| **CNN improvement over ensemble** | **+8.34 pp** |

---

## The 6 Defect Classes

| Class | Description |
|---|---|
| **Crazing** | Network of fine, interconnected cracks caused by thermal fatigue or tensile stress during rolling. |
| **Inclusion** | Non-metallic particles (slag, oxides, sulphides) trapped inside the steel during casting or solidification. |
| **Patches** | Irregular discoloured regions caused by chemical contamination, uneven oxidation, or scale residue. |
| **Pitted Surface** | Small cavities or pits from corrosion, gas bubbles during solidification, or mechanical impact. |
| **Rolled-in Scale** | Oxide scale pressed into the surface during hot rolling, leaving embedded dark flakes or streaks. |
| **Scratches** | Linear surface marks from abrasion, tool wear, or improper handling in manufacturing or transport. |

The dataset is perfectly balanced — 300 images per class (240 train / 60 validation), making accuracy a reliable evaluation metric with no resampling required.

---

## Architecture

```
Input (96×96×1, float32 [0,1])
        │
┌───────┴──────────────────────────────────┐
│  Built-in Augmentation (training only)   │
│  RandomFlip(horizontal) + RandomRotation │
└───────┬──────────────────────────────────┘
        │
┌───────┴─────────────────────────────────────────────────────────────────┐
│ Block 1 │ Conv2D(32,3,same)→BN→ReLU → Conv2D(32,3,same)→BN→ReLU        │
│         │ MaxPool2D → Dropout(0.20)   Output: 48×48×32                  │
├─────────┴─────────────────────────────────────────────────────────────┐ │
│ Block 2 │ Conv2D(64,3,same)→BN→ReLU → Conv2D(64,3,same)→BN→ReLU      │ │
│         │ MaxPool2D → Dropout(0.25)   Output: 24×24×64                │ │
├─────────┴───────────────────────────────────────────────────────────┐ │ │
│ Block 3 │ Conv2D(128,3,same)→BN→ReLU → MaxPool2D → Dropout(0.30)   │ │ │
│         │ Output: 12×12×128                                         │ │ │
├─────────┴─────────────────────────────────────────────────────────┐ │ │ │
│ Block 4 │ Conv2D(256,3,same)→BN→ReLU → MaxPool2D → Dropout(0.35) │ │ │ │
│         │ Output: 6×6×256                                         │ │ │ │
└─────────────────────────────────────────────────────────────────┘ │ │ │
        │
        Flatten
        Dense(256) → BN → ReLU → Dropout(0.50)
        Dense(6, softmax)
        │
Output: 6-class probability distribution
```

**Optimizer**: Adam (lr=0.001) with ReduceLROnPlateau (factor=0.5, patience=6)  
**Loss**: Sparse Categorical Cross-Entropy  
**Regularisation**: Batch Normalisation + Dropout (0.20 → 0.50 progressive)  
**Early stopping**: patience=12, monitors val_accuracy, restores best weights  

---

## Setup and Run

### 1. Clone and create environment

```bash
git clone <your-repo-url>
cd defect-detection-cnn

# Conda (recommended)
conda create -n defect-detection python=3.11
conda activate defect-detection

# Or venv
python -m venv .venv && source .venv/bin/activate  # Linux/macOS
python -m venv .venv && .venv\Scripts\activate     # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train and export the model

Open `N1364759_AAI_Shah.ipynb` in Jupyter, run all cells through **Task C**, then run the final **Task 1 — Export** cell. This creates:

```
defect-detection-cnn/
└── models/
    ├── defect_cnn_model.keras   ← trained model weights
    └── class_names.json         ← class metadata
```

### 4. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### 5. Use the Python API

```python
from src.predict import DefectPredictor
from PIL import Image

predictor = DefectPredictor()
img = Image.open("sample_images/scratches_001.jpg")
result = predictor.predict(img)

print(result["predicted_class"])    # e.g. "scratches"
print(result["confidence"])         # e.g. 0.9872
print(result["inference_time_ms"])  # e.g. 14.3
```

### 6. Run tests

```bash
pytest tests/ -v
```

Tests that require the model file are automatically skipped if `models/defect_cnn_model.keras` is absent, so the test suite runs safely on a clean clone.

---

## Project Structure

```
defect-detection-cnn/
├── models/
│   ├── defect_cnn_model.keras   ← trained model (git-ignored, generate locally)
│   └── class_names.json         ← class metadata
├── src/
│   ├── __init__.py
│   ├── predict.py               ← DefectPredictor class
│   └── utils.py                 ← image loading, visualisation, helpers
├── app/
│   ├── __init__.py
│   └── streamlit_app.py         ← Streamlit web interface
├── tests/
│   └── test_predict.py          ← pytest test suite
├── sample_images/               ← place example images here
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Sample Results

| Image | True Class | Predicted | Confidence |
|---|---|---|---|
| crazing_001.jpg | Crazing | **Crazing** | 98.7% |
| scratches_042.jpg | Scratches | **Scratches** | 97.2% |
| pitted_surface_015.jpg | Pitted Surface | **Pitted Surface** | 94.1% |
| inclusion_007.jpg | Inclusion | **Inclusion** | 91.8% |

Overall test set: **95.28% accuracy** with no single class below 90%.

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep learning framework | TensorFlow / Keras |
| Image processing | Pillow, OpenCV |
| Web application | Streamlit |
| Interactive charts | Plotly |
| Static visualisations | Matplotlib |
| Data manipulation | NumPy, Pandas |
| Testing | pytest |
| Environment management | Anaconda / venv |

---

## Author

**Pratham Shah**  
MSc Artificial Intelligence & Data Science  
Nottingham Trent University  
`shahpratham00@gmail.com`

---

*Part of the Applied AI Coursework portfolio — MSc AI & Data Science, NTU, 2026.*
