"""Streamlit web application for Industrial Surface Defect Detection.

Provides a drag-and-drop image upload interface that runs the trained
NEU_CNN_96 model and displays the predicted defect class, confidence,
and full class probability chart.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup — allow running from the project root OR from the app/ folder
# ---------------------------------------------------------------------------
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.predict import DefectPredictor  # noqa: E402
from src.utils import get_confidence_color  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark-friendly, professional
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Main background */
        .main { background-color: #0f0f1a; }
        /* Card-like containers */
        .stApp { background-color: #0f0f1a; color: #e2e8f0; }
        /* Upload box */
        .stFileUploader > div { background-color: #1e1e2e; border: 2px dashed #4f46e5;
                                border-radius: 12px; padding: 20px; }
        /* Metric cards */
        div[data-testid="metric-container"] { background-color: #1e1e2e;
            border-radius: 10px; padding: 14px; border-left: 4px solid #4f46e5; }
        /* Sidebar */
        section[data-testid="stSidebar"] { background-color: #12122a; }
        /* Footer */
        .footer { text-align: center; color: #6b7280; font-size: 0.82rem;
                  margin-top: 40px; padding-top: 20px;
                  border-top: 1px solid #2d2d4e; }
        /* Confidence bar wrapper */
        .conf-bar-outer { background: #2d2d4e; border-radius: 8px;
                          height: 22px; width: 100%; }
        .conf-bar-inner { height: 22px; border-radius: 8px;
                          display: flex; align-items: center;
                          padding-left: 8px; font-size: 0.8rem;
                          font-weight: 600; color: #0f0f1a; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFECT_INFO = {
    "crazing": {
        "icon": "🕸️",
        "description": (
            "Network of fine, interconnected cracks on the steel surface. "
            "Caused by thermal fatigue, rapid cooling, or tensile stress during rolling."
        ),
    },
    "inclusion": {
        "icon": "🔵",
        "description": (
            "Non-metallic particles (slag, oxides, sulphides) trapped inside the steel "
            "during casting or solidification. Reduces mechanical strength."
        ),
    },
    "patches": {
        "icon": "🟤",
        "description": (
            "Irregular regions of discolouration or altered texture, often from chemical "
            "contamination, uneven oxidation, or scale residue."
        ),
    },
    "pitted_surface": {
        "icon": "🕳️",
        "description": (
            "Small cavities or pits on the surface caused by corrosion, gas bubbles "
            "during solidification, or mechanical impact during handling."
        ),
    },
    "rolled-in_scale": {
        "icon": "📄",
        "description": (
            "Oxide scale pressed into the steel surface during hot rolling, leaving "
            "dark embedded flakes or streaks that weaken the surface layer."
        ),
    },
    "scratches": {
        "icon": "✏️",
        "description": (
            "Linear surface marks caused by abrasion during mechanical contact, "
            "tool wear, or improper handling in transport or manufacturing."
        ),
    },
}

_SAMPLE_DIR = _PROJECT_ROOT / "sample_images"


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading defect detection model …")
def load_predictor() -> DefectPredictor | None:
    """Load and cache the DefectPredictor singleton.

    Returns:
        A ready-to-use :class:`DefectPredictor`, or ``None`` if the model
        files are not found (with an error shown in the UI).
    """
    try:
        return DefectPredictor()
    except FileNotFoundError as exc:
        st.error(
            f"**Model files not found.**\n\n{exc}\n\n"
            "Run the export cell in `N1364759_AAI_Shah.ipynb` to generate "
            "`models/defect_cnn_model.keras` and `models/class_names.json`."
        )
        return None
    except RuntimeError as exc:
        st.error(f"**Failed to load model:** {exc}")
        return None


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _confidence_bar(confidence: float, color: str) -> None:
    """Render a custom HTML confidence progress bar.

    Args:
        confidence: Value in ``[0, 1]``.
        color: Hex colour string for the filled portion.
    """
    pct = confidence * 100
    st.markdown(
        f"""
        <div class="conf-bar-outer">
          <div class="conf-bar-inner"
               style="width:{pct:.1f}%; background:{color};">
            {pct:.1f}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plotly_probability_chart(
    all_probs: dict[str, float], predicted_class: str
) -> go.Figure:
    """Build a Plotly horizontal bar chart of all class probabilities.

    Args:
        all_probs: Mapping from class name to probability.
        predicted_class: The top-predicted class (highlighted in a
            different colour).

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    classes = list(all_probs.keys())
    probs = [all_probs[c] * 100 for c in classes]
    labels = [c.replace("_", " ").title() for c in classes]
    colors = ["#4f46e5" if c == predicted_class else "#334155" for c in classes]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1f}%" for p in probs],
            textposition="outside",
            textfont=dict(color="white", size=11),
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="white", size=11),
        xaxis=dict(
            title="Probability (%)",
            range=[0, 115],
            gridcolor="#2d2d4e",
            color="white",
        ),
        yaxis=dict(color="white", autorange="reversed"),
        margin=dict(l=0, r=10, t=10, b=30),
        height=280,
        showlegend=False,
    )
    return fig


def _run_inference(predictor: DefectPredictor, pil_image: Image.Image) -> dict:
    """Run prediction and return the result dict.

    Args:
        predictor: Loaded :class:`DefectPredictor`.
        pil_image: Uploaded image as a PIL Image.

    Returns:
        Prediction dict from :meth:`DefectPredictor.predict`.
    """
    with st.spinner("Running defect analysis …"):
        return predictor.predict(pil_image)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    """Render the About and Model Info sections in the sidebar."""
    with st.sidebar:
        st.markdown("## 🔬 About This App")
        st.markdown(
            "This tool classifies **steel surface defects** using a Convolutional "
            "Neural Network trained on the NEU Surface Defect Database.\n\n"
            "Upload a grayscale or colour image of a steel surface and the model "
            "will identify the defect type instantly."
        )

        st.divider()
        st.markdown("### 📋 Defect Classes")
        for class_name, info in _DEFECT_INFO.items():
            with st.expander(
                f"{info['icon']}  {class_name.replace('_', ' ').title()}"
            ):
                st.markdown(info["description"])

        st.divider()
        st.markdown("### ⚙️ Model Info")
        st.markdown(
            """
| Property | Value |
|---|---|
| Architecture | NEU_CNN_96 |
| Conv blocks | 4 (32→64→128→256) |
| Input size | 96 × 96 grayscale |
| Test accuracy | **95.28%** |
| Macro F1 | **0.9528** |
| Dataset | NEU-DET (1,800 imgs) |
| Optimizer | Adam + ReduceLROnPlateau |
| Training epochs | Up to 60 (early stop) |
            """
        )

        st.divider()
        st.markdown("### 📦 Dataset")
        st.markdown(
            "**NEU Surface Defect Database** — 1,800 steel surface images "
            "across 6 perfectly balanced classes (300 per class). "
            "Source: Northeastern University, China."
        )


# ---------------------------------------------------------------------------
# Sample images section
# ---------------------------------------------------------------------------

def _render_sample_images(predictor: DefectPredictor) -> Image.Image | None:
    """Show clickable sample images if the sample_images/ folder is populated.

    Args:
        predictor: Loaded predictor (used if user selects a sample).

    Returns:
        The selected sample image as a PIL Image, or ``None`` if the folder
        is empty or no selection was made.
    """
    image_files = sorted(
        [
            p
            for p in _SAMPLE_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )
    if not image_files:
        return None

    st.markdown("---")
    st.markdown("### 🖼️ Sample Images")
    st.markdown("Click a thumbnail to run the model on a known example.")

    cols = st.columns(min(len(image_files), 6))
    for col, img_path in zip(cols, image_files[:6]):
        with col:
            thumb = Image.open(img_path).convert("L")
            if col.button(img_path.stem, use_column_width=True):
                st.session_state["selected_sample"] = img_path
            st.image(thumb, use_column_width=True)

    if "selected_sample" in st.session_state:
        return Image.open(st.session_state["selected_sample"])
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — renders the full Streamlit page."""
    _render_sidebar()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center; color:#e2e8f0;'>"
        "🏭 Industrial Surface Defect Detection"
        "</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#94a3b8; font-size:1.05rem;'>"
        "Deep Learning CNN trained on NEU-DET dataset — "
        "<strong style='color:#4f46e5;'>95.28% accuracy</strong> across 6 defect classes"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Load model ───────────────────────────────────────────────────────────
    predictor = load_predictor()
    if predictor is None:
        st.stop()

    # ── Upload area ──────────────────────────────────────────────────────────
    st.markdown("### 📤 Upload a Steel Surface Image")
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed",
    )

    # ── Sample images (fallback if no upload) ────────────────────────────────
    sample_image = _render_sample_images(predictor)

    # Decide which image to process
    pil_image: Image.Image | None = None
    image_source = ""

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)
            image_source = uploaded_file.name
            # Clear any previously selected sample when a new file is uploaded
            st.session_state.pop("selected_sample", None)
        except Exception as exc:
            st.error(f"Could not open uploaded file: {exc}")
            st.stop()
    elif sample_image is not None:
        pil_image = sample_image
        image_source = st.session_state.get("selected_sample", Path("sample")).name

    # ── Run inference & display results ──────────────────────────────────────
    if pil_image is not None:
        result = _run_inference(predictor, pil_image)

        predicted_class: str = result["predicted_class"]
        confidence: float = result["confidence"]
        all_probs: dict = result["all_probabilities"]
        inference_ms: float = result["inference_time_ms"]
        conf_color = get_confidence_color(confidence)

        st.divider()
        col_img, col_results = st.columns([1, 1], gap="large")

        # ── Left: uploaded image ─────────────────────────────────────────────
        with col_img:
            st.markdown("#### Input Image")
            st.image(
                pil_image.convert("L"),
                caption=image_source,
                use_column_width=True,
            )

        # ── Right: prediction results ────────────────────────────────────────
        with col_results:
            st.markdown("#### Analysis Results")

            # Predicted class badge
            display_name = predicted_class.replace("_", " ").title()
            defect_icon = _DEFECT_INFO.get(predicted_class, {}).get("icon", "⚠️")
            st.markdown(
                f"<div style='background:#1e1e2e; border-left:5px solid {conf_color}; "
                f"border-radius:8px; padding:14px 18px; margin-bottom:14px;'>"
                f"<span style='font-size:1.6rem;'>{defect_icon}</span>&nbsp;"
                f"<span style='font-size:1.4rem; font-weight:700; color:#e2e8f0;'>"
                f"{display_name}</span></div>",
                unsafe_allow_html=True,
            )

            # Confidence bar
            st.markdown("**Confidence**")
            _confidence_bar(confidence, conf_color)

            # Metrics row
            st.markdown("")
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{confidence * 100:.1f}%")
            m2.metric("Inference Time", f"{inference_ms:.1f} ms")

            # Defect description
            desc = _DEFECT_INFO.get(predicted_class, {}).get("description", "")
            if desc:
                st.info(f"ℹ️ {desc}")

        # ── Probability chart ─────────────────────────────────────────────────
        st.divider()
        st.markdown("#### Class Probability Distribution")
        fig = _plotly_probability_chart(all_probs, predicted_class)
        st.plotly_chart(fig, use_column_width=True)

        # ── Low-confidence warning ────────────────────────────────────────────
        if confidence < 0.60:
            st.warning(
                "⚠️ Low confidence prediction. The image may not closely "
                "resemble the training data or may contain multiple defect types. "
                "Consider re-imaging or consulting a human inspector."
            )

    else:
        # ── Placeholder when no image uploaded ───────────────────────────────
        st.markdown("")
        st.markdown(
            "<div style='text-align:center; padding:60px 0; color:#475569;'>"
            "<p style='font-size:3rem;'>🔍</p>"
            "<p style='font-size:1.1rem;'>Upload a steel surface image above to begin analysis</p>"
            "<p style='font-size:0.9rem;'>Supported formats: JPG · PNG · BMP</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='footer'>"
        "Built by <strong>Pratham Shah</strong> — "
        "MSc AI &amp; Data Science, Nottingham Trent University"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
