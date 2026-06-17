from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from src.crop_prediction.predict import CropPredictor, load_metadata
from src.crop_prediction.preprocess import clean_crop_data, load_crop_data
from src.weed_detection.detect import run_detection

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
CROP_DATA_PATH = BASE_DIR / "data" / "crop_production" / "PLACE_DATASET_HERE.csv"
WEED_DATA_DIR = BASE_DIR / "data" / "weed_detection"
CROP_MODEL_PATH = BASE_DIR / "models" / "crop_prediction_model.pkl"
WEED_MODEL_PATH = BASE_DIR / "models" / "best.pt"

st.set_page_config(page_title="AgriSense-AI", page_icon="🌾", layout="wide")


def _safe_load_crop_data() -> pd.DataFrame | None:
    try:
        return clean_crop_data(load_crop_data(CROP_DATA_PATH))
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        LOGGER.warning("Crop data unavailable: %s", exc)
        return None


def home_page() -> None:
    st.title("🌾 AgriSense-AI")
    st.subheader("Smart Agriculture Analytics Platform")
    st.markdown("""
### Objectives
- Predict crop production using machine learning.
- Detect crops and weeds from images using YOLOv8.

### Technology Stack
- Python, Streamlit
- Scikit-learn, Pandas, NumPy
- OpenCV, Ultralytics YOLOv8
""")
    st.info("Architecture: Data ➜ Training ➜ Saved Models ➜ Streamlit Analytics & Inference")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            CropData -> CropTraining -> CropModel;
            WeedData -> YOLOTraining -> YOLOModel;
            CropModel -> Dashboard;
            YOLOModel -> Dashboard;
        }
        """
    )


def crop_prediction_page() -> None:
    st.header("Crop Production Prediction")
    data = _safe_load_crop_data()

    if data is None:
        st.error("Crop dataset is missing. Add CSV at data/crop_production/PLACE_DATASET_HERE.csv")
        return

    states = sorted(data["State"].dropna().unique().tolist())
    crops = sorted(data["Crop"].dropna().unique().tolist())
    seasons = sorted(data["Season"].dropna().unique().tolist())

    col1, col2, col3, col4 = st.columns(4)
    state = col1.selectbox("State", states)
    crop = col2.selectbox("Crop", crops)
    season = col3.selectbox("Season", seasons)
    cost = col4.number_input("Cost", min_value=0.0, value=1000.0, step=100.0)

    if st.button("Predict", type="primary"):
        try:
            predictor = CropPredictor(CROP_MODEL_PATH)
            prediction = predictor.predict(state=state, crop=crop, season=season, cost=cost)
            st.success(f"Predicted Production: {prediction:,.2f}")
        except FileNotFoundError:
            st.error("Model missing. Train and save models/crop_prediction_model.pkl")
        except (ValueError, RuntimeError) as exc:
            st.error(f"Prediction failed: {exc}")

    st.subheader("Model Performance")
    metadata = load_metadata(CROP_MODEL_PATH)
    if metadata:
        st.json(metadata)
    else:
        st.info("Model metadata not available.")

    st.subheader("Charts")
    for path in [
        BASE_DIR / "outputs" / "plots" / "crop_correlation_heatmap.png",
        BASE_DIR / "outputs" / "plots" / "feature_importance.png",
        BASE_DIR / "outputs" / "plots" / "prediction_vs_actual.png",
    ]:
        if path.exists():
            st.image(str(path), caption=path.name)


def weed_detection_page() -> None:
    st.header("Crop and Weed Detection")
    uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        temp_dir = BASE_DIR / "outputs" / "predictions"
        temp_dir.mkdir(parents=True, exist_ok=True)
        image_path = temp_dir / uploaded.name
        image_path.write_bytes(uploaded.read())
        st.image(str(image_path), caption=f"Uploaded image: {uploaded.name}", use_container_width=True)

        if st.button("Run YOLO Inference", type="primary"):
            try:
                result = run_detection(WEED_MODEL_PATH, image_path, output_dir=temp_dir)
                total_detections = len(result["labels"])
                st.image(
                    result["output_image"],
                    caption=f"Detected output ({total_detections} detections)",
                    use_container_width=True,
                )
                st.write("Class counts:", result["counts"])
                st.write(
                    "Detections:",
                    [
                        {"label": label, "confidence": round(conf, 4)}
                        for label, conf in zip(result["labels"], result["confidences"])
                    ],
                )
            except FileNotFoundError:
                st.error("YOLO model missing. Add models/best.pt or train model first.")
            except (ValueError, RuntimeError) as exc:
                st.error(f"Inference failed: {exc}")


def analytics_page() -> None:
    st.header("Analytics")
    data = _safe_load_crop_data()
    if data is not None:
        st.subheader("Crop Dataset Statistics")
        st.dataframe(data.describe(include="all"))
        st.subheader("Feature Distributions")
        st.bar_chart(data["State"].value_counts().head(10))

    metrics_file = BASE_DIR / "outputs" / "reports" / "crop_model_metrics.json"
    if metrics_file.exists():
        st.subheader("Crop Model Metrics")
        st.json(json.loads(metrics_file.read_text(encoding="utf-8")))

    yolo_metrics_file = BASE_DIR / "outputs" / "reports" / "yolo_metrics.json"
    if yolo_metrics_file.exists():
        st.subheader("YOLO Metrics")
        st.json(json.loads(yolo_metrics_file.read_text(encoding="utf-8")))


def about_page() -> None:
    st.header("About")
    st.markdown(
        """
AgriSense-AI combines machine learning and computer vision for modern agriculture.

### Problem Statement
Farmers need production forecasting and automated crop/weed visual monitoring.

### Future Scope
- IoT sensor integration
- Mobile deployment
- Geospatial analytics

### Developer
Raunak-24
"""
    )


PAGES = {
    "Home": home_page,
    "Crop Production Prediction": crop_prediction_page,
    "Crop and Weed Detection": weed_detection_page,
    "Analytics": analytics_page,
    "About": about_page,
}

choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
PAGES[choice]()
