# AgriSense-AI

AgriSense-AI is a smart agriculture analytics platform with two AI modules:
1. Crop Production Prediction (ML Regression)
2. Crop and Weed Detection (YOLOv8 Computer Vision)

## Problem Statement
Build a production-quality AI platform that predicts crop production and detects crops/weeds from images, with a Streamlit dashboard for portfolio and final-year engineering submissions.

## Features
- Data preprocessing with missing-value handling and duplicate removal
- Multi-model training (Linear Regression, Random Forest, Gradient Boosting)
- Automatic best-model selection using R², MAE, RMSE
- Saved model artifacts (`models/crop_prediction_model.pkl`, `models/best.pt`)
- YOLOv8 dataset validation and automatic `data.yaml` generation
- YOLO train/validate/test/inference scripts
- Streamlit multi-page dashboard (Home, Prediction, Detection, Analytics, About)
- User-friendly error handling when dataset/model files are missing

## Project Structure
```text
AgriSense-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── crop_production/
│   │   └── PLACE_DATASET_HERE.csv
│   └── weed_detection/
│       ├── images/
│       ├── labels/
│       └── data.yaml
├── notebooks/
│   ├── crop_prediction_eda.ipynb
│   └── weed_detection_training.ipynb
├── models/
│   ├── crop_prediction_model.pkl
│   └── best.pt
├── src/
│   ├── crop_prediction/
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   └── weed_detection/
│       ├── train.py
│       ├── detect.py
│       └── evaluate.py
├── outputs/
│   ├── plots/
│   ├── reports/
│   └── predictions/
└── screenshots/
```

## Dataset Instructions
### Crop Prediction
Place CSV at:
`data/crop_production/PLACE_DATASET_HERE.csv`

Required columns:
`Crop, Variety, State, Quantity, Production, Season, Unit, Cost, Recommended Zone`

### Weed Detection (YOLO format)
Place image/label files in:
- `data/weed_detection/images/`
- `data/weed_detection/labels/`

Then run training script to auto-create `data/weed_detection/data.yaml`.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
### 1) Train crop prediction model
```bash
python -m src.crop_prediction.train --dataset data/crop_production/PLACE_DATASET_HERE.csv
```

### 2) Train YOLO model
```bash
python -m src.weed_detection.train --data-dir data/weed_detection --output-model models/best.pt
```

### 3) Evaluate YOLO model
```bash
python -m src.weed_detection.evaluate
```

### 4) Run dashboard
```bash
streamlit run app.py
```

## Screenshots
Add dashboard screenshots in `/screenshots`.

## Results
- Crop regression metrics and plots are saved in `outputs/reports` and `outputs/plots`.
- YOLO metrics report is saved in `outputs/reports/yolo_metrics.json`.
- Detection outputs are saved in `outputs/predictions`.

## Future Scope
- Real-time drone feed support
- Edge deployment for low-connectivity farms
- Disease detection and recommendation engine

## Technologies Used
Python, Streamlit, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Joblib, OpenCV, Ultralytics YOLOv8
