from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from ultralytics import YOLO


def evaluate_model(
    model_path: str | Path,
    data_yaml: str | Path,
    output_report: str | Path = "outputs/reports/yolo_metrics.json",
    split: str = "val",
) -> Dict[str, float]:
    model = YOLO(str(model_path))
    results = model.val(data=str(data_yaml), split=split, plots=True)

    metrics = {
        "precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0.0)),
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
    }

    output_report = Path(output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_dir = Path(results.save_dir)
    for matrix_name in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        matrix_path = save_dir / matrix_name
        if matrix_path.exists():
            target = output_report.parent / matrix_name
            target.write_bytes(matrix_path.read_bytes())

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate/Test YOLO model and export metrics")
    parser.add_argument("--model-path", default="models/best.pt")
    parser.add_argument("--data-yaml", default="data/weed_detection/data.yaml")
    parser.add_argument("--output-report", default="outputs/reports/yolo_metrics.json")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_yaml=args.data_yaml,
        output_report=args.output_report,
        split=args.split,
    )


if __name__ == "__main__":
    main()
