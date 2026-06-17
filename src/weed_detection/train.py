from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def validate_dataset_structure(base_dir: Path) -> Dict[str, Path]:
    images = base_dir / "images"
    labels = base_dir / "labels"
    if not images.exists() or not labels.exists():
        raise FileNotFoundError("Expected weed dataset folders: data/weed_detection/images and labels")
    return {"images": images, "labels": labels}


def infer_class_names(labels_dir: Path) -> List[str]:
    max_class_id = -1
    for label_file in labels_dir.glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    max_class_id = max(max_class_id, int(parts[0]))
                except ValueError:
                    continue

    if max_class_id < 0:
        return ["crop", "weed"]
    return [f"class_{i}" for i in range(max_class_id + 1)]


def create_data_yaml(base_dir: Path, yaml_path: Path | None = None) -> Path:
    paths = validate_dataset_structure(base_dir)
    yaml_path = yaml_path or base_dir / "data.yaml"
    names = infer_class_names(paths["labels"])

    payload = {
        "path": str(base_dir),
        "train": "images",
        "val": "images",
        "test": "images",
        "names": {i: n for i, n in enumerate(names)},
        "nc": len(names),
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    LOGGER.info("Generated data.yaml at %s", yaml_path)
    return yaml_path


def train_yolo(data_yaml: Path, model_out: Path, epochs: int = 30, imgsz: int = 640) -> None:
    model = YOLO("yolov8n.pt")
    results = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, project="outputs", name="weed_detection")
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    if best_path.exists():
        model_out.write_bytes(best_path.read_bytes())
        LOGGER.info("Saved trained model to %s", model_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for crop/weed detection")
    parser.add_argument("--data-dir", default="data/weed_detection")
    parser.add_argument("--output-model", default="models/best.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    yaml_path = create_data_yaml(Path(args.data_dir))
    train_yolo(yaml_path, Path(args.output_model), epochs=args.epochs, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
