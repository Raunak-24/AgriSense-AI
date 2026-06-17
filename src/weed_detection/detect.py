from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
from ultralytics import YOLO


def run_detection(model_path: str | Path, image_path: str | Path, output_dir: str | Path = "outputs/predictions") -> Dict:
    model_path = Path(model_path)
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found at {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    results = model.predict(source=str(image_path), save=False, conf=0.25)

    result = results[0]
    boxes = result.boxes
    names = result.names

    labels: List[str] = []
    confidences: List[float] = []
    for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        labels.append(names[int(cls_id)])
        confidences.append(float(conf))

    counts = Counter(labels)
    plotted = result.plot()
    out_path = output_dir / f"detected_{image_path.name}"
    cv2.imwrite(str(out_path), plotted)

    return {
        "output_image": str(out_path),
        "labels": labels,
        "confidences": confidences,
        "counts": dict(counts),
    }
