"""Train the YOLO26 coconut detector on the prepared dataset (see tools/prepare_dataset.py).

Polygon labels are fine for detect: Ultralytics derives each box from its polygon.
imgsz 480 matches the belt-crop input planned for CPU inference.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/coconut_lv1/data.yaml")
p.add_argument("--model", default="yolo26s.pt")
p.add_argument("--imgsz", type=int, default=480)
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--patience", type=int, default=40)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--device", default="0")
p.add_argument("--project", default="train_results")
p.add_argument("--name", default="lv1_yolo26s_480")
a = p.parse_args()

YOLO(a.model).train(
    data=a.data, imgsz=a.imgsz, epochs=a.epochs, patience=a.patience, batch=a.batch, device=a.device,
    flipud=0.5,  # top-down belt: vertical flips are valid views
    degrees=10,
    cache="ram", workers=8, project=str(Path(a.project).resolve()), name=a.name, exist_ok=True,
)
