"""Train the YOLO26 coconut detector on a prepared dataset (see tools/prepare_dataset.py).

Polygon labels are fine for detect: Ultralytics derives each box from its polygon.
Output goes to train_results/<name>_<model>_<imgsz> (see tools/config.py).
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from config import load_config

p = argparse.ArgumentParser()
p.add_argument("config", help="dataset config, e.g. configs/lv1.yaml")
c = load_config(p.parse_known_args()[0].config)
p.add_argument("--data", default=f"{c.data}/data.yaml")
p.add_argument("--model", default=c.model)
p.add_argument("--imgsz", type=int, default=c.imgsz)
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--patience", type=int, default=40)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--device", default="0")
p.add_argument("--project", default=str(Path(c.run_dir).parent))
p.add_argument("--name", default=Path(c.run_dir).name)
a = p.parse_args()

YOLO(a.model).train(
    data=a.data, imgsz=a.imgsz, epochs=a.epochs, patience=a.patience, batch=a.batch, device=a.device,
    flipud=c.flipud,  # only valid when the camera looks straight down on the belt
    degrees=10,
    cache="ram", workers=8, project=str(Path(a.project).resolve()), name=a.name, exist_ok=True,
)
