"""Per-dataset settings (configs/<name>.yaml) and the file paths derived from them."""
from pathlib import Path
from types import SimpleNamespace

import yaml


def load_config(path):
    c = SimpleNamespace(**yaml.safe_load(Path(path).read_text()))
    h, w = c.onnx_imgsz
    model = Path(c.model).stem
    c.data = f"data/coconut_{c.name}"  # prepared dataset (prepare_dataset.py)
    c.run_dir = f"train_results/{c.name}_{model}_{c.imgsz}"  # training run (train.py)
    c.best = f"{c.run_dir}/weights/best.pt"
    c.onnx = f"{c.run_dir}/weights/coconut_{c.name}_{model}_{h}x{w}.onnx"  # CPU model (export_onnx.py)
    return c
