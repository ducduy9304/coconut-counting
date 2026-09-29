"""Export the trained detector to ONNX for CPU inference.

The belt crop is portrait (640x1080), so the model is exported at a fixed rectangular input
(default 480x288 HxW) instead of 480x480: the crop scaled to 480 tall is ~284 wide, so a square
input would spend ~40% of its compute on padding.

YOLO26 has two heads. By default the one-to-many head is exported (raw output (1, 4 + nc, anchors),
needs NMS; bench_cpu.py does it): on the lv1 valid set it scores P/R 0.995/0.995 vs 0.973/0.976 for
the NMS-free one-to-one head, and it is what Ultralytics' own predict/track (count.py) uses.
--end2end exports the one-to-one head instead: output (1, 300, 6) = x1, y1, x2, y2, score, class.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--weights", default="train_results/lv1_yolo26s_480/weights/best.pt")
p.add_argument("--imgsz", type=int, nargs=2, default=[480, 288], metavar=("H", "W"), help="multiples of 32")
p.add_argument("--end2end", action="store_true", help="export the NMS-free one-to-one head")
p.add_argument("--out", default="train_results/lv1_yolo26s_480/weights/coconut_lv1_yolo26s_480x288.onnx")
a = p.parse_args()

f = YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, dynamic=False, simplify=True, device="cpu",
                          nms=False if a.end2end else None)  # nms=False selects the end2end head
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
shutil.move(f, out)
print(f"-> {out}")
