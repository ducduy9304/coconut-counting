"""Count coconuts on the belt: YOLO26 detector + ByteTrack + a counting line (rules in counter.py).

Frames are cropped to the belt region exactly as in training (tools/prepare_dataset.py).
Writes an annotated video, count events (frame, time, id, x, y) and all track points as CSV.
For CPU/ONNX speed measurements use bench_cpu.py.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from counter import LineCounter

p = argparse.ArgumentParser()
p.add_argument("video", nargs="?", default="CoconutVideos/lv1.mp4")
p.add_argument("--weights", default="train_results/lv1_yolo26s_480/weights/best.pt")
p.add_argument("--crop", type=int, nargs=4, default=[620, 0, 1260, 1080], metavar=("X1", "Y1", "X2", "Y2"))
p.add_argument("--imgsz", type=int, default=480)
p.add_argument("--conf", type=float, default=0.1, help="low so ByteTrack's second-stage matching sees weak boxes")
p.add_argument("--tracker", default="bytetrack.yaml")
p.add_argument("--line", type=float, default=0.8, help="counting line, fraction of crop height")
p.add_argument("--mem", type=int, default=25, help="frames a count blocks re-counts of the same coconut")
p.add_argument("--gate", type=float, default=60, help="px in x within which a new track is the same coconut")
p.add_argument("--min-hits", type=int, default=3, help="detections a track needs before it can be counted")
p.add_argument("--out", default="inference_results")
p.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
p.add_argument("--no-video", action="store_true")
a = p.parse_args()

x1, y1, x2, y2 = a.crop
counter = LineCounter(a.line * (y2 - y1), a.mem, a.gate, a.min_hits)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
stem = Path(a.video).stem

model = YOLO(a.weights)
cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS)
writer = None if a.no_video else cv2.VideoWriter(
    str(out / f"{stem}_count.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (x2 - x1, y2 - y1))

points = []  # (frame, id, cx, cy, conf)
t0 = time.time()
f = -1
while True:
    ok, frame = cap.read()
    f += 1
    if not ok or (a.max_frames and f >= a.max_frames):
        break
    crop = frame[y1:y2, x1:x2]
    r = model.track(crop, persist=True, tracker=a.tracker, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
    boxes = r.boxes
    ids = boxes.id.int().tolist() if boxes.id is not None else []
    xyxy = boxes.xyxy.tolist()
    for (bx1, by1, bx2, by2), tid, c in zip(xyxy, ids, boxes.conf.tolist()):
        points.append((f, tid, round((bx1 + bx2) / 2, 1), round((by1 + by2) / 2, 1), round(c, 3)))
    counter.update(f, [(*b, tid) for b, tid in zip(xyxy, ids)])

    if writer:
        im = crop.copy()
        line_y = int(counter.line_y)
        cv2.line(im, (0, line_y), (im.shape[1], line_y), (0, 0, 255), 2)
        for (bx1, by1, bx2, by2), tid in zip(boxes.xyxy.int().tolist(), ids):
            col = (0, 255, 0) if tid in counter.counted else (255, 200, 0)
            cv2.rectangle(im, (bx1, by1), (bx2, by2), col, 2)
            cv2.putText(im, str(tid), (bx1 + 4, by1 + 22), 0, 0.7, col, 2)
        cv2.rectangle(im, (0, 0), (im.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(im, f"count {counter.count}   frame {f}", (10, 32), 0, 1.0, (255, 255, 255), 2)
        writer.write(im)

dt = time.time() - t0
cap.release()
if writer:
    writer.release()
with open(out / f"{stem}_events.csv", "w", newline="") as fh:
    csv.writer(fh).writerows([("frame", "time_s", "id", "x", "y", "kind"),
                              *((e[0], round(e[0] / fps, 2), *e[1:]) for e in counter.events)])
with open(out / f"{stem}_tracks.csv", "w", newline="") as fh:
    csv.writer(fh).writerows([("frame", "id", "cx", "cy", "conf"), *points])
print(f"{stem}: count={counter.count} (dups suppressed={sum(e[4] == 'dup' for e in counter.events)}) "
      f"frames={f} {f / dt:.1f} fps -> {out}")
