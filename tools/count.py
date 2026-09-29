"""Count coconuts on the belt: YOLO26 detector + ByteTrack + a counting line (rules in counter.py).

Frames are cropped to the belt region exactly as in training (tools/prepare_dataset.py).
Writes an annotated video, count events (frame, time, id, x, y) and all track points as CSV.
For CPU/ONNX speed measurements use bench_cpu.py.
--realtime plays the video at its own FPS like a live camera and shows it in a window: frames that arrive
while the previous one is being processed are dropped, so the count shows how the pipeline would do live.
"""
import argparse
import csv
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import load_config
from counter import LineCounter

p = argparse.ArgumentParser()
p.add_argument("config", help="dataset config, e.g. configs/lv1.yaml")
c = load_config(p.parse_known_args()[0].config)
p.add_argument("--video", default=c.video)
p.add_argument("--weights", default=c.best)
p.add_argument("--crop", type=int, nargs=4, default=c.crop, metavar=("X1", "Y1", "X2", "Y2"))
p.add_argument("--imgsz", type=int, default=c.imgsz)
p.add_argument("--conf", type=float, default=0.1, help="low so ByteTrack's second-stage matching sees weak boxes")
p.add_argument("--tracker", default="bytetrack.yaml")
p.add_argument("--line", type=float, default=c.line, help="counting line, fraction of crop height")
p.add_argument("--mem", type=int, default=25, help="frames a count blocks re-counts of the same coconut")
p.add_argument("--gate", type=float, default=60, help="px in x within which a new track is the same coconut")
p.add_argument("--min-hits", type=int, default=3, help="detections a track needs before it can be counted")
p.add_argument("--out", default="inference_results")
p.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
p.add_argument("--no-video", action="store_true")
p.add_argument("--realtime", action="store_true",
               help="play the video at its FPS like a camera, drop frames not processed in time, show it live")
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


def read():
    """Yield (frame index, frame) until the video ends or --max-frames is reached."""
    f = 0
    while not (a.max_frames and f >= a.max_frames):
        ok, frame = cap.read()
        if not ok:
            return
        yield f, frame
        f += 1


def live():
    """Play the video at its FPS in a thread, like a camera, and yield only the newest frame: frames that
    arrive while the previous one is being processed are dropped."""
    latest, ended = None, False
    cond, stop = threading.Condition(), threading.Event()

    def play():
        nonlocal latest, ended
        start = time.perf_counter()
        for item in read():
            time.sleep(max(0.0, start + item[0] / fps - time.perf_counter()))
            if stop.is_set():
                return
            with cond:
                latest = item
                cond.notify()
        with cond:
            ended = True
            cond.notify()

    player = threading.Thread(target=play, daemon=True)
    player.start()
    try:
        while True:
            with cond:
                cond.wait_for(lambda: latest is not None or ended)
                item, latest = latest, None
            if item is None:
                return
            yield item
    finally:  # also on quit: the player must stop reading before cap is released
        stop.set()
        player.join()


if a.realtime:  # load the model now: the first call takes seconds, and the clock starts with playback
    model.predict(np.zeros((y2 - y1, x2 - x1, 3), np.uint8), imgsz=a.imgsz, verbose=False)
points = []  # (frame, id, cx, cy, conf)
t0 = time.time()
n = 0  # frames processed
f = -1
for f, frame in live() if a.realtime else read():
    n += 1
    crop = frame[y1:y2, x1:x2]
    r = model.track(crop, persist=True, tracker=a.tracker, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
    boxes = r.boxes
    ids = boxes.id.int().tolist() if boxes.id is not None else []
    xyxy = boxes.xyxy.tolist()
    for (bx1, by1, bx2, by2), tid, c in zip(xyxy, ids, boxes.conf.tolist()):
        points.append((f, tid, round((bx1 + bx2) / 2, 1), round((by1 + by2) / 2, 1), round(c, 3)))
    counter.update(f, [(*b, tid) for b, tid in zip(xyxy, ids)])

    if writer or a.realtime:
        im = crop.copy()
        line_y = int(counter.line_y)
        cv2.line(im, (0, line_y), (im.shape[1], line_y), (0, 0, 255), 2)
        for (bx1, by1, bx2, by2), tid in zip(boxes.xyxy.int().tolist(), ids):
            col = (0, 255, 0) if tid in counter.counted else (255, 200, 0)
            cv2.rectangle(im, (bx1, by1), (bx2, by2), col, 2)
            cv2.putText(im, str(tid), (bx1 + 4, by1 + 22), 0, 0.7, col, 2)
        cv2.rectangle(im, (0, 0), (im.shape[1], 44), (0, 0, 0), -1)
        text = f"count {counter.count}   frame {f}" + (f"   dropped {f + 1 - n}" if a.realtime else "")
        cv2.putText(im, text, (10, 32), 0, 1.0, (255, 255, 255), 2)
        if writer:
            writer.write(im)
        if a.realtime:
            cv2.imshow("coconut counting (q to quit)", im)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

dt = time.time() - t0
cap.release()
if writer:
    writer.release()
if a.realtime:
    cv2.destroyAllWindows()
with open(out / f"{stem}_events.csv", "w", newline="") as fh:
    csv.writer(fh).writerows([("frame", "time_s", "id", "x", "y", "kind"),
                              *((e[0], round(e[0] / fps, 2), *e[1:]) for e in counter.events)])
with open(out / f"{stem}_tracks.csv", "w", newline="") as fh:
    csv.writer(fh).writerows([("frame", "id", "cx", "cy", "conf"), *points])
print(f"{stem}: count={counter.count} (dups suppressed={sum(e[4] == 'dup' for e in counter.events)}) "
      f"frames={n}{f' of {f + 1} (dropped {f + 1 - n})' if a.realtime else ''} {n / dt:.1f} fps -> {out}")
