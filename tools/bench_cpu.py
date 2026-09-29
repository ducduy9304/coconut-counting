"""Benchmark the coconut counting pipeline on CPU only, with ONNX Runtime.

GPU use is ruled out: CUDA devices are hidden and ONNX Runtime gets only CPUExecutionProvider.
By default the stages run as a pipeline, one thread each: decode + preprocess | inference + NMS |
tracking + counting, so a frame is tracked while the next one is inferred; the reported FPS is the
measured wall-clock throughput. --sequential runs the stages one after another instead and times each
(video decode, preprocess, inference, postprocess, tracking + counting).

Video, crop, counting line and ONNX model come from the dataset config.
Detector only (needs: onnxruntime, opencv-python, numpy, pyyaml):
    python tools/bench_cpu.py configs/lv1.yaml
Full pipeline with ByteTrack + counting (also needs: ultralytics, lap, CPU torch):
    python tools/bench_cpu.py configs/lv1.yaml --track
The annotated full frame (ROI, counting line, boxes/IDs, count, live CPU FPS) is written to
inference_results/<name>_cpu.mp4 by default; change it with --save-video PATH, turn it off with --no-video.
Drawing and video writing are excluded from the FPS drawn on the video (pipeline mode: the throughput
the slowest stage sustains). In pipeline mode they run in the main thread and, being slower than the
stages, cap the measured wall-clock FPS printed at the end, so use --no-video for a clean figure.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before any library can grab a GPU

import argparse
import collections
import platform
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from config import load_config

p = argparse.ArgumentParser()
p.add_argument("config", help="dataset config, e.g. configs/lv1.yaml")
c = load_config(p.parse_known_args()[0].config)
p.add_argument("--video", default=c.video)
p.add_argument("--onnx", default=c.onnx)
p.add_argument("--crop", type=int, nargs=4, default=c.crop, metavar=("X1", "Y1", "X2", "Y2"))
p.add_argument("--conf", type=float, default=0.1, help="low so ByteTrack's second-stage matching sees weak boxes")
p.add_argument("--iou", type=float, default=0.7, help="NMS IoU for the one-to-many head (Ultralytics default)")
p.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads; 0 = ORT default")
p.add_argument("--warmup", type=int, default=20, help="frames excluded from timing")
p.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
p.add_argument("--track", action="store_true", help="add ByteTrack + line counting (needs ultralytics)")
p.add_argument("--tracker", default="bytetrack.yaml")
p.add_argument("--line", type=float, default=c.line, help="counting line, fraction of crop height")
p.add_argument("--target-fps", type=float, default=25)
p.add_argument("--save-video", default=f"inference_results/{c.name}_cpu.mp4",
               help="write the annotated full frame video here")
p.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
p.add_argument("--sequential", action="store_true", help="run stages one after another and time each")
a = p.parse_args()

so = ort.SessionOptions()
so.intra_op_num_threads = a.threads
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
assert sess.get_providers() == ["CPUExecutionProvider"], sess.get_providers()
inp = sess.get_inputs()[0]
H, W = inp.shape[2:]
end2end = sess.get_modelmeta().custom_metadata_map.get("end2end") == "True"  # else raw (1, 4 + nc, anchors) + NMS

x1, y1, x2, y2 = a.crop
ch, cw = y2 - y1, x2 - x1
r = min(H / ch, W / cw)  # letterbox exactly like Ultralytics: keep aspect, center, pad 114
nh, nw = round(ch * r), round(cw * r)
top, left = int(round((H - nh) / 2 - 0.1)), int(round((W - nw) / 2 - 0.1))
bottom, right = H - nh - top, W - nw - left

if a.track:
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    from counter import LineCounter

    class Dets:
        """Minimal NumPy stand-in for Ultralytics Boxes: what BYTETracker reads."""

        def __init__(self, xyxy, conf, cls):
            self.xyxy, self.conf, self.cls = xyxy, conf, cls
            self.xywh = np.c_[(xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]]

        def __len__(self):
            return len(self.conf)

        def __getitem__(self, m):
            return Dets(self.xyxy[m], self.conf[m], self.cls[m])

    tracker = BYTETracker(IterableSimpleNamespace(**YAML.load(check_yaml(a.tracker))))
    counter = LineCounter(a.line * ch)

cap = cv2.VideoCapture(a.video)
writer = None
if not a.no_video:
    Path(a.save_video).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(a.save_video, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


def read():
    """Yield (frame index, frame) until the video ends or --max-frames is reached."""
    f = 0
    while not (a.max_frames and f >= a.max_frames):
        ok, frame = cap.read()
        if not ok:
            return
        yield f, frame
        f += 1


def preprocess(frame):
    """Crop to the belt, letterbox to the model input, HWC BGR uint8 -> NCHW RGB float."""
    im = cv2.resize(frame[y1:y2, x1:x2], (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255


def postprocess(det):
    """Raw model output -> (N, 6) x1, y1, x2, y2, score, class in crop pixels."""
    if end2end:  # (300, 6): x1, y1, x2, y2, score, class
        det = det[det[:, 4] >= a.conf]
    else:  # (4 + nc, anchors): cx, cy, w, h, class scores
        det = det.T
        score, cls = det[:, 4:].max(1), det[:, 4:].argmax(1)
        m = score >= a.conf
        xywh, score, cls = det[m, :4], score[m], cls[m]
        tlwh = np.c_[xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, 2:]]
        keep = np.array(cv2.dnn.NMSBoxesBatched(tlwh, score, cls, a.conf, a.iou), int).reshape(-1)
        det = np.c_[tlwh[keep, :2], tlwh[keep, :2] + tlwh[keep, 2:], score[keep], cls[keep]]
    det[:, :4] = np.clip((det[:, :4] - [left, top, left, top]) / r, 0, [cw, ch, cw, ch])
    return det


def track(f, det):
    """Update tracker and counter. Returns what draw() needs for this frame, snapshotted so the pipeline's
    drawing thread is not ahead of or behind it: boxes, track ids (None without --track), per-box counted
    flags and the running count."""
    if not a.track:
        return det[:, :4], None, None, None
    tr = tracker.update(Dets(det[:, :4], det[:, 4], det[:, 5]))
    counter.update(f, tr[:, :5] if len(tr) else [])
    ids = tr[:, 4].astype(int) if len(tr) else np.zeros(0, int)
    return tr[:, :4] if len(tr) else np.zeros((0, 4)), ids, [i in counter.counted for i in ids], counter.count


def draw(frame, boxes, ids, counted, count, fps):
    """Annotate the full frame in place: ROI box, counting line, boxes (+ track IDs), count and FPS."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
    if a.track:
        ly = y1 + int(counter.line_y)
        cv2.line(frame, (x1, ly), (x2, ly), (0, 0, 255), 3)
    for i, (bx1, by1, bx2, by2) in enumerate(boxes.astype(int)):
        col = (0, 255, 0) if a.track and counted[i] else (255, 200, 0)
        cv2.rectangle(frame, (x1 + bx1, y1 + by1), (x1 + bx2, y1 + by2), col, 2)
        if ids is not None:
            cv2.putText(frame, str(ids[i]), (x1 + bx1 + 4, y1 + by1 + 22), 0, 0.7, col, 2)
    lines = ([f"count: {count}"] if a.track else []) + [f"FPS (CPU): {fps:.1f}"]
    w = max(cv2.getTextSize(t, 0, 1.4, 3)[0][0] for t in lines)
    cv2.rectangle(frame, (0, 0), (w + 24, 16 + 52 * len(lines)), (0, 0, 0), -1)  # left panel, clear of the ROI
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (12, 52 + 52 * i), 0, 1.4, (255, 255, 255), 3)


busy = []  # per pipeline stage: busy ms of its last 25 frames, for the video's FPS overlay


def stage(fn, src, src_is_work=False):
    """Run fn over the items of src in a worker thread; return a queue of results ending with None.
    Records the stage's busy time per frame: fn, plus fetching the item when src_is_work (video decode)
    rather than waiting on the previous stage."""
    out = queue.Queue(maxsize=8)
    ms = collections.deque(maxlen=25)
    busy.append(ms)

    def run():
        try:
            it = iter(src)
            while True:
                t0 = time.perf_counter()
                item = next(it, None)
                if item is None:
                    break
                if not src_is_work:
                    t0 = time.perf_counter()
                res = fn(*item)
                ms.append((time.perf_counter() - t0) * 1000)
                out.put(res)
        finally:
            out.put(None)

    threading.Thread(target=run, daemon=True).start()
    return out


def drain(q):
    while (item := q.get()) is not None:
        yield item


stages = ["decode", "preprocess", "inference", "postprocess"] + (["track+count"] if a.track else [])
times = {s: [] for s in stages}
recent = []  # last frames' processing time (ms) for the live FPS overlay
n_frames = 0
if a.sequential:
    frames = read()
    while True:
        t0 = time.perf_counter()
        item = next(frames, None)
        if item is None:
            break
        f, frame = item
        t1 = time.perf_counter()
        blob = preprocess(frame)
        t2 = time.perf_counter()
        raw = sess.run(None, {inp.name: blob})[0][0]
        t3 = time.perf_counter()
        det = postprocess(raw)
        t4 = time.perf_counter()
        drawn = track(f, det)
        t5 = time.perf_counter()
        n_frames += 1
        if f >= a.warmup:
            for s, dt in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                times[s].append(dt * 1000)
        if writer:
            recent = (recent + [(t5 - t0) * 1000])[-25:]
            draw(frame, *drawn, 1000 / np.mean(recent))
            writer.write(frame)
else:
    decoded = stage(lambda f, frame: (f, frame, preprocess(frame)), read(), src_is_work=True)
    detected = stage(lambda f, frame, blob: (f, frame, postprocess(sess.run(None, {inp.name: blob})[0][0])),
                     drain(decoded))
    tracked = stage(lambda f, frame, det: (f, frame, track(f, det)), drain(detected))
    done = []  # wall-clock time each frame left the pipeline
    for f, frame, drawn in drain(tracked):
        done.append(time.perf_counter())
        n_frames += 1
        if writer:  # overlay: throughput the stages sustain (slowest one), excluding drawing and writing
            draw(frame, *drawn, 1000 / max(np.mean(ms) for ms in busy))
            writer.write(frame)
cap.release()
if writer:
    writer.release()

print(f"CPU: {platform.processor() or platform.machine()} | logical cores {os.cpu_count()} | "
      f"onnxruntime {ort.__version__} {sess.get_providers()} | intra-op threads {a.threads or 'default'}")
if a.sequential:
    n = len(times["decode"])
    print(f"model: {Path(a.onnx).name} input {H}x{W} {'end2end' if end2end else '+ NMS'} | video: {Path(a.video).name} "
          f"| timed frames {n} (after {a.warmup} warmup)\n")
    print(f"{'stage':<12} {'mean ms':>8} {'p95 ms':>8}")
    for s in stages:
        v = np.array(times[s])
        print(f"{s:<12} {v.mean():8.2f} {np.percentile(v, 95):8.2f}")
    total = sum(np.mean(times[s]) for s in stages)
    fps = 1000 / total
    print(f"{'total':<12} {total:8.2f}\n")
    print(f"sequential FPS (1 thread end to end): {fps:6.1f}  {'OK' if fps >= a.target_fps else 'BELOW'} "
          f"target {a.target_fps:g}")
else:
    timed = done[a.warmup:]
    fps = (len(timed) - 1) / (timed[-1] - timed[0]) if len(timed) > 1 else float("nan")
    print(f"model: {Path(a.onnx).name} input {H}x{W} {'end2end' if end2end else '+ NMS'} | video: {Path(a.video).name} "
          f"| timed frames {len(timed)} (after {a.warmup} warmup)\n")
    print(f"pipelined FPS (decode | infer | track{'+count' if a.track else ''} threads, wall clock"
          f"{', incl. video writing' if writer else ''}): {fps:6.1f}  {'OK' if fps >= a.target_fps else 'BELOW'} "
          f"target {a.target_fps:g}")
if a.track:
    print(f"\ncount = {counter.count} (whole run incl. warmup frames: {n_frames} frames)")
if writer:
    print(f"video -> {a.save_video}")
