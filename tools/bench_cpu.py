"""Benchmark the coconut counting pipeline on CPU only, with ONNX Runtime.

GPU use is ruled out: CUDA devices are hidden and ONNX Runtime gets only CPUExecutionProvider.
Times each stage per frame (video decode, preprocess, inference, postprocess, tracking + counting)
and reports sequential FPS plus the FPS a decode/inference/tracking thread split could reach
(bounded by the slowest stage).

Detector only (needs: onnxruntime, opencv-python, numpy):
    python tools/bench_cpu.py CoconutVideos/lv1.mp4
Full pipeline with ByteTrack + counting (also needs: ultralytics, lap, CPU torch):
    python tools/bench_cpu.py CoconutVideos/lv1.mp4 --track
Also write the annotated full frame (ROI, counting line, boxes/IDs, count, live CPU FPS):
    python tools/bench_cpu.py CoconutVideos/lv1.mp4 --track --save-video inference_results/lv1_cpu.mp4
Drawing and video writing are excluded from all timings and FPS figures.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before any library can grab a GPU

import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

p = argparse.ArgumentParser()
p.add_argument("video", nargs="?", default="CoconutVideos/lv1.mp4")
p.add_argument("--onnx", default="train_results/lv1_yolo26s_480/weights/coconut_lv1_yolo26s_480x288.onnx")
p.add_argument("--crop", type=int, nargs=4, default=[620, 0, 1260, 1080], metavar=("X1", "Y1", "X2", "Y2"))
p.add_argument("--conf", type=float, default=0.1, help="low so ByteTrack's second-stage matching sees weak boxes")
p.add_argument("--iou", type=float, default=0.7, help="NMS IoU for the one-to-many head (Ultralytics default)")
p.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads; 0 = ORT default")
p.add_argument("--warmup", type=int, default=20, help="frames excluded from timing")
p.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
p.add_argument("--track", action="store_true", help="add ByteTrack + line counting (needs ultralytics)")
p.add_argument("--tracker", default="bytetrack.yaml")
p.add_argument("--line", type=float, default=0.8, help="counting line, fraction of crop height")
p.add_argument("--target-fps", type=float, default=25)
p.add_argument("--save-video", default="", help="write the annotated full frame video here; '' = off")
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

stages = ["decode", "preprocess", "inference", "postprocess"] + (["track+count"] if a.track else [])
times = {s: [] for s in stages}
cap = cv2.VideoCapture(a.video)
writer = None
if a.save_video:
    Path(a.save_video).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(a.save_video, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
recent = []  # last frames' processing time (ms) for the live FPS overlay


def draw(frame, boxes, ids, fps):
    """Annotate the full frame in place: ROI box, counting line, boxes (+ track IDs), count and FPS."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
    if a.track:
        ly = y1 + int(counter.line_y)
        cv2.line(frame, (x1, ly), (x2, ly), (0, 0, 255), 3)
    for i, (bx1, by1, bx2, by2) in enumerate(boxes.astype(int)):
        tid = ids[i] if ids is not None else None
        col = (0, 255, 0) if tid in (counter.counted if a.track else ()) else (255, 200, 0)
        cv2.rectangle(frame, (x1 + bx1, y1 + by1), (x1 + bx2, y1 + by2), col, 2)
        if tid is not None:
            cv2.putText(frame, str(tid), (x1 + bx1 + 4, y1 + by1 + 22), 0, 0.7, col, 2)
    lines = ([f"count: {counter.count}"] if a.track else []) + [f"FPS (CPU): {fps:.1f}"]
    w = max(cv2.getTextSize(t, 0, 1.4, 3)[0][0] for t in lines)
    cv2.rectangle(frame, (0, 0), (w + 24, 16 + 52 * len(lines)), (0, 0, 0), -1)  # left panel, clear of the ROI
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (12, 52 + 52 * i), 0, 1.4, (255, 255, 255), 3)


f = -1
while True:
    t0 = time.perf_counter()
    ok, frame = cap.read()
    f += 1
    if not ok or (a.max_frames and f >= a.max_frames):
        break
    t1 = time.perf_counter()
    crop = frame[y1:y2, x1:x2]
    im = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    blob = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    t2 = time.perf_counter()
    det = sess.run(None, {inp.name: blob})[0][0]
    t3 = time.perf_counter()
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
    xyxy = (det[:, :4] - [left, top, left, top]) / r
    xyxy = np.clip(xyxy, 0, [cw, ch, cw, ch])
    t4 = time.perf_counter()
    if a.track:
        tr = tracker.update(Dets(xyxy, det[:, 4], det[:, 5]))
        counter.update(f, tr[:, :5] if len(tr) else [])
    t5 = time.perf_counter()
    if f >= a.warmup:
        for s, dt in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
            times[s].append(dt * 1000)
    if writer:
        recent = (recent + [(t5 - t0) * 1000])[-25:]
        if a.track:
            draw(frame, tr[:, :4] if len(tr) else np.zeros((0, 4)), tr[:, 4].astype(int) if len(tr) else None,
                 1000 / np.mean(recent))
        else:
            draw(frame, xyxy, None, 1000 / np.mean(recent))
        writer.write(frame)
cap.release()
if writer:
    writer.release()

n = len(times["decode"])
print(f"CPU: {platform.processor() or platform.machine()} | logical cores {os.cpu_count()} | "
      f"onnxruntime {ort.__version__} {sess.get_providers()} | intra-op threads {a.threads or 'default'}")
print(f"model: {Path(a.onnx).name} input {H}x{W} {'end2end' if end2end else '+ NMS'} | video: {Path(a.video).name} | timed frames {n} "
      f"(after {a.warmup} warmup)\n")
print(f"{'stage':<12} {'mean ms':>8} {'p95 ms':>8}")
for s in stages:
    v = np.array(times[s])
    print(f"{s:<12} {v.mean():8.2f} {np.percentile(v, 95):8.2f}")
total = sum(np.mean(times[s]) for s in stages)
seq_fps = 1000 / total
pipe_fps = 1000 / max(np.mean(times["decode"]), np.mean(times["inference"]),
                      sum(np.mean(times[s]) for s in stages if s not in ("decode", "inference")))
print(f"{'total':<12} {total:8.2f}\n")
print(f"sequential FPS (1 thread end to end): {seq_fps:6.1f}  "
      f"{'OK' if seq_fps >= a.target_fps else 'BELOW'} target {a.target_fps:g}")
print(f"pipelined FPS bound (decode | infer | rest in separate threads): {pipe_fps:6.1f}")
if a.track:
    print(f"\ncount = {counter.count} (whole run incl. warmup frames: {f} frames)")
if writer:
    print(f"video -> {a.save_video}")
