"""Turn a Roboflow YOLO export (polygon labels) into a clean, time-split training set.

- Pools every image from the export and ignores Roboflow's random split and data.yaml
  (adjacent frames in train/valid leak near-duplicates).
- Drops junk labels: lines with < 3 points (stray 1px boxes), polygons smaller than --min-size px,
  and duplicate polygons drawn twice over the same coconut.
- Crops to the belt region (same crop the detector sees at inference) and shifts polygons into it.
- Splits train/valid by time: frame index >= --val-start goes to valid.
Polygons are kept, so the set trains both detect (boxes derived from polygons) and -seg models.
"""
import argparse
import re
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("zip", nargs="?", default="data/coconut.v1i.yolo26.zip")
p.add_argument("--out", default="data/coconut_lv1")
p.add_argument("--crop", type=int, nargs=4, default=[620, 0, 1260, 1080], metavar=("X1", "Y1", "X2", "Y2"))
p.add_argument("--val-start", type=int, default=3640, help="frames with index >= this go to valid")
p.add_argument("--min-size", type=float, default=20, help="drop polygons whose bbox max side (px) is below this")
p.add_argument("--dup-iou", type=float, default=0.7, help="drop a polygon whose bbox overlaps an earlier one above this IoU")
p.add_argument("--overlay", default="data/review/overlay_coconut_lv1", help="write labeled previews; '' = skip")
a = p.parse_args()

def iou(p, q):
    w = max(0, min(p[2], q[2]) - max(p[0], q[0]))
    h = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = w * h
    return inter / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter)


x1, y1, x2, y2 = a.crop
cw, ch = x2 - x1, y2 - y1
out = Path(a.out)
shutil.rmtree(out, ignore_errors=True)
for split in ("train", "valid"):
    (out / split / "images").mkdir(parents=True)
    (out / split / "labels").mkdir(parents=True)

zf = zipfile.ZipFile(a.zip)
images = sorted(n for n in zf.namelist() if "/images/" in n)
stats = {"train": [0, 0], "valid": [0, 0]}
dropped = []
for name in images:
    stem = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", Path(name).stem)  # lv1_00018_jpg.rf.<hash> -> lv1_00018
    idx = int(re.search(r"_(\d+)$", stem).group(1))
    split = "valid" if idx >= a.val_start else "train"

    im = cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_COLOR)
    H, W = im.shape[:2]
    lab = name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    lines, kept = [], []
    for i, line in enumerate(zf.read(lab).decode().splitlines()):
        v = line.split()
        if not v:
            continue
        pts = np.array(v[1:], float).reshape(-1, 2) * [W, H]
        bw, bh = np.ptp(pts, axis=0)
        if len(pts) < 3 or max(bw, bh) < a.min_size:
            dropped.append((stem, i, len(pts), round(bw), round(bh)))
            continue
        box = np.r_[pts.min(0), pts.max(0)]
        if any(iou(box, b) > a.dup_iou for b in kept):
            dropped.append((stem, i, "duplicate"))
            continue
        kept.append(box)
        if pts[:, 0].min() < x1 - 1 or pts[:, 0].max() > x2 + 1 or pts[:, 1].min() < y1 - 1 or pts[:, 1].max() > y2 + 1:
            raise SystemExit(f"{stem} line {i}: polygon leaves crop {a.crop}; widen --crop")
        rel = np.clip((pts - [x1, y1]) / [cw, ch], 0, 1)
        lines.append(v[0] + " " + " ".join(f"{c:.6f}" for c in rel.ravel()) + "\n")

    cv2.imwrite(str(out / split / "images" / f"{stem}.jpg"), im[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
    (out / split / "labels" / f"{stem}.txt").write_text("".join(lines))
    stats[split][0] += 1
    stats[split][1] += len(lines)

(out / "data.yaml").write_text(
    f"path: {out.resolve()}\ntrain: train/images\nval: valid/images\nnc: 1\nnames: ['coconut']\n")
for s, (n, k) in stats.items():
    print(f"{s}: {n} images, {k} polygons")
print(f"dropped {len(dropped)} labels (stem, line, points, w, h):")
for d in dropped:
    print("  ", d)

if a.overlay:
    ov = Path(a.overlay)
    shutil.rmtree(ov, ignore_errors=True)
    ov.mkdir(parents=True)
    for img_path in sorted(out.rglob("images/*.jpg")):
        im = cv2.imread(str(img_path))
        H, W = im.shape[:2]
        lines = (img_path.parent.parent / "labels" / f"{img_path.stem}.txt").read_text().splitlines()
        for line in lines:
            pts = (np.array(line.split()[1:], float).reshape(-1, 2) * [W, H]).astype(np.int32)
            cv2.polylines(im, [pts], True, (0, 255, 0), 2)
        split = img_path.parent.parent.name
        cv2.putText(im, f"{split} {img_path.stem} n={len(lines)}", (8, 28), 0, 0.8, (0, 255, 255), 2)
        cv2.imwrite(str(ov / f"{split}_{img_path.stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
