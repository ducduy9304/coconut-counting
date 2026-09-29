"""Line-crossing coconut counter shared by count.py (Ultralytics tracking) and bench_cpu.py (ONNX Runtime).

A track is counted once, the first time its center is below the line (belt moves down), provided it
was first seen above the line and has at least `min_hits` detections. This skips blips born at the
bottom edge, objects stuck below the line, and coconuts already past the line when the video starts.
- Coconuts that roll back above the line and come down again keep their counted flag.
- If the tracker loses a coconut and re-acquires it under a new ID, the new track is not counted
  again when it was born near the line (within `gate` px above it) and a track counted within `mem`
  frames at a nearby x (`gate` px) is no longer visible. Coconuts in a column share x, so a nearby
  count alone is not a duplicate.
"""


class LineCounter:
    def __init__(self, line_y, mem=25, gate=60, min_hits=3):
        self.line_y, self.mem, self.gate, self.min_hits = line_y, mem, gate, min_hits
        self.count = 0
        self.counted = set()  # track ids already counted (or judged to be a re-acquired, counted coconut)
        self.born_y = {}  # track id -> center y when first seen
        self.hits = {}  # track id -> number of detections
        self.events = []  # (frame, id, x, y, kind)

    def update(self, frame, tracks):
        """tracks: iterable of (x1, y1, x2, y2, track_id) in crop pixels for this frame."""
        tracks = list(tracks)
        ids = {int(t[4]) for t in tracks}
        for x1, y1, x2, y2, tid in tracks:
            tid = int(tid)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            self.born_y.setdefault(tid, cy)
            self.hits[tid] = self.hits.get(tid, 0) + 1
            if (tid in self.counted or cy < self.line_y or self.born_y[tid] >= self.line_y
                    or self.hits[tid] < self.min_hits):
                continue
            self.counted.add(tid)
            dup = self.born_y[tid] > self.line_y - self.gate and any(
                frame - e[0] <= self.mem and e[4] == "count" and e[1] not in ids and abs(e[2] - cx) < self.gate
                for e in self.events)
            if not dup:
                self.count += 1
            self.events.append((frame, tid, round(cx, 1), round(cy, 1), "dup" if dup else "count"))
