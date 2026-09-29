# coconut-counting

Count coconuts on a conveyor belt with a YOLO26 detector, ByteTrack and a counting line.

Each video has its own config in `configs/` (`lv1.yaml`, `lv2.yaml`) holding its crop, validation split, model input size and counting line. Every script takes the config as its first argument.

## Setup

```bash
uv sync
```

Videos are not in the repo because they exceed GitHub's file size limit. Copy them into `CoconutVideos/` (`lv1.mp4`, `lv2.avi`).

## Run

The examples use `lv2`. For the other video, pass `configs/lv1.yaml` instead.

```bash
# 1. Build the training set from the Roboflow export (crop, clean labels, time-based train/valid split)
uv run tools/prepare_dataset.py configs/lv2.yaml

# 2. Train (GPU) -> train_results/lv2_yolo26n_480/
uv run tools/train.py configs/lv2.yaml

# 3. Export the trained model to ONNX for CPU inference
uv run tools/export_onnx.py configs/lv2.yaml

# 4. Count on CPU only, and report FPS
uv run tools/bench_cpu.py configs/lv2.yaml --track | tee inference_results/lv2_bench.txt
```

Step 4 also writes the annotated video to `inference_results/lv2_cpu.mp4`. Change the path with `--save-video PATH`, or skip it with `--no-video` for a clean FPS figure.

Step 4 runs decode, inference and tracking in separate threads, so the FPS it reports is the real wall-clock throughput. Writing the video is slower than the rest of the pipeline and caps that FPS, so measure speed with `--no-video`. Add `--sequential` to run the stages one after another and see the time each one takes. Set the number of ONNX Runtime threads with `--threads N`; the number of physical cores is usually the fastest.

## Realtime

`tools/count.py` counts with the trained `.pt` model through Ultralytics. It writes the annotated video `inference_results/lv2_count.mp4` and the count events and track points as CSV next to it. By default it is offline: every frame of the video is processed, as fast as the machine allows.

To see how the counter would do on a live camera, add `--realtime`:

```bash
uv run tools/count.py configs/lv2.yaml --realtime
```

The video then plays at its own FPS (at 25 fps, one frame every 40 ms) and the counter always takes the newest frame. Frames that arrive while the previous one is still being processed are dropped, as with a real camera. The annotated view is shown in a window (press `q` to quit), and the number of dropped frames is shown on it and printed at the end. With 0 dropped frames the machine keeps up and the count matches the offline run. With dropped frames the tracker sees bigger jumps between frames and the count can differ. The window needs a display, so run it on a machine with a screen. Add `--no-video` to skip writing the annotated video, which also slows the loop.
