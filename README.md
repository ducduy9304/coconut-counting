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
