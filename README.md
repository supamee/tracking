# Drone Tracker

Track a drone in video using AI object detection ([Moondream2](https://huggingface.co/vikhyatk/moondream2)) combined with Lucas-Kanade optical flow for frame-to-frame tracking.

## How it works

1. **AI Detection** -- Moondream2 detects the drone in the first frame and produces a bounding box.
2. **Optical Flow Tracking** -- Good features (Shi-Tomasi corners) are detected inside the bounding box. Lucas-Kanade optical flow tracks these points frame-to-frame, with outlier filtering (2 std).
3. **Periodic Re-detection** -- When processing runs ahead of real-time, the AI re-detects the drone to correct any drift accumulated by optical flow.

The output is an H.264 MP4 video with the bounding box and tracked feature points drawn on each frame.

## Requirements

- Python 3.10+
- CUDA-capable GPU
- ffmpeg (for H.264 encoding)

## Setup

```bash
uv sync
```

## Usage

```bash
uv run combined.py <source_video> <output_video>
```

Example:

```bash
uv run combined.py drone.mp4 output/tracked.mp4
```

<video src="output/tracked.mp4" autoplay loop muted></video>

Current limitations:
Does not support multiple drones.

Tested on a racing drone video and it sturggles the occlusion, which is to be expected.

No velocity prediction or kalman filter. Redetection generally happened fast enough that that worked better.
