import torch
from transformers import AutoModelForCausalLM
import cv2
import numpy as np
from PIL import Image
import os
import time
video_path = 'drone4.mp4'
output_dir= 'aioutput'
cap = cv2.VideoCapture(video_path)

# Read first frame
ret, frame = cap.read()


# moondream = AutoModelForCausalLM.from_pretrained(
#     "moondream/moondream3-preview",
#     trust_remote_code=True,
#     dtype=torch.bfloat16,
#     device_map={"": "cuda"},
# )
# moondream.compile()
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map={"": "cuda"}  # ...or 'mps', on Apple Silicon
)



ai_box=[0,0,0,0]


start_time = time.time()
if not ret:
    raise RuntimeError("Cannot read video")
frame_idx = 0
while cap.isOpened():
    frame_idx += 1
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx%30==0:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detects = model.detect(image, "flying drone")
        objects = detects["objects"]
        print(f"detects: {detects}")

        h, w = frame.shape[:2]
        display = frame.copy()
        for obj in objects:
            x_min = int(obj["x_min"] * w)
            y_min = int(obj["y_min"] * h)
            x_max = int(obj["x_max"] * w)
            y_max = int(obj["y_max"] * h)
            print(f"{time.time() - start_time:.2f}s Bounding box (px): ({x_min}, {y_min}) -> ({x_max}, {y_max})")
            ai_box = [x_min, y_min, x_max - x_min, y_max - y_min]
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")

            cv2.imwrite(filename, display)
            print(f"Saved {filename}")
