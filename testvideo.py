import cv2
import subprocess
import os

input_path = "drone.mp4"
temp_path = "drone_framed_temp.mp4"
output_path = "drone_framed.mp4"

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    text = f"Frame: {frame_num} / {total_frames}"
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    out.write(frame)
    frame_num += 1

cap.release()
out.release()
print(f"Processed {frame_num} frames")

# Re-encode to H.264 for VSCode compatibility
subprocess.run(
    ["ffmpeg", "-y", "-i", temp_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", output_path],
    check=True,
)
os.remove(temp_path)
print(f"Saved to {output_path}")
