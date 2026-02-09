import torch
from transformers import AutoModelForCausalLM
import cv2
import numpy as np
from PIL import Image
import os
import time
import subprocess
import argparse

OBJECT_OF_INTEREST = "drone"

class DroneTracker:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-06-21",
            trust_remote_code=True,
            device_map={"": "cuda"},
        )
        self.feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.box = None          # [x, y, w, h]
        self.points = None       # (N, 2) tracked points
        self.old_gray = None     # previous grayscale frame
        self._frame_count = 0    # frames since last re-detect

    def detect(self, frame):
        """Use moondream to detect the drone and reinitialize tracking points."""
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        objects = self.model.detect(image, OBJECT_OF_INTEREST)["objects"]
        if not objects:
            return False

        h, w = frame.shape[:2]
        obj = objects[0] # ignore multiple detections for now
        x_min = int(obj["x_min"] * w)
        y_min = int(obj["y_min"] * h)
        x_max = int(obj["x_max"] * w)
        y_max = int(obj["y_max"] * h)
        self.box = [x_min, y_min, x_max - x_min, y_max - y_min]

        self._init_points(frame)
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frame_count = 0
        return True

    def _init_points(self, frame):
        """Detect good features inside the current box."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.box
        mask_roi = np.zeros_like(gray)
        mask_roi[y:y+h, x:x+w] = 255
        pts = cv2.goodFeaturesToTrack(gray, mask=mask_roi, **self.feature_params)
        if pts is not None:
            self.points = pts.reshape(-1, 2)
        else:
            self.points = None

    def track(self, frame):
        """Track the drone in a new frame using optical flow. Returns the box or None."""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.old_gray is None or self.points is None or len(self.points) == 0:
            # No tracking state yet — need a detect() call first
            return self.box

        self._frame_count += 1
        p0 = self.points.reshape(-1, 1, 2).astype(np.float32)

        # Optical flow
        p1, status, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, p0, None, **self.lk_params)

        good_new = p1[status.flatten() == 1].reshape(-1, 2)
        if len(good_new) == 0:
            self.old_gray = frame_gray
            return self.box

        # Filter outliers beyond 2 std from mean
        avg = np.mean(good_new, axis=0)
        std = np.std(good_new, axis=0)
        if np.all(std > 0):
            within = np.all(np.abs(good_new - avg) <= std * 2, axis=1)
            good_new = good_new[within]

        if len(good_new) == 0:
            self.old_gray = frame_gray
            return self.box

        # Update bounding box from points
        bx = int(np.min(good_new[:, 0]))
        by = int(np.min(good_new[:, 1]))
        bx2 = int(np.max(good_new[:, 0]))
        by2 = int(np.max(good_new[:, 1]))
        self.box = [bx, by, bx2 - bx, by2 - by]

        # Re-detect features around box edges every 10 frames
        # if self._frame_count % 10 == 0:
        #     good_new = self._refresh_points(frame_gray, good_new)

        self.points = good_new
        self.old_gray = frame_gray
        return self.box

    # Dont end up using this 
    def _refresh_points(self, frame_gray, good_new):
        """Re-detect features around the bounding box edges and merge with existing."""
        x, y, w, h = self.box
        pad = int(((w + h) / 2) * 0.1)
        edge_mask = np.zeros_like(frame_gray)
        ox1 = max(0, x - pad)
        oy1 = max(0, y - pad)
        ox2 = min(frame_gray.shape[1], x + w + pad)
        oy2 = min(frame_gray.shape[0], y + h + pad)
        edge_mask[oy1:oy2, ox1:ox2] = 255
        # Cut out interior
        ix1 = min(ox2, x + pad)
        iy1 = min(oy2, y + pad)
        ix2 = max(ox1, x + w - pad)
        iy2 = max(oy1, y + h - pad)
        if ix1 < ix2 and iy1 < iy2:
            edge_mask[iy1:iy2, ix1:ix2] = 0

        new_pts = cv2.goodFeaturesToTrack(frame_gray, mask=edge_mask, **self.feature_params)
        if new_pts is not None:
            new_pts = new_pts.reshape(-1, 2)
            keep = []
            for pt in new_pts:
                dists = np.linalg.norm(good_new - pt, axis=1)
                if np.min(dists) > self.feature_params['minDistance']:
                    keep.append(pt)
            if keep:
                good_new = np.vstack([good_new, np.array(keep)])
        return good_new

    def draw(self, frame):
        """Return a copy of frame with bounding box and tracked points drawn."""
        display = frame.copy()
        if self.points is not None:
            for pt in self.points:
                cv2.circle(display, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        if self.box is not None:
            x, y, w, h = self.box
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return display

    def imgwrite(self, frame, filename):
        """Save a debug image showing the bounding box and tracked points."""
        display = self.draw(frame)
        cv2.imwrite(filename, display)


# --- Main loop ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track a drone in a video using AI + optical flow")
    parser.add_argument("source", help="Path to input video file")
    parser.add_argument("dest", help="Path to output video file (h264 mp4)")
    args = parser.parse_args()

    video_path = args.source
    dest_video_h264 = args.dest
    output_dir = os.path.dirname(dest_video_h264) or '.'
    dest_video = os.path.join(output_dir, '_temp_tracking.mp4')
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read video")

    tracker = DroneTracker()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter(dest_video, fourcc, fps, (w, h))

    start_time = time.time()
    print("Running initial AI detection...")
    if not tracker.detect(frame):
        raise RuntimeError("AI could not detect drone in first frame")
    print(f"Initial box: {tracker.box}")
    video_out.write(tracker.draw(frame))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1



        # Re-detect with AI every 30 frames to correct drift
        # if frame_idx % 10 == 0:
        if frame_idx/frame_rate>time.time()-start_time:  # also ensure at least 5 seconds between AI calls
            print(f"{time.time() - start_time:.2f}s Re-detecting with AI at frame {frame_idx}...")
            tracker.detect(frame)
            print(f"  AI box: {tracker.box}")
        box = tracker.track(frame)
        video_out.write(tracker.draw(frame))

    cap.release()
    video_out.release()
    # this is just to allow me to view the video in vscode. 
    subprocess.run(
        ["ffmpeg", "-y", "-i", dest_video, "-vcodec", "libx264", "-pix_fmt", "yuv420p", dest_video_h264],
        check=True,
    )
    os.remove(dest_video)
    print(f"Done — saved to {dest_video_h264}")
