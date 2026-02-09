import cv2
import numpy as np
import os

video_path = 'drone.mp4'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
start_box = (200, 5, 200, 130)  # (x, y, width, height)

# Read first frame
ret, old_frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read video")

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect good features to track inside the start box
x, y, w, h = start_box
mask_roi = np.zeros_like(old_gray)
mask_roi[y:y+h, x:x+w] = 255

feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask_roi, **feature_params)

if p0 is None:
    raise RuntimeError("No features found in start box")

# Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Save frame 0 with start box
cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite(os.path.join(output_dir, "frame_0000.jpg"), old_frame)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("Processing frame", frame_idx)

    # Calculate optical flow
    p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Keep only good points
    mask = status.flatten() == 1
    good_new = p1[mask].reshape(-1, 2)
    good_old = p0[mask].reshape(-1, 2)
    # Compute average position and standard deviation of good_new points
    avg_pos = np.mean(good_new, axis=0)
    std_pos = np.std(good_new, axis=0)
    # print(f"Average position: {avg_pos}, Standard deviation: {std_pos}")
    # Discard points not within 1 standard deviation of the average
    within_std = np.all(np.abs(good_new - avg_pos) <= std_pos*2, axis=1)
    good_new = good_new[within_std]
    good_old = good_old[within_std]
    print(f"Tracking {len(good_new)} points after filtering")

    if len(good_new) == 0:
        print(f"Lost tracking at frame {frame_idx}")
        break

    # Compute bounding box from tracked points
    bx = int(np.min(good_new[:, 0]))
    by = int(np.min(good_new[:, 1]))
    bx2 = int(np.max(good_new[:, 0]))
    by2 = int(np.max(good_new[:, 1]))

    # Bounding box is the min/max of the tracked points
    new_x = bx
    new_y = by
    w = bx2 - bx
    h = by2 - by

    # Save every 10 frames
    if frame_idx % 10 == 0:
        display = frame.copy()
        for pt in good_new:
            cv2.circle(display, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        cv2.rectangle(display, (new_x, new_y), (new_x + w, new_y + h), (0, 255, 0), 2)
        cv2.putText(display, f"Frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(filename, display)
        print(f"Saved {filename}")

    # Re-detect features around bounding box edges every 10 frames
    if frame_idx % 10 == 0:
        pad = int(((w+h)/2)*0.5)  # Pad is 20% of average dimension
        print("pad:", pad)
        edge_mask = np.zeros_like(frame_gray)
        # Outer rectangle
        ox1 = max(0, new_x - pad)
        oy1 = max(0, new_y - pad)
        ox2 = min(frame_gray.shape[1], new_x + w + pad)
        oy2 = min(frame_gray.shape[0], new_y + h + pad)
        edge_mask[oy1:oy2, ox1:ox2] = 255
        # Cut out the interior so we only detect on edges
        ix1 = min(ox2, new_x + pad)
        iy1 = min(oy2, new_y + pad)
        ix2 = max(ox1, new_x + w - pad)
        iy2 = max(oy1, new_y + h - pad)
        if ix1 < ix2 and iy1 < iy2:
            edge_mask[iy1:iy2, ix1:ix2] = 0
        new_pts = cv2.goodFeaturesToTrack(frame_gray, mask=edge_mask, **feature_params)
        if new_pts is not None:
            new_pts = new_pts.reshape(-1, 2)
            # Remove duplicates that are too close to existing points
            existing = good_new
            keep = []
            for pt in new_pts:
                dists = np.linalg.norm(existing - pt, axis=1)
                if np.min(dists) > feature_params['minDistance']:
                    keep.append(pt)
            if keep:
                good_new = np.vstack([good_new, np.array(keep)])
                print(f"Added {len(keep)} new points, total: {len(good_new)}")

    # Update for next iteration
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
print("Done")
