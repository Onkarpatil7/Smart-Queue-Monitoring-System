# detection.py (edited)
import cv2
import time
import numpy as np
from ultralytics import YOLO
import math
from collections import OrderedDict
import requests
from datetime import datetime
import threading
import os
import json

# -------------------- CONFIGURATION & CONSTANTS --------------------
# Model and Detection Settings
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.6
MIN_BOX_AREA = 4000

# Tracking Settings
MAX_DIST = 120
MAX_LOST = 50

# Frame and Room Geometry
FRAME_W, FRAME_H = 960, 540
ROOM_PADDING = 30
ROOM_X1, ROOM_Y1 = ROOM_PADDING, ROOM_PADDING
ROOM_X2, ROOM_Y2 = FRAME_W - ROOM_PADDING, FRAME_H - ROOM_PADDING

# Crossing Lines (Logic: Right -> Left)
ENTRY_LINE_X = ROOM_X2 - 80 # Green line (Right)
EXIT_LINE_X = ROOM_X1 + 80 # Red line (Left)

# Colors (B-G-R)
COL_ENTRY = (0, 255, 0) # Green
COL_EXIT = (0, 0, 255) # Red
COL_ROOM = (60, 60, 60) # Dark Grey
COL_INSIDE = (255, 255, 0) # Yellow/Cyan
COL_DEFAULT = (255, 255, 255) # White
COL_ALERT = (0, 0, 255) # Red for alert

# **[NEW FEATURE]** Crowd Limit Constant
MAX_PEOPLE = 4 # Do not detect or assign ID to more than 1 people (set as needed)

# Backend endpoints
BACKEND_BASE = "http://127.0.0.1:8000"
PUBLISH_URL = f"{BACKEND_BASE}/publish/"
UPDATE_URL = f"{BACKEND_BASE}/updateData/"

# Stats Storage
waiting_times = [] # Individual stay times. This list remains global.

# Throttle publishing (seconds)
PUBLISH_INTERVAL = 1.0  # publish max once per second

# -------------------- UTILITIES --------------------

def calculate_distance(p1: tuple, p2: tuple) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_centroid(box: tuple) -> tuple:
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_box_area(box: tuple) -> int:
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)

def post_exit_to_backend(person_id: int, entry_epoch: float, exit_epoch: float, wait_seconds: float, alert_flag: bool):
    """Post exit data to the backend FastAPI which persists to Postgres.
    Runs in a background thread to avoid blocking the main loop.
    """
    try:
        payload = {
            "id": int(person_id),
            "entryTime": datetime.fromtimestamp(entry_epoch).isoformat(),
            "exitTime": datetime.fromtimestamp(exit_epoch).isoformat(),
            "waitTime": float(wait_seconds),
            "alert": 1 if alert_flag else 0,
        }
        resp = requests.post(UPDATE_URL, json=payload, timeout=5)
        if resp.status_code == 200:
            print(f"[POST_EXIT] Posted exit data for ID {person_id} to backend")
        else:
            print(f"[POST_EXIT] Backend returned {resp.status_code} for ID {person_id}: {resp.text}")
    except Exception as e:
        print(f"[POST_EXIT] Failed to post exit data for ID {person_id}: {e}")

def publish_stats(payload: dict):
    """Send aggregated stats to backend /publish/ to be broadcast over websocket.
    Fire-and-forget in a thread.
    """
    try:
        resp = requests.post(PUBLISH_URL, json=payload, timeout=2)
        if resp.status_code != 200:
            print(f"[PUBLISH] backend returned {resp.status_code}: {resp.text}")
    except Exception as e:
        # keep errors non-fatal to detection loop
        print(f"[PUBLISH] failed: {e}")

'''def draw_stats_panel(panel: np.ndarray, entered: int, exited: int, total_wait: float, avg_wait: float):
    cv2.putText(panel, "People Tracker", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL_DEFAULT, 2)
    cv2.putText(panel, f"Entered: {entered}", (14, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_ENTRY, 2)
    cv2.putText(panel, f"Exited: {exited}", (14, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_EXIT, 2)
    cv2.putText(panel, f"Inside: {entered - exited}", (14, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_INSIDE, 2)
    cv2.putText(panel, "WAIT TIME (s)", (14, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_DEFAULT, 1)
    cv2.putText(panel, f"Total: {int(total_wait)}", (14, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_INSIDE, 2)
    cv2.putText(panel, f"Average: {avg_wait:.1f}", (14, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_INSIDE, 2)
    cv2.putText(panel, f"Max Limit: {MAX_PEOPLE}", (14, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_DEFAULT, 1)
    cv2.putText(panel, "Press Q to Quit", (14, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)'''

# Load Model
model = YOLO(MODEL_PATH)

# Initialize Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Cannot open camera")
    exit()

# Tracking Variables
next_id = 0
tracks = OrderedDict()
entered, exited = 0, 0

print("press 'q' to quit.")

last_publish_time = 0.0

# main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    results = model(frame, verbose=False, classes=0, conf=CONF_THRESHOLD)
    detections = results[0].boxes.data.cpu().numpy()

    is_crowded_at_start = (entered - exited) >= MAX_PEOPLE

    # 1. Prepare Valid Detections
    boxes = []
    for x1, y1, x2, y2, conf, cls in detections:
        box = (int(x1), int(y1), int(x2), int(y2))
        if get_box_area(box) > MIN_BOX_AREA:
            boxes.append(box)

    det_centroids = [get_centroid(b) for b in boxes]

    # If crowded, ignore new detections
    if is_crowded_at_start:
        used_det_indices = [True] * len(boxes)
    else:
        used_det_indices = [False] * len(boxes)

    # 2. Track Matching & Update
    ids = list(tracks.keys())
    tr_centroids = [tracks[i]['centroid'] for i in ids]

    for i, c in enumerate(det_centroids):
        best_tid, best_d = None, MAX_DIST + 1

        for j, tid in enumerate(ids):
            d = calculate_distance(c, tr_centroids[j])
            if d < best_d and d <= MAX_DIST:
                best_tid, best_d = tid, d

        if best_tid is not None:
            t = tracks[best_tid]
            t['centroid'] = c
            t['box'] = boxes[i]
            t['lost'] = 0
            t['history'].append(c)
            used_det_indices[i] = True

    # 3. Create New Tracks
    if not is_crowded_at_start:
        for i, is_used in enumerate(used_det_indices):
            if not is_used:
                if len(tracks) < MAX_PEOPLE:
                    tracks[next_id] = {
                        'centroid': det_centroids[i],
                        'box': boxes[i],
                        'history': [det_centroids[i]],
                        'lost': 0,
                        'entered': False,
                        'exited': False,
                        'entry_time': None,
                        'waiting_time': None
                    }
                    next_id += 1
                else:
                    break

    # 4. Remove Lost Tracks
    for tid in list(tracks.keys()):
        tracks[tid]['lost'] += 1
        if tracks[tid]['lost'] > MAX_LOST:
            if tracks[tid]['entered'] and not tracks[tid]['exited']:
                print(f" ID {tid} LOST without exiting!")
            del tracks[tid]

    # 5. ENTRY/EXIT LOGIC & Statistics Update
    for tid, t in tracks.items():
        if len(t['history']) < 2:
            continue

        p_prev = t['history'][-2]
        p_curr = t['history'][-1]

        # ENTRY crossing
        if (not t['entered'] and p_prev[0] > ENTRY_LINE_X and p_curr[0] < ENTRY_LINE_X):
            t['entered'] = True
            t['entry_time'] = time.time()
            entered += 1
            print(f" ID {tid} ENTERED at {time.strftime('%H:%M:%S')}")

        # EXIT crossing
        elif (t['entered'] and not t['exited'] and p_prev[0] > EXIT_LINE_X and p_curr[0] < EXIT_LINE_X):
            t['exited'] = True
            exit_time = time.time()
            exited += 1
            stay = exit_time - t['entry_time']
            t['waiting_time'] = stay
            waiting_times.append(stay)
            print(f" ID {tid} EXITED | Stay: {stay:.1f}s")

            print(f" Individual Wait Details:")
            print(f"     Person ID: {tid}")
            print(f"     Entry Time: {time.strftime('%H:%M:%S', time.localtime(t['entry_time']))}")
            print(f"     Exit Time:  {time.strftime('%H:%M:%S', time.localtime(exit_time))}")
            print(f"     Wait Duration: {stay:.2f} seconds\n")

            # Determine whether alert was active while the person was inside.
            try:
                alert_flag_at_exit = (entered - exited + 1) >= MAX_PEOPLE
                threading.Thread(
                    target=post_exit_to_backend,
                    args=(tid, t['entry_time'], exit_time, stay, alert_flag_at_exit),
                    daemon=True,
                ).start()
            except Exception as e:
                print(f"Failed to start background post thread for ID {tid}: {e}")

    # After processing compute current inside count and crowd status
    current_inside = entered - exited
    is_crowded = current_inside >= MAX_PEOPLE

    # 6. DRAWING SECTION
    cv2.rectangle(frame, (ROOM_X1, ROOM_Y1), (ROOM_X2, ROOM_Y2), COL_ROOM, 2)
    cv2.line(frame, (ENTRY_LINE_X, ROOM_Y1), (ENTRY_LINE_X, ROOM_Y2), COL_ENTRY, 3)
    cv2.putText(frame, "ENTRY", (ENTRY_LINE_X - 70, ROOM_Y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_ENTRY, 2)
    cv2.line(frame, (EXIT_LINE_X, ROOM_Y1), (EXIT_LINE_X, ROOM_Y2), COL_EXIT, 3)
    cv2.putText(frame, "EXIT", (EXIT_LINE_X + 10, ROOM_Y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_EXIT, 2)

    for tid, t in tracks.items():
        x1, y1, x2, y2 = t['box']
        if t['entered'] and not t['exited']:
            color = COL_INSIDE
        elif t['exited']:
            color = COL_EXIT
        else:
            color = COL_DEFAULT

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{tid}"
        if t['entered'] and not t['exited']:
            stay = int(time.time() - t['entry_time'])
            label += f" | {stay}s (in)"
        elif t['exited']:
            stay = int(t['waiting_time'])
            label += f" | {stay}s (wait)"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if is_crowded:
        alert_text = "CROWD ALERT: OVER LIMIT!"
        (text_w, text_h), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        text_x = int((FRAME_W - text_w) / 2)
        text_y = int(FRAME_H / 2)
        cv2.rectangle(frame, (text_x - 10, text_y - text_h - 10), (text_x + text_w + 10, text_y + 10), (0, 0, 255), -1)
        cv2.putText(frame, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        print(f" CROWD ALERT: Limit of {MAX_PEOPLE} reached or exceeded. New people are NOT being tracked.")

    # 7. SIDE PANEL
    panel = np.full((FRAME_H, 300, 3), (25, 25, 25), np.uint8)
    current_total_waiting = sum(waiting_times)
    current_average_waiting = current_total_waiting / len(waiting_times) if waiting_times else 0.0

    # draw_stats_panel(panel, entered, exited, current_total_waiting, current_average_waiting)

    # Save current frame and stats for backend to serve if needed
    try:
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        frame_path = os.path.join(backend_dir, "detection_frame.jpg")
        state_path = os.path.join(backend_dir, "detection_state.json")
        cv2.imwrite(frame_path, frame)

        stats = {
            'entered': entered,
            'exited': exited,
            'inside': entered - exited,
            'total_wait_time': current_total_waiting,
            'average_wait_time': current_average_waiting,
            'is_crowded': is_crowded,
            'current_people': len(tracks),
            'max_limit': MAX_PEOPLE,
            'ts': datetime.utcnow().isoformat()
        }
        with open(state_path, 'w') as f:
            json.dump(stats, f)
    except Exception as e:
        print(f"Error saving state/frame: {e}")

    # Publish stats to backend (throttled)
    now = time.time()
    if now - last_publish_time >= PUBLISH_INTERVAL:
        try:
            threading.Thread(target=publish_stats, args=(stats,), daemon=True).start()
        except Exception as e:
            print(f"[PUBLISH_THREAD] failed to start: {e}")
        last_publish_time = now

    # Show locally (optional)
    cv2.imshow("Smart Entry-Exit Tracker", frame)

    # Quit Handler
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

final_total_waiting = sum(waiting_times)
final_average_waiting = final_total_waiting / len(waiting_times) if waiting_times else 0.0

print(f"\n FINAL SUMMARY")
print(f"  → Total Entered: {entered}")
print(f"  → Total Exited: {exited}")
print(f"  → Current Inside (Estimated): {entered - exited}")
print(f"  → Total Wait Time Recorded: {final_total_waiting:.1f}s")
print(f"  → Average Wait Time: {final_average_waiting:.1f}s")
