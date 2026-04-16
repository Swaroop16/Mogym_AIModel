import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

# Download model file if not already present
MODEL_PATH = 'pose_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task'

if not os.path.exists(MODEL_PATH):
    print('Downloading pose model...')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def get_point(landmarks, idx):
    l = landmarks[idx]
    return np.array([l.x, l.y, l.z])

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))
def check_elbow_stability(s, e):
    return abs(e[0] - s[0]) < 0.12

def curl_pushdown_rules(s, e, counters):
    if not check_elbow_stability(s, e):
        counters["elbow_movement"] += 1

def shoulder_press_rules(s, e, w, counters):
    angle = calculate_angle(s, e, w)

    if angle < 90:
        counters["low_angle"] += 1

    if abs(e[0] - s[0]) < 0.12:
        counters["tucked_elbows"] += 1

def bench_press_rules(s, e, counters):
    if abs(e[0] - s[0]) > 0.15:
        counters["flare"] += 1
def get_side_points(landmarks, side="right"):
    if side == "right":
        return (
            get_point(landmarks, 12),
            get_point(landmarks, 14),
            get_point(landmarks, 16)
        )
    else:
        return (
            get_point(landmarks, 11),
            get_point(landmarks, 13),
            get_point(landmarks, 15)
        )

def get_visibility(landmarks, idx):
    return landmarks[idx].visibility

def analyze(video_path: str,exercise_type: str):
    """
    Takes a path to a video file.
    Returns a dict with corrections found.
    """
    print(f"Analyzing: {video_path}, exercise: {exercise_type}")

    counters = {
        "elbow_movement": 0,
        "low_angle": 0,
        "tucked_elbows": 0,
        "flare": 0
    }
    total_frames = 0

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ── Create landmarker in VIDEO mode inside the function ──
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionTaskRunningMode.VIDEO,  # ← KEY FIX
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            timestamp_ms = int((total_frames / fps) * 1000)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # ── Use detect_for_video instead of detect ──
            result = landmarker.detect_for_video(mp_image, timestamp_ms)  # ← KEY FIX

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                vis_r = (get_visibility(landmarks, 12) +
                         get_visibility(landmarks, 14) +
                         get_visibility(landmarks, 16))
                vis_l = (get_visibility(landmarks, 11) +
                         get_visibility(landmarks, 13) +
                         get_visibility(landmarks, 15))

                if vis_r >= vis_l:
                    s, e, w = get_side_points(landmarks, "right")
                else:
                    s, e, w = get_side_points(landmarks, "left")

                if exercise_type in ["bicep_curl", "tricep_pushdown"]:
                    curl_pushdown_rules(s, e, counters)
                elif exercise_type == "shoulder_press":
                    shoulder_press_rules(s, e, w, counters)
                elif exercise_type == "bench_press":
                    bench_press_rules(s, e, counters)

    cap.release()
    print(f"Total frames: {total_frames}, Counters: {counters}")

    corrections = []
    if counters["elbow_movement"] > total_frames * 0.2:
        corrections.append("Keep elbows fixed — avoid swinging")
    if counters["low_angle"] > total_frames * 0.2:
        corrections.append("Elbow angle too low — keep above 90°")
    if counters["tucked_elbows"] > total_frames * 0.2:
        corrections.append("Elbows too tucked — flare slightly outward")
    if counters["flare"] > total_frames * 0.2:
        corrections.append("Elbows flaring too much — tuck slightly")

    score = max(0, 100 - len(corrections) * 15)

    return {
        "exercise": exercise_type,
        "score": score,
        "corrections": corrections if corrections else ["Great form!"],
        "frames_analyzed": total_frames
    }
if __name__ == "__main__":
    result = analyze("sample.mp4", "shoulder_press")
    print(result)
