import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

# ── Model download ────────────────────────────────────────────────────────────
MODEL_PATH = 'pose_landmarker.task'
MODEL_URL  = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task'

if not os.path.exists(MODEL_PATH):
    print('Downloading pose model...')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print('Done.')

# ── Pose connections for drawing (Tasks API compatible) ───────────────────────
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),
    (15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),
]

# ── Helper functions ──────────────────────────────────────────────────────────

def get_point(landmarks, idx):
    l = landmarks[idx]
    return np.array([l.x, l.y, l.z])

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def check_elbow_stability(s, e):
    return abs(e[0] - s[0]) < 0.12

def curl_pushdown_rules(s, e, counters):
    if not check_elbow_stability(s, e):
        counters["elbow_movement"] += 1

def shoulder_press_rules(s, e, w_point, counters):
    angle = calculate_angle(s, e, w_point)
    if angle < 90:
        counters["low_angle"] += 1
    if abs(e[0] - s[0]) < 0.05:
        counters["tucked_elbows"] += 1

def bench_press_rules(s, e, counters):
    if abs(e[0] - s[0]) > 0.15:
        counters["flare"] += 1

def get_side_points(landmarks, side="right"):
    if side == "right":
        return (get_point(landmarks, 12), get_point(landmarks, 14), get_point(landmarks, 16))
    else:
        return (get_point(landmarks, 11), get_point(landmarks, 13), get_point(landmarks, 15))

def get_visibility(landmarks, idx):
    return landmarks[idx].visibility

# ── Draw landmarks using Tasks API landmarks ──────────────────────────────────

def draw_landmarks_on_frame(frame, landmarks, frame_w, frame_h):
    """Draw pose skeleton on frame using Tasks API landmark format."""

    # Draw connections
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        start = landmarks[start_idx]
        end   = landmarks[end_idx]
        if start.visibility < 0.3 or end.visibility < 0.3:
            continue
        x1 = int(start.x * frame_w)
        y1 = int(start.y * frame_h)
        x2 = int(end.x   * frame_w)
        y2 = int(end.y   * frame_h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

    # Draw joint dots
    for lm in landmarks:
        if lm.visibility < 0.3:
            continue
        cx = int(lm.x * frame_w)
        cy = int(lm.y * frame_h)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
        cv2.circle(frame, (cx, cy), 5, (0, 150, 255),  2)

# ── Overlay text helpers ──────────────────────────────────────────────────────

def put_label(frame, text, pos, color=(255,255,255), scale=0.6, thickness=2):
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color,  thickness,   cv2.LINE_AA)

def draw_hud(frame, exercise_type, feedback_lines, frame_num, total_f, score, frame_w, frame_h):
    """Draw a clean HUD overlay on the frame."""

    # Top bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Exercise name top-left
    put_label(frame, exercise_type.replace('_', ' ').upper(),
              (16, 30), color=(0, 220, 150), scale=0.75, thickness=2)

    # Frame counter top-right
    put_label(frame, f"Frame {frame_num}/{total_f}",
              (frame_w - 180, 30), color=(200, 200, 200), scale=0.55)

    # Score top-right below frame counter
    score_color = (0, 220, 100) if score >= 80 else (0, 165, 255) if score >= 60 else (0, 60, 255)
    put_label(frame, f"Score: {score}%",
              (frame_w - 180, 58), color=score_color, scale=0.6, thickness=2)

    # Feedback lines — bottom bar
    bar_h = 36 + max(0, len(feedback_lines) - 1) * 28
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, frame_h - bar_h), (frame_w, frame_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)

    for i, (text, color) in enumerate(feedback_lines):
        y = frame_h - bar_h + 26 + i * 28
        put_label(frame, text, (16, y), color=color, scale=0.65, thickness=2)

# ── Main analyze function ─────────────────────────────────────────────────────

def analyze(video_path: str, exercise_type: str):
    exercise_type = exercise_type or "bicep_curl"  # default if None
    print(f"\nAnalyzing: {video_path}  |  Exercise: {exercise_type}")

    counters = {
        "elbow_movement": 0,
        "low_angle":      0,
        "tucked_elbows":  0,
        "flare":          0,
    }

    cap = cv2.VideoCapture(video_path)
    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # BUG 1 FIX: renamed from w
    frame_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # renamed from h
    total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video path — same folder as input
    base      = os.path.splitext(os.path.basename(video_path))[0]
    out_dir   = 'C:/Users/Swaroop/Documents/dbs project/Outputs'
    output_path = os.path.join(out_dir, f"analysis_{base}.mp4")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_w, frame_h)   # BUG 1 FIX: uses frame_w/frame_h not w/h
    )

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    total_frames = 0
    prev_e       = None

    # Collect per-frame corrections for live overlay
    frame_corrections = []

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            timestamp_ms = int((total_frames / fps) * 1000)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result    = landmarker.detect_for_video(mp_image, timestamp_ms)

            feedback_lines = [("Analysing...", (200, 200, 200))]

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # BUG 2 FIX: use our own draw function instead of mp.solutions.drawing_utils
                draw_landmarks_on_frame(frame, landmarks, frame_w, frame_h)

                vis_r = (get_visibility(landmarks, 12) +
                         get_visibility(landmarks, 14) +
                         get_visibility(landmarks, 16))
                vis_l = (get_visibility(landmarks, 11) +
                         get_visibility(landmarks, 13) +
                         get_visibility(landmarks, 15))

                if vis_r >= vis_l:
                    s, e, w_pt = get_side_points(landmarks, "right")  # BUG 1 FIX: w_pt not w
                else:
                    s, e, w_pt = get_side_points(landmarks, "left")

                # Per-frame live feedback
                frame_feedback = []

                if exercise_type in ["bicep_curl", "tricep_pushdown"]:
                    curl_pushdown_rules(s, e, counters)
                    if not check_elbow_stability(s, e):
                        frame_feedback.append(("⚠ Keep elbows fixed", (0, 60, 255)))
                    else:
                        frame_feedback.append(("✓ Good elbow position", (0, 220, 100)))

                elif exercise_type == "shoulder_press":
                    shoulder_press_rules(s, e, w_pt, counters)
                    angle = calculate_angle(s, e, w_pt)
                    if angle < 90:
                        frame_feedback.append(("⚠ Raise elbows above 90°", (0, 60, 255)))
                    elif abs(e[0] - s[0]) < 0.05:
                        frame_feedback.append(("⚠ Elbows too tucked", (0, 165, 255)))
                    else:
                        frame_feedback.append(("✓ Good form", (0, 220, 100)))

                elif exercise_type == "bench_press":
                    bench_press_rules(s, e, counters)
                    if abs(e[0] - s[0]) > 0.15:
                        frame_feedback.append(("⚠ Elbows flaring — tuck in", (0, 60, 255)))
                    else:
                        frame_feedback.append(("✓ Good elbow position", (0, 220, 100)))

                else:
                    frame_feedback.append(("Monitoring...", (200, 200, 200)))

                feedback_lines = frame_feedback
                prev_e = e

            # Running score estimate for HUD
            run_score = max(0, 100 - (
                counters["elbow_movement"] +
                counters["low_angle"] +
                counters["tucked_elbows"] +
                counters["flare"]
            ) * 2)

            draw_hud(frame, exercise_type, feedback_lines,
                     total_frames, total_est, min(run_score, 100),
                     frame_w, frame_h)

            writer.write(frame)

    cap.release()
    writer.release()

    print(f"Total frames: {total_frames} | Counters: {counters}")
    print(f"Output video saved → {output_path}")

    corrections = []
    if counters["elbow_movement"] > total_frames * 0.2:
        corrections.append("Keep elbows fixed — avoid swinging")
    if counters["low_angle"] > total_frames * 0.2:
        corrections.append("Elbow angle too low — keep above 90°")
    if counters["tucked_elbows"] > total_frames * 0.2:
        corrections.append("Elbows too tucked — flare slightly outward")
    if counters["flare"] > total_frames * 0.2:
        corrections.append("Elbows flaring too much — tuck slightly")

    score = max(0, 100 - len(corrections) * 30)

    return {
        "exercise":       exercise_type,
        "score":          score,
        "corrections":    corrections if corrections else ["Great form!"],
        "frames_analyzed": total_frames,
        "output_video":   output_path,
    }

if __name__ == "__main__":
    result = analyze("sample.mp4", "bicep_curl")
    print(result)