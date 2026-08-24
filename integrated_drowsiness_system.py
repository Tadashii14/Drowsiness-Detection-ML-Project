# ============================================================
# INTEGRATED DROWSINESS DETECTION SYSTEM
# Script version of 05_integrated_drowsiness_system.ipynb
#
# Pipeline:
#   Camera -> MediaPipe Face Landmarker (one call per frame)
#          -> Eyes: EAR -> open/closed -> temporal events
#          -> Mouth: crop -> CNN -> Closed/Talking/Yawn
#          -> Fusion: 0.7*eye_score + 0.3*mouth_score
#          -> NORMAL / WARNING / DROWSY
#
# Usage:
#   python integrated_drowsiness_system.py
#   Press Q in the video window to quit.
# ============================================================

from pathlib import Path
from collections import deque
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MEDIAPIPE_MODEL_PATH = PROJECT_ROOT / "assets" / "face_landmarker.task"
MOUTH_MODEL_PATH = PROJECT_ROOT / "models" / "mouth_cnn_best.pth"

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30
MIRROR_CAMERA = True

# Eye analysis
EAR_THRESHOLD = 0.19            # EAR >= threshold -> OPEN
BLINK_THRESHOLD = 0.50          # seconds
LONG_BLINK_THRESHOLD = 1.00     # seconds
PROLONGED_CLOSURE_THRESHOLD = 1.00

# Mouth CNN
NUM_CLASSES = 3
CLASS_NAMES = ["Closed", "Talking", "Yawn"]
MOUTH_HISTORY_SIZE = 20

# Fusion
EYE_WEIGHT = 0.70
MOUTH_WEIGHT = 0.30
NORMAL_THRESHOLD = 40.0
DROWSY_THRESHOLD = 70.0
FINAL_SCORE_DECAY_PER_SECOND = 1.5
EYE_PROLONGED_POINTS = 30.0
EYE_LONG_BLINK_POINTS = 5.0
MOUTH_YAWN_POINTS = 20.0
EYE_EVENT_HISTORY_SECONDS = 30

# ============================================================
# CAMERA MANAGER
# ============================================================


class CameraManager:

    def __init__(self, camera_index=0, width=1280, height=720, fps=30, mirror=True):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror = mirror
        self.cap = None
        self.is_open = False

    def open(self):
        if self.is_open:
            return True

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.is_open = False
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.is_open = True
        return True

    def read(self):
        if not self.is_open:
            raise RuntimeError("Camera is not open.")

        ret, frame = self.cap.read()

        if not ret:
            return False, None

        if self.mirror:
            frame = cv2.flip(frame, 1)

        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.is_open = False


# ============================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================


def create_face_landmarker(model_path):
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


face_landmarker = create_face_landmarker(MEDIAPIPE_MODEL_PATH)


def detect_face_landmarks(frame):
    """Run MediaPipe Face Landmarker on one OpenCV frame."""
    if frame is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]


# ============================================================
# LANDMARK INDICES
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_POINTS = [
    61, 146, 91, 181, 84,
    17, 314, 405, 321, 375,
    291, 409, 270, 269, 267,
    0, 37, 39, 40, 185,
]

# ============================================================
# EYE ANALYSIS
# ============================================================


def calculate_ear(landmarks, eye_indices, frame_width, frame_height):
    """Calculate Eye Aspect Ratio (EAR) from six eye landmarks."""
    points = []
    for idx in eye_indices:
        x = landmarks[idx].x * frame_width
        y = landmarks[idx].y * frame_height
        points.append(np.array([x, y]))

    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def classify_eye_state(ear):
    return "OPEN" if ear >= EAR_THRESHOLD else "CLOSED"


def analyze_eyes(landmarks, frame_width, frame_height):
    """Calculate both-eye EAR and eye states from detected landmarks."""
    left_ear = calculate_ear(landmarks, LEFT_EYE, frame_width, frame_height)
    right_ear = calculate_ear(landmarks, RIGHT_EYE, frame_width, frame_height)
    average_ear = (left_ear + right_ear) / 2.0

    left_state = classify_eye_state(left_ear)
    right_state = classify_eye_state(right_ear)

    both_closed = left_state == "CLOSED" and right_state == "CLOSED"

    if both_closed:
        combined_state = "BOTH EYES CLOSED"
    elif left_state == "OPEN" and right_state == "OPEN":
        combined_state = "BOTH EYES OPEN"
    else:
        combined_state = "ONE EYE CLOSED"

    return {
        "left_ear": left_ear,
        "right_ear": right_ear,
        "average_ear": average_ear,
        "left_state": left_state,
        "right_state": right_state,
        "both_closed": both_closed,
        "combined_state": combined_state,
    }


def classify_closure_duration(duration):
    if duration < BLINK_THRESHOLD:
        return "NORMAL BLINK"
    elif duration < LONG_BLINK_THRESHOLD:
        return "LONG BLINK"
    return "PROLONGED CLOSURE"


def update_eye_temporal_state(both_closed, current_time, state):
    """Update eye closure timing based on the both-eyes-closed state."""
    if both_closed:
        if not state["eyes_closed"]:
            state["eyes_closed"] = True
            state["closure_start_time"] = current_time
            state["current_closure_duration"] = 0.0
        else:
            state["current_closure_duration"] = current_time - state["closure_start_time"]
    else:
        if state["eyes_closed"]:
            duration = state["current_closure_duration"]
            state["last_completed_closure"] = duration
            state["last_closure_event"] = classify_closure_duration(duration)
            if duration > state["longest_closure"]:
                state["longest_closure"] = duration

        state["eyes_closed"] = False
        state["closure_start_time"] = None
        state["current_closure_duration"] = 0.0

    return state


# ============================================================
# MOUTH CNN
# ============================================================


class CustomCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mouth_model = CustomCNN().to(device)
mouth_model.load_state_dict(torch.load(MOUTH_MODEL_PATH, map_location=device))
mouth_model.eval()

mouth_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def crop_mouth_from_landmarks(frame, landmarks):
    """Crop the mouth region using already-detected landmarks."""
    if frame is None or landmarks is None:
        return None

    h, w = frame.shape[:2]

    points = []
    for idx in MOUTH_POINTS:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        points.append([x, y])

    points = np.array(points, dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(points)

    pad_x = int(bw * 0.30)
    pad_y = int(bh * 0.40)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    mouth = frame[y1:y2, x1:x2]

    if mouth.size == 0:
        return None

    return cv2.resize(mouth, (128, 128), interpolation=cv2.INTER_AREA)


def predict_mouth_state(mouth_image):
    """Predict the mouth state. Returns (prediction, confidence, probabilities)."""
    if mouth_image is None:
        return None, 0.0, None

    mouth_rgb = cv2.cvtColor(mouth_image, cv2.COLOR_BGR2RGB)
    mouth_pil = Image.fromarray(mouth_rgb).convert("L")
    input_tensor = mouth_transform(mouth_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = mouth_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_index].item() * 100.0

    return CLASS_NAMES[predicted_index], confidence, probabilities[0].cpu().numpy()


# ============================================================
# MOUTH TEMPORAL SMOOTHING
# ============================================================

mouth_prediction_history = deque(maxlen=MOUTH_HISTORY_SIZE)


def update_mouth_temporal_state(prediction, confidence, state):
    """Update recent mouth predictions and calculate the smoothed state."""
    if prediction is None:
        return state

    mouth_prediction_history.append(prediction)
    state["current_prediction"] = prediction
    state["current_confidence"] = confidence

    counts = {}
    for item in mouth_prediction_history:
        counts[item] = counts.get(item, 0) + 1

    smoothed_prediction = max(counts, key=counts.get)
    state["smoothed_prediction"] = smoothed_prediction
    state["smoothed_count"] = counts[smoothed_prediction]
    state["yawn_count"] = sum(item == "Yawn" for item in mouth_prediction_history)

    return state


# ============================================================
# DROWSINESS SCORING / FUSION
# ============================================================

eye_event_history = deque()


def calculate_eye_score(eye_temporal_state, current_time, eye_event_history):
    """Eye-based drowsiness score from prolonged closures and long blinks."""
    score = 0.0

    current_duration = eye_temporal_state["current_closure_duration"]
    if eye_temporal_state["eyes_closed"] and current_duration >= PROLONGED_CLOSURE_THRESHOLD:
        extra_duration = min(current_duration, 3.0)
        score += EYE_PROLONGED_POINTS * (extra_duration / 3.0)

    for event in eye_event_history:
        age = current_time - event["time"]
        if age <= EYE_EVENT_HISTORY_SECONDS:
            recency_factor = max(0.0, 1.0 - (age / EYE_EVENT_HISTORY_SECONDS))
            if event["type"] == "PROLONGED CLOSURE":
                score += EYE_PROLONGED_POINTS * recency_factor
            elif event["type"] == "LONG BLINK":
                score += EYE_LONG_BLINK_POINTS * recency_factor

    return min(100.0, max(0.0, score))


def calculate_mouth_score(mouth_prediction_history):
    """Mouth-based supporting score from recent yawn ratio."""
    if len(mouth_prediction_history) == 0:
        return 0.0

    yawn_count = sum(p == "Yawn" for p in mouth_prediction_history)
    yawn_ratio = yawn_count / len(mouth_prediction_history)

    return min(100.0, max(0.0, yawn_ratio * 100.0))


def calculate_final_drowsiness_score(eye_score, mouth_score):
    final_score = EYE_WEIGHT * eye_score + MOUTH_WEIGHT * mouth_score
    return min(100.0, max(0.0, final_score))


def classify_drowsiness_state(final_score):
    if final_score >= DROWSY_THRESHOLD:
        return "DROWSY"
    elif final_score >= NORMAL_THRESHOLD:
        return "WARNING"
    return "NORMAL"


def update_fusion_state(eye_temporal_state, mouth_prediction_history,
                        current_time, state, eye_events):
    """Update eye/mouth/final scores with gradual decay."""
    # Record newly completed eye event
    last_event = eye_temporal_state["last_closure_event"]
    last_completed_duration = eye_temporal_state["last_completed_closure"]

    if last_event != "None" and last_completed_duration > 0:
        already_recorded = False
        if len(eye_events) > 0:
            already_recorded = (
                abs(eye_events[-1]["duration"] - last_completed_duration) < 0.01
            )
        if not already_recorded:
            eye_events.append({
                "time": current_time,
                "type": last_event,
                "duration": last_completed_duration,
            })

    # Remove old eye events
    cutoff = current_time - EYE_EVENT_HISTORY_SECONDS
    while eye_events and eye_events[0]["time"] < cutoff:
        eye_events.popleft()

    eye_score = calculate_eye_score(eye_temporal_state, current_time, eye_events)
    mouth_score = calculate_mouth_score(mouth_prediction_history)
    final_score = calculate_final_drowsiness_score(eye_score, mouth_score)

    # Score decay / recovery
    elapsed = current_time - state["last_update_time"]

    if eye_score < NORMAL_THRESHOLD and mouth_score < NORMAL_THRESHOLD:
        state["final_score"] -= FINAL_SCORE_DECAY_PER_SECOND * elapsed
        state["final_score"] = max(state["final_score"], 0.0)
        state["final_score"] = max(state["final_score"], final_score)
    else:
        state["final_score"] = final_score

    state["final_score"] = min(100.0, max(0.0, state["final_score"]))
    state["eye_score"] = eye_score
    state["mouth_score"] = mouth_score
    state["state"] = classify_drowsiness_state(state["final_score"])
    state["last_update_time"] = current_time

    return state


# ============================================================
# STATE INITIALIZATION
# ============================================================


def fresh_eye_temporal_state():
    return {
        "eyes_closed": False,
        "closure_start_time": None,
        "current_closure_duration": 0.0,
        "last_completed_closure": 0.0,
        "last_closure_event": "None",
        "longest_closure": 0.0,
    }


def fresh_mouth_temporal_state():
    return {
        "current_prediction": "None",
        "current_confidence": 0.0,
        "smoothed_prediction": "None",
        "smoothed_count": 0,
        "yawn_count": 0,
    }


def fresh_fusion_state():
    return {
        "eye_score": 0.0,
        "mouth_score": 0.0,
        "final_score": 0.0,
        "state": "NORMAL",
        "last_update_time": time.time(),
    }


# ============================================================
# MAIN LOOP
# ============================================================


def draw_landmarks(display_frame, landmarks, w, h):
    for idx in LEFT_EYE + RIGHT_EYE:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        cv2.circle(display_frame, (x, y), 3, (0, 255, 0), -1)
    for idx in MOUTH_POINTS:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        cv2.circle(display_frame, (x, y), 2, (255, 0, 0), -1)


def main():
    print("=" * 60)
    print("INTEGRATED DROWSINESS DETECTION SYSTEM")
    print("=" * 60)
    print(f"Device        : {device}")
    print(f"Mouth model   : {MOUTH_MODEL_PATH}")
    print(f"MediaPipe     : {MEDIAPIPE_MODEL_PATH}")
    print(f"Camera        : {CAMERA_INDEX} @ {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print("Press Q in the video window to quit.")
    print("=" * 60)

    camera = CameraManager(
        camera_index=CAMERA_INDEX,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=TARGET_FPS,
        mirror=MIRROR_CAMERA,
    )

    if not camera.open():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    eye_temporal_state = fresh_eye_temporal_state()
    mouth_temporal_state = fresh_mouth_temporal_state()
    fusion_state = fresh_fusion_state()

    mouth_prediction_history.clear()
    eye_event_history.clear()

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to read camera frame.")
                break

            display_frame = frame.copy()

            # One shared MediaPipe call per frame
            landmarks = detect_face_landmarks(frame)

            if landmarks is not None:
                h, w = frame.shape[:2]

                # Eyes
                eye_data = analyze_eyes(landmarks, w, h)
                eye_temporal_state = update_eye_temporal_state(
                    eye_data["both_closed"], time.time(), eye_temporal_state
                )

                # Mouth
                mouth = crop_mouth_from_landmarks(frame, landmarks)
                prediction, confidence, _ = predict_mouth_state(mouth)
                if prediction is not None:
                    mouth_temporal_state = update_mouth_temporal_state(
                        prediction, confidence, mouth_temporal_state
                    )

                # Fusion
                fusion_state = update_fusion_state(
                    eye_temporal_state,
                    mouth_prediction_history,
                    time.time(),
                    fusion_state,
                    eye_event_history,
                )

                draw_landmarks(display_frame, landmarks, w, h)

                # Eye HUD
                cv2.putText(display_frame, f"EAR: {eye_data['average_ear']:.3f}",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Eyes: {eye_data['combined_state']}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
                cv2.putText(display_frame,
                            f"Eye Closure: {eye_temporal_state['current_closure_duration']:.2f}s",
                            (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Eye Score: {fusion_state['eye_score']:.1f}",
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

                # Mouth HUD
                cv2.putText(display_frame,
                            f"Mouth: {mouth_temporal_state['current_prediction']}",
                            (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 150, 0), 2)
                cv2.putText(display_frame, f"Mouth Score: {fusion_state['mouth_score']:.1f}",
                            (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 150, 0), 2)
                cv2.putText(display_frame,
                            f"Recent Yawns: {mouth_temporal_state['yawn_count']}/{len(mouth_prediction_history)}",
                            (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

                # Final score + state
                cv2.putText(display_frame,
                            f"FINAL DROWSINESS SCORE: {fusion_state['final_score']:.1f}/100",
                            (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 255), 2)

                final_state = fusion_state["state"]
                if final_state == "DROWSY":
                    state_color = (0, 0, 255)
                elif final_state == "WARNING":
                    state_color = (0, 165, 255)
                else:
                    state_color = (0, 255, 0)

                cv2.putText(display_frame, f"STATUS: {final_state}",
                            (20, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.90, state_color, 3)

                # Mouth crop window
                if mouth is not None:
                    cv2.imshow("Mouth Crop", mouth)

            else:
                cv2.putText(display_frame, "FACE NOT DETECTED",
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2)

            cv2.imshow("Integrated Drowsiness System", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        camera.release()

    print("=" * 60)
    print("SESSION COMPLETE")
    print("=" * 60)
    print(f"Final Score    : {fusion_state['final_score']:.1f}/100")
    print(f"Final State    : {fusion_state['state']}")
    print(f"Eye Score      : {fusion_state['eye_score']:.1f}")
    print(f"Mouth Score    : {fusion_state['mouth_score']:.1f}")
    print(f"Longest Closure: {eye_temporal_state['longest_closure']:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
