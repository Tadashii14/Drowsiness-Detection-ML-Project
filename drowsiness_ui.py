# ============================================================
# DROWSINESS DETECTION - TKINTER UI
# Single-window version of integrated_drowsiness_system.py
#
# Usage:
#   python drowsiness_ui.py
# ============================================================

import time
import tkinter as tk

import cv2
from PIL import Image, ImageTk

from integrated_drowsiness_system import (
    CAMERA_INDEX,
    CLASS_NAMES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    CameraManager,
    analyze_eyes,
    crop_mouth_from_landmarks,
    detect_face_landmarks,
    draw_landmarks,
    eye_event_history,
    fresh_eye_temporal_state,
    fresh_fusion_state,
    fresh_mouth_temporal_state,
    mouth_model,
    mouth_prediction_history,
    predict_mouth_state,
    update_eye_temporal_state,
    update_fusion_state,
    update_mouth_temporal_state,
)

STATE_COLORS = {
    "NORMAL": "#00cc44",
    "WARNING": "#ff9900",
    "DROWSY": "#ff2222",
}


class DrowsinessUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection")
        self.root.resizable(False, False)

        self.camera = CameraManager(
            camera_index=CAMERA_INDEX,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            mirror=True,
        )

        self.eye_state = fresh_eye_temporal_state()
        self.mouth_state = fresh_mouth_temporal_state()
        self.fusion_state = fresh_fusion_state()

        self.running = False
        self._photo_refs = {}

        self._build_layout()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if not self.camera.open():
            self.set_status_text("ERROR: could not open camera", "#ff2222")
            return

        self.running = True
        self.update_loop()

    def _build_layout(self):
        main = tk.Frame(self.root, bg="#1e1e1e", padx=10, pady=10)
        main.pack(fill=tk.BOTH, expand=True)

        self.camera_label = tk.Label(main, bg="#000000")
        self.camera_label.grid(row=0, column=0, rowspan=6, padx=(0, 10))

        right = tk.Frame(main, bg="#1e1e1e")
        right.grid(row=0, column=1, sticky="n")

        tk.Label(right, text="Mouth Crop", fg="#cccccc", bg="#1e1e1e",
                 font=("Segoe UI", 10, "bold")).pack()
        self.mouth_label = tk.Label(right, bg="#000000", width=256, height=256)
        self.mouth_label.pack(pady=(2, 12))

        self.ear_var = tk.StringVar(value="EAR: --")
        self.eyes_var = tk.StringVar(value="Eyes: --")
        self.mouth_var = tk.StringVar(value="Mouth: --")
        self.eye_score_var = tk.StringVar(value="Eye Score: --")
        self.mouth_score_var = tk.StringVar(value="Mouth Score: --")

        for var in (self.ear_var, self.eyes_var, self.mouth_var,
                    self.eye_score_var, self.mouth_score_var):
            tk.Label(right, textvariable=var, fg="#dddddd", bg="#1e1e1e",
                     font=("Consolas", 11), anchor="w").pack(fill=tk.X)

        self.state_label = tk.Label(right, text="STATUS: --", fg="#00cc44",
                                    bg="#1e1e1e", font=("Segoe UI", 18, "bold"))
        self.state_label.pack(pady=(14, 4))

        self.score_label = tk.Label(right, text="Score: --/100", fg="#eeeeee",
                                    bg="#1e1e1e", font=("Consolas", 13))
        self.score_label.pack()

        self.quit_button = tk.Button(right, text="Quit", command=self.on_close,
                                     font=("Segoe UI", 10), width=12)
        self.quit_button.pack(pady=(16, 0))

    def set_status_text(self, text, color):
        self.state_label.config(text=text, fg=color)

    def update_loop(self):
        if not self.running:
            return

        ret, frame = self.camera.read()
        if not ret:
            self.running = False
            self.set_status_text("ERROR: camera read failed", "#ff2222")
            return

        display = frame.copy()
        landmarks = detect_face_landmarks(frame)

        mouth = None
        if landmarks is not None:
            h, w = frame.shape[:2]

            eye_data = analyze_eyes(landmarks, w, h)
            self.eye_state = update_eye_temporal_state(
                eye_data["both_closed"], time.time(), self.eye_state
            )

            mouth = crop_mouth_from_landmarks(frame, landmarks)
            prediction, confidence, _ = predict_mouth_state(mouth)
            if prediction is not None:
                self.mouth_state = update_mouth_temporal_state(
                    prediction, confidence, self.mouth_state
                )

            self.fusion_state = update_fusion_state(
                self.eye_state,
                mouth_prediction_history,
                time.time(),
                self.fusion_state,
                eye_event_history,
            )

            draw_landmarks(display, landmarks, w, h)

            self.ear_var.set(f"EAR: {eye_data['average_ear']:.3f}")
            self.eyes_var.set(f"Eyes: {eye_data['combined_state']}")
            self.mouth_var.set(
                f"Mouth: {self.mouth_state['current_prediction']} "
                f"({self.mouth_state['current_confidence']:.0f}%)"
            )
            self.eye_score_var.set(f"Eye Score: {self.fusion_state['eye_score']:.1f}")
            self.mouth_score_var.set(f"Mouth Score: {self.fusion_state['mouth_score']:.1f}")

            final_state = self.fusion_state["state"]
            self.state_label.config(
                text=f"STATUS: {final_state}",
                fg=STATE_COLORS[final_state],
            )
            self.score_label.config(
                text=f"Score: {self.fusion_state['final_score']:.1f}/100"
            )
        else:
            self.ear_var.set("EAR: --")
            self.eyes_var.set("Eyes: FACE NOT DETECTED")

        self._show_image(self.camera_label, display, (960, 540), "camera")

        if mouth is not None:
            self._show_image(self.mouth_label, mouth, (256, 256), "mouth")

        self.root.after(10, self.update_loop)

    def _show_image(self, label, bgr_image, size, key):
        resized = cv2.resize(bgr_image, size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._photo_refs[key] = photo
        label.config(image=photo, width=size[0], height=size[1])

    def on_close(self):
        self.running = False
        try:
            self.camera.release()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessUI(root)
    root.mainloop()
