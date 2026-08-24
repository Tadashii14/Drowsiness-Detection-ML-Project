# ============================================================
# DROWSINESS DETECTION - TKINTER UI
# Single-window version of integrated_drowsiness_system.py
#
# Usage:
#   python drowsiness_ui.py
# ============================================================

import time
import tkinter as tk
import threading
import queue
from collections import deque
from tkinter import ttk

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
    "NORMAL": "#6fe0b0",
    "WARNING": "#f0c674",
    "DROWSY": "#f27f9b",
}

UI_BG = "#1d2734"
PANEL_BG = "#526378"
RAISED_PANEL = "#63758a"
TEXT_COLOR = "#f4f6fb"
MUTED_COLOR = "#c1cad8"
ACCENT_COLOR = "#8ee7f4"
LILAC_COLOR = "#c8b8ff"


class DrowsinessUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)
        self.root.resizable(True, True)

        self.camera_choices = self._find_cameras()
        self.camera_index = (
            CAMERA_INDEX if CAMERA_INDEX in self.camera_choices
            else (self.camera_choices[0] if self.camera_choices else CAMERA_INDEX)
        )

        self.camera = CameraManager(
            camera_index=self.camera_index,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            mirror=True,
            backend=cv2.CAP_DSHOW,
        )

        self.eye_state = fresh_eye_temporal_state()
        self.mouth_state = fresh_mouth_temporal_state()
        self.fusion_state = fresh_fusion_state()

        self.running = False
        self.update_job = None
        self.switching = False
        self.camera_result_queue = queue.Queue()
        self._photo_refs = {}
        self.frame_times = deque(maxlen=30)
        self.fps_var = tk.StringVar(value="-- FPS")
        self.settings_window = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.capture_thread = None
        self.capture_stop = threading.Event()
        self.last_detection_time = 0
        self.last_landmarks = None
        self.last_mouth = None
        self.last_eye_data = None
        self.score_history = deque(maxlen=90)
        self.last_chart_time = 0

        self._build_layout()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if not self.camera.open():
            self.set_status_text("ERROR: could not open camera", "#ff2222")
            return

        self._start_capture(self.camera)
        self.running = True
        self.update_loop()

    def _build_layout(self):
        self.root.configure(bg=UI_BG)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Modern.TButton", padding=(12, 7), font=("Segoe UI", 9),
                background="#7187a0", foreground="#f4f6fb",
                bordercolor="#8da1b7", lightcolor="#8da1b7",
                darkcolor="#52677e", relief="flat")
        style.map("Modern.TButton", background=[("active", "#8ba2ba"), ("disabled", "#43566b")],
              foreground=[("disabled", "#9eacbb")])
        style.configure("Modern.TCombobox", padding=5, font=("Segoe UI", 9),
                fieldbackground="#7187a0", background="#7187a0",
                foreground="#f4f6fb", arrowcolor="#f4f6fb",
                bordercolor="#8da1b7", lightcolor="#8da1b7",
                darkcolor="#52677e", relief="flat")
        style.map("Modern.TCombobox",
              fieldbackground=[("readonly", "#7187a0"), ("active", "#8ba2ba")],
              foreground=[("readonly", "#f4f6fb")],
              selectbackground=[("readonly", "#8ba2ba")],
              selectforeground=[("readonly", "#f4f6fb")])

        shell = tk.Frame(self.root, bg=UI_BG)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.grid_columnconfigure(1, weight=1)
        shell.grid_rowconfigure(0, weight=1)

        rail = tk.Frame(shell, bg="#17212d", width=190, padx=20, pady=24)
        rail.grid(row=0, column=0, sticky="ns")
        rail.grid_propagate(False)
        tk.Label(rail, text="DROWSY", fg=ACCENT_COLOR, bg="#17212d",
                 font=("Segoe UI", 18, "bold")).pack(anchor="w")
        tk.Label(rail, text="DRIVER MONITOR", fg=MUTED_COLOR, bg="#17212d",
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 38))
        self._rail_item(rail, "Dashboard", True)
        self._rail_item(rail, "Camera settings", False, self.open_settings)
        self._rail_item(rail, "Quit session", False, self.on_close)

        content = tk.Frame(shell, bg=UI_BG, padx=28, pady=24)
        content.grid(row=0, column=1, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(2, weight=1)

        top = tk.Frame(content, bg=UI_BG)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        tk.Label(top, text="Good to see you", fg=TEXT_COLOR, bg=UI_BG,
                 font=("Segoe UI", 21, "bold")).pack(side=tk.LEFT)
        tk.Label(top, text="  /  LIVE SESSION", fg=ACCENT_COLOR, bg=UI_BG,
                 font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, pady=(7, 0))
        self.fps_badge = tk.Label(top, textvariable=self.fps_var, fg="#b9f5e2",
                      bg="#31534f", padx=12, pady=6,
                                  font=("Consolas", 9, "bold"))
        self.fps_badge.pack(side=tk.RIGHT, padx=(12, 0))

        controls = tk.Frame(top, bg=UI_BG)
        controls.pack(side=tk.RIGHT)
        self.camera_var = tk.StringVar()
        self.camera_selector = ttk.Combobox(controls, textvariable=self.camera_var,
                                             state="readonly", width=15,
                                             style="Modern.TCombobox")
        self.camera_selector.pack(side=tk.LEFT)
        self.camera_selector.bind("<<ComboboxSelected>>", self.switch_camera)
        self.camera_selector.configure(postcommand=self._camera_dropdown_opened)
        ttk.Button(controls, text="Refresh", command=self.refresh_cameras,
                   style="Modern.TButton").pack(side=tk.LEFT, padx=(8, 0))
        self._update_camera_selector()

        body = tk.Frame(content, bg=UI_BG)
        body.grid(row=1, column=0, rowspan=2, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=UI_BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)
        self.camera_card = tk.Frame(left, bg="#71849a", width=760, height=430,
                        padx=10, pady=10)
        self.camera_card.grid(row=0, column=0, sticky="nsew")
        self.camera_card.grid_propagate(False)
        self.camera_view = tk.Frame(self.camera_card, bg="#15202b")
        self.camera_view.pack(fill=tk.BOTH, expand=True)
        self.camera_label = tk.Label(self.camera_view, bg="#15202b")
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        chart_card = tk.Frame(left, bg="#4a5c70", padx=18, pady=14)
        chart_card.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        chart_head = tk.Frame(chart_card, bg="#4a5c70")
        chart_head.pack(fill=tk.X)
        tk.Label(chart_head, text="Drowsiness score", fg=TEXT_COLOR, bg="#4a5c70",
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        tk.Label(chart_head, text="last 60 readings", fg=MUTED_COLOR, bg="#4a5c70",
                 font=("Segoe UI", 8)).pack(side=tk.RIGHT)
        self.score_chart = tk.Canvas(chart_card, height=145, bg="#4a5c70",
                                     highlightthickness=0)
        self.score_chart.pack(fill=tk.X, pady=(10, 0))

        right = tk.Frame(body, bg=UI_BG, width=270)
        right.grid(row=0, column=1, sticky="ns")
        right.grid_propagate(False)
        self._metric_card(right, "CURRENT SCORE", "score_label", "-- / 100")
        self._metric_card(right, "EYE SIGNAL", "eye_score_label", "--")
        self._metric_card(right, "MOUTH SIGNAL", "mouth_score_label", "--")
        mouth_card = tk.Frame(right, bg=PANEL_BG, padx=12, pady=12)
        mouth_card.pack(fill=tk.X, pady=(2, 14))
        tk.Label(mouth_card, text="MOUTH CROP", fg=MUTED_COLOR, bg=PANEL_BG,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.mouth_label = tk.Label(mouth_card, bg="#15202b", width=220, height=150)
        self.mouth_label.pack(fill=tk.X, pady=(8, 0))

        self.ear_var = tk.StringVar(value="EAR  --")
        self.eyes_var = tk.StringVar(value="Eyes  --")
        self.mouth_var = tk.StringVar(value="Mouth  --")
        details = tk.Frame(right, bg=PANEL_BG, padx=12, pady=10)
        details.pack(fill=tk.X)
        for var in (self.ear_var, self.eyes_var, self.mouth_var):
            tk.Label(details, textvariable=var, fg=TEXT_COLOR, bg=PANEL_BG,
                     font=("Consolas", 10), anchor="w").pack(fill=tk.X, pady=2)
        self.state_label = tk.Label(details, text="STATUS  --", fg="#58e68a",
                                    bg=PANEL_BG, font=("Segoe UI", 16, "bold"))
        self.state_label.pack(anchor="w", pady=(12, 0))

    def _rail_item(self, parent, text, active=False, command=None):
        button = tk.Button(parent, text=text, command=command, anchor="w", relief=tk.FLAT,
                           bd=0, padx=12, pady=10, bg="#536b82" if active else "#17212d",
                           fg=TEXT_COLOR if active else MUTED_COLOR,
                           activebackground="#637e98", activeforeground=TEXT_COLOR,
                           font=("Segoe UI", 9, "bold"))
        button.pack(fill=tk.X, pady=3)

    def _metric_card(self, parent, title, attribute, initial):
        card = tk.Frame(parent, bg=PANEL_BG, padx=14, pady=12)
        card.pack(fill=tk.X, pady=(0, 10))
        tk.Label(card, text=title, fg=MUTED_COLOR, bg=PANEL_BG,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        label = tk.Label(card, text=initial, fg=TEXT_COLOR, bg=PANEL_BG,
                         font=("Segoe UI", 20, "bold"))
        label.pack(anchor="w", pady=(4, 0))
        setattr(self, attribute, label)

    def open_settings(self):
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return

        window = tk.Toplevel(self.root)
        self.settings_window = window
        window.title("Camera Settings")
        window.configure(bg=UI_BG)
        window.resizable(False, False)
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", lambda: self._close_settings(window))

        content = tk.Frame(window, bg=UI_BG, padx=18, pady=16)
        content.pack(fill=tk.BOTH, expand=True)
        tk.Label(content, text="CAMERA SETTINGS", fg=TEXT_COLOR, bg=UI_BG,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(content, text="Tune the active camera input", fg=MUTED_COLOR, bg=UI_BG,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 14))

        controls = {}
        resolution_var = tk.StringVar(value=f"{self.camera.width} x {self.camera.height}")
        resolution_row = tk.Frame(content, bg=UI_BG)
        resolution_row.pack(fill=tk.X, pady=(0, 10))
        tk.Label(resolution_row, text="Resolution", width=12, anchor="w", fg=TEXT_COLOR,
                 bg=UI_BG, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        resolution_selector = ttk.Combobox(
            resolution_row,
            textvariable=resolution_var,
            values=["640 x 480", "1280 x 720", "1920 x 1080"],
            state="readonly",
            width=16,
        )
        resolution_selector.pack(side=tk.LEFT, fill=tk.X, expand=True)

        properties = (
            ("Exposure", cv2.CAP_PROP_EXPOSURE, -13, 0),
            ("Brightness", cv2.CAP_PROP_BRIGHTNESS, 0, 255),
            ("Contrast", cv2.CAP_PROP_CONTRAST, 0, 255),
            ("Saturation", cv2.CAP_PROP_SATURATION, 0, 255),
            ("Focus", cv2.CAP_PROP_FOCUS, 0, 255),
        )
        for label_text, property_id, minimum, maximum in properties:
            row = tk.Frame(content, bg=UI_BG)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=label_text, width=12, anchor="w", fg=TEXT_COLOR,
                     bg=UI_BG, font=("Segoe UI", 9)).pack(side=tk.LEFT)
            value = tk.StringVar(value=f"{self._camera_property_value(property_id, minimum):.1f}")
            entry = ttk.Entry(row, textvariable=value, width=10, justify="right")
            entry.pack(side=tk.LEFT)
            controls[property_id] = value

        auto_exposure = tk.BooleanVar(value=True)
        autofocus = tk.BooleanVar(value=True)
        tk.Checkbutton(content, text="Auto exposure", variable=auto_exposure,
                       fg=TEXT_COLOR, bg=UI_BG, selectcolor=PANEL_BG,
                       activebackground=UI_BG, activeforeground=TEXT_COLOR).pack(anchor="w", pady=(10, 0))
        tk.Checkbutton(content, text="Autofocus", variable=autofocus,
                       fg=TEXT_COLOR, bg=UI_BG, selectcolor=PANEL_BG,
                       activebackground=UI_BG, activeforeground=TEXT_COLOR).pack(anchor="w")

        actions = tk.Frame(content, bg=UI_BG)
        actions.pack(fill=tk.X, pady=(16, 0))
        ttk.Button(actions, text="Auto optimize",
                   command=lambda: self.auto_optimize(window, resolution_var),
                   style="Modern.TButton").pack(side=tk.LEFT)
        ttk.Button(actions, text="Apply",
                   command=lambda: self.apply_settings(
                       controls, resolution_var.get(), auto_exposure.get(), autofocus.get(), window
                   ),
                   style="Modern.TButton").pack(side=tk.RIGHT)

    def _close_settings(self, window):
        if window.winfo_exists():
            window.destroy()
        self.settings_window = None

    def _camera_property_value(self, property_id, fallback):
        value = self.camera.get_property(property_id)
        return fallback if value is None else value

    def apply_settings(self, controls, resolution, auto_exposure, autofocus, window=None):
        if self.switching:
            return
        try:
            width, height = (int(value.strip()) for value in resolution.split("x"))
            values = {"width": width, "height": height}
        except (ValueError, AttributeError):
            return

        self.camera.set_property(cv2.CAP_PROP_FRAME_WIDTH, values["width"])
        self.camera.set_property(cv2.CAP_PROP_FRAME_HEIGHT, values["height"])
        self.camera.set_property(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto_exposure else 0.25)
        self.camera.set_property(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
        for property_id, value in controls.items():
            try:
                self.camera.set_property(property_id, float(value.get()))
            except ValueError:
                continue
        self.camera.width = values["width"]
        self.camera.height = values["height"]
        if window is not None:
            self._close_settings(window)

    def auto_optimize(self, window=None, resolution_var=None):
        # A smaller capture size leaves enough CPU headroom for live inference.
        self.camera.set_property(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set_property(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set_property(cv2.CAP_PROP_FPS, 30)
        self.camera.set_property(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        self.camera.set_property(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.camera.width = 640
        self.camera.height = 480
        if resolution_var is not None:
            resolution_var.set("640 x 480")
        if window is not None:
            self._close_settings(window)

    def _find_cameras(self):
        available = []
        for index in range(10):
            capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if capture.isOpened():
                ret, _ = capture.read()
                if ret:
                    available.append(index)
            capture.release()
        return available

    def _update_camera_selector(self):
        values = [f"Camera {index}" for index in self.camera_choices]
        self.camera_selector.configure(values=values)
        if self.camera_index in self.camera_choices:
            self.camera_var.set(f"Camera {self.camera_index}")
        elif values:
            self.camera_var.set(values[0])
        else:
            self.camera_var.set("No camera found")

    def refresh_cameras(self):
        if self.switching:
            return
        selected_index = self.camera_index
        self.camera_choices = self._find_cameras()
        self.camera_index = selected_index
        self._update_camera_selector()

    def _camera_dropdown_opened(self):
        self.camera_selector.configure(values=[f"Camera {index}" for index in self.camera_choices])

    def switch_camera(self, _event=None):
        if self.switching:
            return

        selected = self.camera_var.get()
        if not selected.startswith("Camera "):
            return

        new_index = int(selected.split(" ", 1)[1])
        if new_index == self.camera_index and self.camera.is_open:
            return

        previous_camera = self.camera
        was_running = self.running
        self.running = False
        self._stop_capture()
        self.switching = True
        self.camera_selector.configure(state="disabled")
        self.set_status_text(f"SWITCHING TO CAMERA {new_index}...", "#ff9900")

        threading.Thread(
            target=self._open_camera_in_background,
            args=(new_index, previous_camera),
            daemon=True,
        ).start()
        self.root.after(50, self._finish_camera_switch, was_running)

    def _open_camera_in_background(self, camera_index, previous_camera):
        new_camera = CameraManager(
            camera_index=camera_index,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            mirror=True,
            backend=cv2.CAP_DSHOW,
        )
        if not new_camera.open():
            self.camera_result_queue.put((False, camera_index, None, previous_camera))
            return

        previous_camera.release()
        self.camera_result_queue.put((True, camera_index, new_camera, previous_camera))

    def _finish_camera_switch(self, was_running):
        try:
            success, camera_index, new_camera, previous_camera = self.camera_result_queue.get_nowait()
        except queue.Empty:
            if self.switching:
                self.root.after(50, self._finish_camera_switch, was_running)
            return

        self.switching = False
        self.camera_selector.configure(state="readonly")
        if not success:
            self.running = was_running
            self.camera_var.set(f"Camera {self.camera_index}")
            self.set_status_text(f"ERROR: could not read camera {camera_index}", "#ff2222")
            if self.running:
                self._start_capture(previous_camera)
                self.update_loop()
            return

        self.camera = new_camera
        self.camera_index = camera_index
        self._start_capture(self.camera)
        self.set_status_text(f"STATUS: CAMERA {camera_index}", "#00cc44")
        self.running = True
        self.update_loop()

    def set_status_text(self, text, color):
        self.state_label.config(text=text, fg=color)

    def _start_capture(self, camera):
        self._stop_capture()
        self.capture_stop.clear()
        self.capture_thread = threading.Thread(
            target=self._capture_loop, args=(camera,), daemon=True
        )
        self.capture_thread.start()

    def _stop_capture(self):
        self.capture_stop.set()

    def _capture_loop(self, camera):
        while not self.capture_stop.is_set() and camera.is_open:
            ret, frame = camera.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.02)

    def update_loop(self):
        if not self.running:
            self.update_job = None
            return

        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is None:
            self.update_job = self.root.after(20, self.update_loop)
            return

        display = frame.copy()
        now = time.perf_counter()
        self.frame_times.append(now)
        if len(self.frame_times) > 1:
            elapsed = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times) - 1) / elapsed if elapsed > 0 else 0
            self.fps_var.set(f"{fps:.1f} FPS")
        landmarks = self.last_landmarks
        mouth = self.last_mouth
        if now - self.last_detection_time >= 0.12:
            self.last_detection_time = now
            landmarks = detect_face_landmarks(frame)
            mouth = None
            if landmarks is not None:
                h, w = frame.shape[:2]
                eye_data = analyze_eyes(landmarks, w, h)
                self.last_eye_data = eye_data
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
                    self.eye_state, mouth_prediction_history, time.time(),
                    self.fusion_state, eye_event_history,
                    mouth_available=mouth is not None,
                )
            self.last_landmarks = landmarks
            self.last_mouth = mouth

        if landmarks is not None:
            h, w = frame.shape[:2]
            draw_landmarks(display, landmarks, w, h)

            if self.last_eye_data is not None:
                self.ear_var.set(f"EAR: {self.last_eye_data['average_ear']:.3f}")
                self.eyes_var.set(f"Eyes: {self.last_eye_data['combined_state']}")
            if mouth is None:
                self.mouth_var.set("Mouth  ERROR: OUT OF FRAME")
                self.mouth_score_label.config(text="ERR")
            else:
                self.mouth_var.set(
                    f"Mouth  {self.mouth_state['current_prediction']} "
                    f"({self.mouth_state['current_confidence']:.0f}%)"
                )
            self.eye_score_label.config(text=f"{self.fusion_state['eye_score']:.1f}")
            self.mouth_score_label.config(text=f"{self.fusion_state['mouth_score']:.1f}")
            self.score_label.config(text=f"{self.fusion_state['final_score']:.1f} / 100")
            if now - self.last_chart_time >= 0.5:
                self.score_history.append(self.fusion_state["final_score"])
                self.last_chart_time = now
                self._draw_score_chart()

            final_state = self.fusion_state["state"]
            self.state_label.config(
                text=f"STATUS: {final_state}",
                fg=STATE_COLORS[final_state],
            )
            self.score_label.config(
                text=f"{self.fusion_state['final_score']:.1f} / 100"
            )
        else:
            self.ear_var.set("EAR: --")
            self.eyes_var.set("Eyes: FACE NOT DETECTED")
            self.mouth_var.set("Mouth  ERROR: OUT OF FRAME")

        self._show_image(self.camera_label, display, (800, 450), "camera")

        if mouth is not None:
            self._show_image(self.mouth_label, mouth, (256, 256), "mouth")

        self.update_job = self.root.after(10, self.update_loop)

    def _show_image(self, label, bgr_image, size, key):
        available_width = label.winfo_width()
        available_height = label.winfo_height()
        if available_width > 1 and available_height > 1:
            source_height, source_width = bgr_image.shape[:2]
            scale = min(available_width / source_width, available_height / source_height)
            size = (
                max(1, int(source_width * scale)),
                max(1, int(source_height * scale)),
            )

        resized = cv2.resize(bgr_image, size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._photo_refs[key] = photo
        label.config(image=photo)

    def _draw_score_chart(self):
        canvas = self.score_chart
        width = max(canvas.winfo_width(), 320)
        height = 145
        canvas.delete("all")
        canvas.configure(height=height)
        left, top, right, bottom = 8, 12, width - 8, height - 20

        for threshold, color in ((70, "#624452"), (40, "#665a45"), (0, "#3d6657")):
            y_top = bottom - (threshold / 100) * (bottom - top)
            y_bottom = bottom if threshold == 0 else bottom - (threshold / 100) * (bottom - top) + (30 if threshold == 70 else 35)
            if threshold == 70:
                y_bottom = top
            elif threshold == 40:
                y_bottom = bottom - (70 / 100) * (bottom - top)
            canvas.create_rectangle(left, y_top, right, y_bottom, fill=color, outline="")

        for value, label in ((100, "100"), (70, "DROWSY"), (40, "WARNING"), (0, "0")):
            y = bottom - (value / 100) * (bottom - top)
            canvas.create_line(left, y, right, y, fill="#31404a", dash=(2, 4))
            canvas.create_text(right - 2, y - 2, text=label, fill=MUTED_COLOR,
                               anchor="se", font=("Segoe UI", 7))

        values = list(self.score_history)
        if len(values) < 2:
            return
        points = []
        for index, value in enumerate(values):
            x = left + index * (right - left) / max(len(values) - 1, 1)
            y = bottom - max(0, min(100, value)) / 100 * (bottom - top)
            points.extend((x, y))
        color = STATE_COLORS.get(self.fusion_state.get("state"), LILAC_COLOR)
        canvas.create_line(*points, fill=color, width=3, smooth=True)
        canvas.create_oval(points[-2] - 4, points[-1] - 4, points[-2] + 4,
                           points[-1] + 4, fill=color, outline=PANEL_BG, width=2)

    def on_close(self):
        self.running = False
        self._stop_capture()
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
            self.update_job = None
        try:
            self.camera.release()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessUI(root)
    root.mainloop()
