# ============================================================
# MOUTH PREPROCESSING + CONVNEXT FINE-TUNING (YAWDD)
# ============================================================
# Two-stage script:
#
#   prepare : scan YawDD videos -> MediaPipe face landmarks ->
#             mouth crops + MAR -> temporal auto-labeling
#             (mouth open with same open shape for >= YAWN_SECONDS
#              => yawn, else no_yawn) -> dataset folder + manifest
#
#   train   : fine-tune a pretrained ConvNeXt (timm) on the
#             prepared mouth crops (RGB 224x224)
#
# Run mode is configured via the STAGE / KAGGLE / COLAB variables
# in the "RUN MODE" section below - edit them, then run:
#
#   python script.py
# ============================================================

import csv
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# TERMINAL HELPERS - LOUD OUTPUT AS REQUESTED
# ============================================================


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70, flush=True)


def info(msg):
    print(f"[INFO] {msg}", flush=True)


def warn(msg):
    print(f"[WARN] {msg}", flush=True)


def error(msg):
    print(f"[ERROR] {msg}", flush=True)


def stage(msg):
    print(f"\n--- {msg} " + "-" * max(0, 60 - len(msg)), flush=True)


# ============================================================
# RUN MODE - EDIT THESE VARIABLES
# ============================================================

# "prepare" | "train" | "all"
STAGE = "prepare"

KAGGLE = False   # True -> use Kaggle paths (/kaggle/input, /kaggle/working)
COLAB = False    # True -> use Colab paths (/content)

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg")

# Frame sampling: process every Nth frame (6 fps at 30 fps source).
# Enough resolution for a 4-second temporal rule, 5x cheaper.
FRAME_STRIDE = 5

# MediaPipe Face Landmarker model.
# On Kaggle/Colab it is downloaded automatically; locally it should
# point at the repo's assets/face_landmarker.task.
LOCAL_LANDMARKER_PATH = Path(__file__).resolve().parents[2] / "assets" / "face_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Mouth landmark indices (inner lips + corners) for MAR.
MAR_TOP = 13
MAR_BOTTOM = 14
MAR_LEFT = 61
MAR_RIGHT = 291

# All mouth-region landmarks used for the crop (same as notebook 02).
MOUTH_POINTS = [
    61, 146, 91, 181, 84,
    17, 314, 405, 321, 375,
    291, 409, 270, 269, 267,
    0, 37, 39, 40, 185,
]

CROP_SIZE = 224           # ConvNeXt input
CROP_PAD_X = 0.30
CROP_PAD_Y = 0.40

# Temporal auto-labeling rule.
MAR_OPEN_THRESHOLD = 0.55   # frame counts as "mouth open" above this
MAR_PEAK_THRESHOLD = 0.75   # a yawn must peak above this
YAWN_SECONDS = 4.0          # sustained open duration required for a yawn
SMOOTH_WINDOW = 7           # median filter window over MAR signal (frames)

CLASSES = ["no_yawn", "yawn"]

# Local paths (used when KAGGLE = False and COLAB = False).
# Leave DATA_DIR as None when running on Kaggle/Colab.
DATA_DIR = None
OUT_DIR = None

MIN_CROP_PIXELS = 24        # discard degenerate crops

# Training configuration (Kaggle quota friendly).
MODEL_NAME = "convnext_nano.in12k_ft_in1k"   # pretrained ConvNeXt Nano
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS_HEAD = 4          # frozen backbone, train head only
EPOCHS_FULL = 16         # unfreeze last stages, low LR
HEAD_LR = 1e-3
FULL_LR = 2e-5
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
EARLY_STOP_PATIENCE = 5
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 2

# ============================================================
# PATH RESOLUTION (local / kaggle / colab)
# ============================================================


def resolve_paths():
    if KAGGLE:
        data_dir = Path("/kaggle/input/yawdd")
        work_dir = Path("/kaggle/working")
    elif COLAB:
        data_dir = Path("/content/YawDD")
        work_dir = Path("/content")
    else:
        data_dir = DATA_DIR if DATA_DIR else None
        work_dir = Path(OUT_DIR) if OUT_DIR else Path(__file__).resolve().parent / "output"

    if data_dir is not None and not data_dir.exists():
        # Search Kaggle input mounts recursively for a yawdd folder.
        found = find_yawdd_root(Path("/kaggle/input"))
        if found is not None:
            data_dir = found

    return data_dir, work_dir


def find_yawdd_root(mount):
    """Locate the YawDD root inside a mount, searching recursively."""
    if not mount.exists():
        return None
    for dirpath, dirnames, _ in os.walk(mount):
        for name in dirnames:
            if "yawdd" in name.lower():
                return Path(dirpath) / name
    return None


# ============================================================
# MEDIAPIPE LANDMARKER
# ============================================================


def get_landmarker_path(work_dir):
    path = work_dir / "face_landmarker.task"
    if path.exists():
        info(f"Landmarker model found: {path}")
        return path

    if LOCAL_LANDMARKER_PATH.exists():
        info(f"Using repo landmarker: {LOCAL_LANDMARKER_PATH}")
        return LOCAL_LANDMARKER_PATH

    info(f"Downloading Face Landmarker from {LANDMARKER_URL} ...")
    import urllib.request
    work_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(LANDMARKER_URL, path)
    info(f"Downloaded to {path}")
    return path


def create_landmarker(model_path):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    info("MediaPipe Face Landmarker loaded")
    return landmarker


# ============================================================
# MOUTH CROPPING + MAR
# ============================================================


def compute_mar(landmarks):
    """Mouth Aspect Ratio from inner-lip vertical gap / corner distance."""
    top = np.array([landmarks[MAR_TOP].x, landmarks[MAR_TOP].y])
    bottom = np.array([landmarks[MAR_BOTTOM].x, landmarks[MAR_BOTTOM].y])
    left = np.array([landmarks[MAR_LEFT].x, landmarks[MAR_LEFT].y])
    right = np.array([landmarks[MAR_RIGHT].x, landmarks[MAR_RIGHT].y])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    if horizontal < 1e-6:
        return 0.0
    return float(vertical / horizontal)


def crop_mouth(frame, landmarks):
    h, w = frame.shape[:2]

    points = []
    for idx in MOUTH_POINTS:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        points.append([x, y])

    points = np.array(points, dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(points)

    pad_x = int(bw * CROP_PAD_X)
    pad_y = int(bh * CROP_PAD_Y)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    if x2 - x1 < MIN_CROP_PIXELS or y2 - y1 < MIN_CROP_PIXELS:
        return None

    mouth = frame[y1:y2, x1:x2]
    if mouth.size == 0:
        return None

    return cv2.resize(mouth, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


# ============================================================
# STAGE 1 - PREPARE
# ============================================================


def median_filter(signal, window):
    if len(signal) < window:
        window = max(1, len(signal))
    if window <= 1:
        return signal[:]
    pad = window // 2
    padded = [signal[0]] * pad + list(signal) + [signal[-1]] * pad
    out = []
    for i in range(len(signal)):
        chunk = sorted(padded[i:i + window])
        out.append(chunk[window // 2])
    return out


def find_videos(root):
    videos = []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(VIDEO_EXTENSIONS):
                videos.append(Path(dirpath) / name)
    return sorted(videos)


def prepare_dataset(data_dir, work_dir):
    banner("STAGE 1 - PREPARE DATASET")

    if data_dir is None or not Path(data_dir).exists():
        error(f"YawDD data directory not found: {data_dir}")
        error("Set DATA_DIR in this file, or set KAGGLE/COLAB = True "
              "with the YawDD dataset attached.")
        sys.exit(1)

    out_dir = work_dir / "mouth_dataset"
    for cls in CLASSES:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)
    info(f"Output dataset: {out_dir}")

    landmarker_path = get_landmarker_path(work_dir)
    landmarker = create_landmarker(landmarker_path)

    videos = find_videos(data_dir)
    info(f"Found {len(videos)} videos in {data_dir}")
    for i, v in enumerate(videos):
        print(f"  [{i}] {v.relative_to(data_dir)}")

    manifest_path = out_dir / "manifest.csv"
    manifest = csv.writer(open(manifest_path, "w", newline="", encoding="utf-8"))
    manifest.writerow(["crop_path", "source", "frame_index", "fps", "mar", "label"])

    stats = {"yawn": 0, "no_yawn": 0, "no_face": 0, "no_crop": 0}
    t_start = time.time()

    for vi, video_path in enumerate(videos):
        rel_name = video_path.relative_to(data_dir)
        safe_stem = "_".join(video_path.stem.split())
        print(f"\n[{vi + 1}/{len(videos)}] {rel_name}", flush=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            warn(f"  Could not open, skipping")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps <= 0:
            fps = 30.0
            warn("  fps not reported by container, assuming 30.0")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info(f"  fps={fps:.1f} frames={total_frames} stride={FRAME_STRIDE}")

        # ---- Pass 1: sample frames, compute MAR, keep landmarks ----
        import mediapipe as mp

        frame_indices = []
        mar_signal = []
        landmarks_per_frame = {}

        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if fi % FRAME_STRIDE == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)

                if result.face_landmarks:
                    lm = result.face_landmarks[0]
                    mar = compute_mar(lm)
                    landmarks_per_frame[fi] = lm
                    frame_indices.append(fi)
                    mar_signal.append(mar)
                else:
                    stats["no_face"] += 1

            fi += 1
            if fi % 1500 == 0:
                print(f"    ... frame {fi}/{total_frames}", flush=True)

        cap.release()

        if not mar_signal:
            warn("  No faces detected in entire video, skipping")
            continue

        # ---- Pass 2: temporal labeling ----
        smoothed = median_filter(mar_signal, SMOOTH_WINDOW)
        sample_fps = fps / FRAME_STRIDE
        sustained_frames = int(math.ceil(YAWN_SECONDS * sample_fps))

        labels = ["no_yawn"] * len(smoothed)
        open_run = 0
        run_peak = 0.0
        run_start = 0
        yawn_events = 0

        for i, mar in enumerate(smoothed):
            if mar >= MAR_OPEN_THRESHOLD:
                if open_run == 0:
                    run_start = i
                open_run += 1
                run_peak = max(run_peak, mar)
            else:
                if open_run >= sustained_frames and run_peak >= MAR_PEAK_THRESHOLD:
                    yawn_events += 1
                    for j in range(run_start, i):
                        labels[j] = "yawn"
                open_run = 0
                run_peak = 0.0

        # trailing run
        if open_run >= sustained_frames and run_peak >= MAR_PEAK_THRESHOLD:
            yawn_events += 1
            for j in range(run_start, len(smoothed)):
                labels[j] = "yawn"

        print(f"  MAR: min={min(smoothed):.2f} max={max(smoothed):.2f} "
              f"mean={np.mean(smoothed):.2f}")
        print(f"  Yawn events detected: {yawn_events} "
              f"(>= {YAWN_SECONDS:.0f}s open, peak MAR >= {MAR_PEAK_THRESHOLD})")

        # ---- Pass 3: save crops (sequential read - seeking is slow) ----
        cap = cv2.VideoCapture(str(video_path))
        saved = {"yawn": 0, "no_yawn": 0}

        wanted = dict(zip(frame_indices, range(len(frame_indices))))
        last_wanted = max(frame_indices)
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if fi in wanted:
                i = wanted[fi]
                label = labels[i]
                lm = landmarks_per_frame[fi]
                mouth = crop_mouth(frame, lm)

                if mouth is not None:
                    filename = f"{safe_stem}_f{fi:06d}.jpg"
                    crop_path = out_dir / label / filename
                    cv2.imwrite(str(crop_path), mouth)
                    manifest.writerow([str(crop_path), str(rel_name), fi,
                                       f"{fps:.2f}", f"{mar_signal[i]:.3f}", label])
                    saved[label] += 1
                    stats[label] += 1
                else:
                    stats["no_crop"] += 1

            fi += 1
            if fi > last_wanted:
                break

        cap.release()
        print(f"  Saved: yawn={saved['yawn']} no_yawn={saved['no_yawn']}", flush=True)

    elapsed = time.time() - t_start
    banner("PREPARE COMPLETE")
    print(f"Videos processed : {len(videos)}")
    print(f"Yawn crops       : {stats['yawn']:,}")
    print(f"No-yawn crops    : {stats['no_yawn']:,}")
    print(f"No-face frames   : {stats['no_face']:,}")
    print(f"Bad crops        : {stats['no_crop']:,}")
    print(f"Elapsed          : {elapsed / 60:.1f} min")
    print(f"Manifest         : {manifest_path}")
    print("=" * 70)

    return out_dir


# ============================================================
# STAGE 2 - TRAIN (ConvNeXt fine-tuning)
# ============================================================


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    info(f"Seeded everything with {seed}")


def build_loaders(data_dir):
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from sklearn.model_selection import train_test_split

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),          # mouth is symmetric
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full = datasets.ImageFolder(str(data_dir), transform=train_tfms)
    classes = full.classes
    info(f"Classes: {classes}")
    for c in classes:
        n = len((data_dir / c).glob("*"))
        print(f"  {c}: {n:,} images")

    targets = full.targets
    idx = list(range(len(full)))
    train_idx, temp_idx = train_test_split(idx, test_size=VAL_SPLIT + TEST_SPLIT,
                                           stratify=targets, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx,
                                         test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
                                         stratify=[targets[i] for i in temp_idx],
                                         random_state=SEED)
    info(f"Split: train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    train_ds = Subset(full, train_idx)
    val_ds = Subset(datasets.ImageFolder(str(data_dir), transform=eval_tfms), val_idx)
    test_ds = Subset(datasets.ImageFolder(str(data_dir), transform=eval_tfms), test_idx)

    loaders = {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            drop_last=True, persistent_workers=NUM_WORKERS > 0),
        "val": DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True),
    }
    return loaders, classes


def build_model(num_classes, device):
    import torch
    import timm

    info(f"Creating {MODEL_NAME} with ImageNet pretrained weights...")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    info(f"Parameters: {n_params / 1e6:.1f}M | device: {device}")
    return model


def set_backbone_frozen(model, frozen):
    """frozen=True: train classifier head only.
    frozen=False: also unfreeze the last stage for fine-tuning."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True

    if not frozen:
        # unfreeze the last stage (stage4 / blocks[-1]) for full fine-tune
        if hasattr(model, "stages"):
            for p in model.stages[-1].parameters():
                p.requires_grad = True
            norm = getattr(model, "norm", None)
            if norm is not None:
                for p in norm.parameters():
                    p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info(f"Trainable parameters: {trainable / 1e6:.1f}M ({'head only' if frozen else 'head + last stage'})")


def run_epoch(model, loader, criterion, optimizer, device, phase, scaler):
    import torch

    model.train() if phase == "train" else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for bi, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if phase == "train":
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if phase == "train":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        if phase == "train" and (bi + 1) % 20 == 0:
            print(f"    batch {bi + 1}/{len(loader)} "
                  f"loss={loss.item():.4f} acc={100 * correct / total:.1f}%", flush=True)

    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def train(data_dir, work_dir):
    banner("STAGE 2 - TRAIN CONVNEXT")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import classification_report, confusion_matrix

    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        warn("No GPU found - training will be extremely slow. Set KAGGLE = True!")

    out_dir = work_dir / "runs" / time.strftime("exp_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    info(f"Run directory: {out_dir}")

    loaders, classes = build_loaders(Path(data_dir))
    model = build_model(len(classes), device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    best_val = 0.0
    patience_left = EARLY_STOP_PATIENCE
    best_path = out_dir / "convnext_mouth_best.pth"

    # ---- Phase A: head only ----
    banner("PHASE A - LINEAR PROBE (backbone frozen)")
    set_backbone_frozen(model, frozen=True)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=HEAD_LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS_HEAD):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion,
                                    optimizer, device, "train", scaler)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion,
                                    optimizer, device, "eval", scaler)
        print(f"[HEAD {epoch + 1}/{EPOCHS_HEAD}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.2f}% | "
              f"val_loss={va_loss:.4f} acc={va_acc:.2f}% | "
              f"{time.time() - t0:.0f}s", flush=True)
        history.append({"phase": "head", "epoch": epoch + 1,
                        "train_loss": tr_loss, "val_loss": va_loss,
                        "train_acc": tr_acc, "val_acc": va_acc})

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), best_path)
            info(f"  new best val acc: {best_val:.2f}% -> saved")

    # ---- Phase B: unfreeze last stage ----
    banner("PHASE B - FINE-TUNE (last stage unfrozen)")
    set_backbone_frozen(model, frozen=False)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=FULL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FULL)

    for epoch in range(EPOCHS_FULL):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion,
                                    optimizer, device, "train", scaler)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion,
                                    optimizer, device, "eval", scaler)
        scheduler.step()

        print(f"[FULL {epoch + 1}/{EPOCHS_FULL}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.2f}% | "
              f"val_loss={va_loss:.4f} acc={va_acc:.2f}% | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | "
              f"{time.time() - t0:.0f}s", flush=True)
        history.append({"phase": "full", "epoch": epoch + 1,
                        "train_loss": tr_loss, "val_loss": va_loss,
                        "train_acc": tr_acc, "val_acc": va_acc})

        if va_acc > best_val:
            best_val = va_acc
            patience_left = EARLY_STOP_PATIENCE
            torch.save(model.state_dict(), best_path)
            info(f"  new best val acc: {best_val:.2f}% -> saved")
        else:
            patience_left -= 1
            info(f"  no improvement ({patience_left} early-stop patience left)")
            if patience_left <= 0:
                warn("Early stopping triggered")
                break

    # ---- Test ----
    banner("TEST EVALUATION")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loaders["test"]:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    test_acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nTest Accuracy: {test_acc:.2f}%\n")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_labels, all_preds))

    torch.save(model.state_dict(), out_dir / "convnext_mouth_final.pth")
    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "epoch", "train_loss",
                                               "val_loss", "train_acc", "val_acc"])
        writer.writeheader()
        writer.writerows(history)

    banner("TRAIN COMPLETE")
    print(f"Best val accuracy : {best_val:.2f}%")
    print(f"Test accuracy     : {test_acc:.2f}%")
    print(f"Best weights      : {best_path}")
    print(f"Run directory     : {out_dir}")
    print("Download convnext_mouth_best.pth and drop it in models/ on your laptop.")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================


def main():
    banner("MOUTH PREPROC - CONVNEXT PIPELINE")
    print(f"Stage      : {STAGE}")
    print(f"Model      : {MODEL_NAME}")
    print(f"Input      : RGB {IMAGE_SIZE}x{IMAGE_SIZE} mouth crops only")
    print(f"Yawn rule  : mouth open >= {YAWN_SECONDS:.0f}s "
          f"(MAR >= {MAR_OPEN_THRESHOLD}, peak >= {MAR_PEAK_THRESHOLD})")
    print(f"Mode       : kaggle={KAGGLE} colab={COLAB}")

    data_dir, work_dir = resolve_paths()
    info(f"Data dir   : {data_dir}")
    info(f"Work dir   : {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    if STAGE in ("prepare", "all"):
        data_dir = prepare_dataset(data_dir, work_dir)
    else:
        data_dir = data_dir / "mouth_dataset" if data_dir else None

    if STAGE in ("train", "all"):
        if KAGGLE:
            data_dir = Path("/kaggle/working/mouth_dataset")
        elif COLAB:
            data_dir = Path("/content/mouth_dataset")
        if data_dir is None or not Path(data_dir).exists():
            error(f"Dataset not found at {data_dir} - run prepare first.")
            sys.exit(1)
        train(data_dir, work_dir)


if __name__ == "__main__":
    main()

