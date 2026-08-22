# Drowsiness Detection ML Project

This project contains the separate pieces of a webcam-based driver drowsiness detection workflow. The trained mouth-state CNN and eye-based EAR analysis are available, while the final integrated drowsiness detector is still under development.

## Workflow

1. Dataset Preparation: YawDD preparation and frame labeling are preserved in `archive/notebooks/`.
2. Mouth Preprocessing: `02_mouth_preprocessing.ipynb` detects faces, extracts mouth landmarks, crops mouths, resizes them to 128 x 128, and creates the mouth dataset.
3. Mouth CNN Training: `03_mouth_cnn_training.ipynb` handles splitting, CNN training, checkpointing, evaluation, confusion matrices, reports, and curves.
4. Live Mouth Inference: `archive/notebooks/live_mouth_inference.ipynb` contains the current standalone mouth inference workflow.
5. Eye Analysis: `04_eye_analysis.ipynb` covers MediaPipe eye landmarks, EAR, eye open/closed classification, closure duration, blink analysis, and eye-based drowsiness scoring.
6. Future Integrated Drowsiness System: eye and mouth signals will be combined in a later stage.

## Project Structure

```text
01_dataset_preparation.ipynb  # Missing; no replacement was invented
02_mouth_preprocessing.ipynb
03_mouth_cnn_training.ipynb
04_eye_analysis.ipynb
archive/notebooks/             # Experimental and live-inference notebooks
models/mouth_cnn_best.pth
models/archive/                # Older duplicate model files
assets/face_landmarker.task
docs/                          # Reference paper
frames/                        # Local dataset and generated metadata
```

The large image dataset is intentionally excluded from this Git repository. Supply the existing `frames/dataset` and `frames/mouth_dataset` directories separately before running the notebooks. The dataset was not moved, deleted, renamed, or modified during this cleanup.

## Current Status

At this stage the project contains:

- a trained mouth-state CNN
- standalone live mouth inference
- eye-based EAR analysis
- eye closure analysis

The complete integrated drowsiness detector is not finished yet.