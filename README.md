# Proctor-AI — YOLO11-Pose + Standard KNN

A single-camera, upload-based examination behaviour **flagging** application. It recreates the black/green UI shown in Figure 8 of the accompanying research paper. Flags are not proof of misconduct; review is required.

## What is included

- `app.py`: Flask upload, job status, live MJPEG preview, playback and download.
- `detector.py`: YOLO11-Pose inference, 17 pose landmarks, 34 coordinate input features, trained KNN predictions, annotated MP4 generation.
- `models/yolo11n-pose.pt`: original supplied YOLO11n-Pose weights.
- `models/best_model.pkl`: original supplied fitted KNN pipeline (standard KNN k=1, with the paper's internal feature-selection/scaling pipeline).
- `templates/index.html`, `static/`: Figure 8-style interface.
- `run.bat`: normal/full-frame launch (Windows); `run_fast.bat`: optional quicker mode (less complete temporal coverage).
- `requirements.txt`: dependencies; `check_setup.py`: skip pip when packages already exist.

## Getting started (Windows)

1. Install **64-bit Python 3.10 or 3.11** from <https://www.python.org/downloads/> if not installed; during installation select **Add Python to PATH**.
2. Download the repository, extract it, and open the `Proctor-AI` folder.
3. Double-click **`run.bat`**. On first run it installs required missing packages into your existing Python installation. Later runs skip pip if all imports are present.
4. Open <http://127.0.0.1:5000> in Chrome or Edge.
5. Choose an MP4/MOV/AVI video, click **Upload & Start Detection**, and observe the progressive live frame preview. After processing, the annotated video plays in the same page and can be downloaded.
6. Stop the server with **Ctrl+C** in the terminal.

**No virtual environment is created or required.** You can still choose one if you prefer isolation. Installing globally can affect other Python projects; using a dedicated Python installation avoids conflicts.

### Faster optional mode

Double-click `run_fast.bat` instead. It processes only one of every three frames and uses a smaller inference resolution, which can miss brief behaviours and alter detection outcomes. **Use `run.bat` when comparing to the paper.** GPU auto-detection is enabled if your PyTorch install supports CUDA; otherwise it runs on CPU.

### macOS or Linux

```bash
python3 -m pip install -r requirements.txt
python3 app.py
```

The `python3 -m pip install -r requirements.txt` operation does not reinstall already-satisfied packages; run it once initially and when dependencies change. Files use automatic relative paths, so there is no manual configuration of model paths.

## Model and paper consistency

This distribution preserves the **original uploaded `.pt` and `.pkl` files without retraining, ONNX replacement or conversions**. YOLO11-Pose estimates 17 skeletal landmarks, producing ordered columns `x0,y0,...,x16,y16`; the saved KNN pipeline performs its own trained selection/scaling. Class 0 is Normal and class 1 is Suspicious. Do not normalize a second time without verifying how the training data were constructed.

Paper reports **3,414 labeled frames**, **2,170 Normal / 1,244 Suspicious**, stratified 80/20 frame split, 5-fold CV, 30 selected input features and standard KNN **k=1**. Reported held-out results: accuracy 99.56%, precision 99.20%, recall 99.60%, F1 99.40%, AUC 0.9957; confusion matrix [[432,2],[1,248]]. Those are *paper experiment results*, **not measured live-video accuracy**. The frame-level split can share subjects/recordings across train and test and does not establish independent real-world performance. The text of the paper also describes per-person min–max normalization, whereas this runtime passes YOLO-provided image-normalized coordinates; strict reproducibility requires confirming the actual training preprocessing.

Default run analyses every frame with inference size 640 and auto GPU/CPU selection. Since live footage, webcam angle and lighting differ from training data, no claim of 100% detection is made. The screen preview is progressive **after upload** and can run slower than source-video playback on CPU. It is not direct webcam capture.

## Project tree

```text
Proctor-AI/
  app.py
  detector.py
  check_setup.py
  run.bat
  run_fast.bat
  requirements.txt
  README.md
  models/
    best_model.pkl
    yolo11n-pose.pt
  templates/index.html
  static/app.js
  static/style.css
  uploads/.gitkeep
  outputs/.gitkeep
```

## Practical notes

- PyTorch is a **required dependency** for the original `.pt` weights. Avoiding `.venv` removes a *duplicate environment*, but it does **not** remove PyTorch's disk footprint. No claim of sub-1-GB full installation is possible for every computer.
- For CUDA, a CUDA-enabled PyTorch installation matching your NVIDIA driver is required. A CPU-only PyTorch install will work but be slower.
- A `404` for `/favicon.ico` is cosmetic and does not stop detection.
- Uploaded video is removed after processing. Processed videos in `outputs/` remain on the local machine until you delete them. Do not publish actual student footage without permission.
- Only load `.pkl` model files from sources you trust. Python pickles can execute code during loading.
- `app.py` launches Flask's development server, intended for local testing rather than a public production deployment. A production service needs authentication, rate limits, HTTPS and appropriate data retention controls.
- The source archive deliberately excludes example student videos to keep the public repo small and avoid publishing identifiable exam footage.
