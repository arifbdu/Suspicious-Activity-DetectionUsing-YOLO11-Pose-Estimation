from flask import Flask, render_template, request, Response, redirect, url_for
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------------
# Load YOLO Pose + XGBoost Model
# -----------------------------
pose_model = YOLO("yolo11s-pose.pt")      # YOLOv11 pose model
xgb_model = xgb.Booster()
xgb_model.load_model("trained_model.json")

uploaded_video_path = None

# -----------------------------
# Extract keypoints (17 → 34 features)
# -----------------------------
def extract_keypoints_yolo(results):
    all_kp = []
    all_boxes = []

    for res in results:
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else []
        keypoints = res.keypoints.xyn.cpu().numpy() if len(res.keypoints) > 0 else []

        for i in range(len(keypoints)):
            kp = keypoints[i]
            flat_kp = []
            for j in range(17):
                flat_kp.append(kp[j][0])  # x
                flat_kp.append(kp[j][1])  # y

            all_kp.append(flat_kp)

        for box in boxes:
            all_boxes.append(box)

    return all_kp, all_boxes

# -----------------------------
# Stream video frames
# -----------------------------
def generate_frames():
    global uploaded_video_path
    cap = cv2.VideoCapture(uploaded_video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1018, 600))

        results = pose_model(frame, verbose=False)
        annotated_frame = frame.copy()

        keypoints_list, boxes = extract_keypoints_yolo(results)

        for idx, kp in enumerate(keypoints_list):
            df = pd.DataFrame([kp], columns=[f"p{i}" for i in range(34)])
            dmat = xgb.DMatrix(df)
            pred = xgb_model.predict(dmat)
            binary = int(pred > 0.5)

            x1, y1, x2, y2 = map(int, boxes[idx])

            # Suspicious
            if binary == 0:
                color = (0, 0, 255)
                label = "Suspicious"
            else:
                color = (0, 255, 0)
                label = "Normal"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    global uploaded_video_path

    if request.method == "POST":
        video = request.files["video"]
        if video.filename == "":
            return redirect(request.url)

        uploaded_video_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded_video.mp4")
        video.save(uploaded_video_path)

        return redirect(url_for("stream_page"))

    return render_template("index.html")

@app.route("/stream_page")
def stream_page():
    return render_template("stream.html")

@app.route("/stream")
def stream():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
