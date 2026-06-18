"""
yoga_webcam.py — Real-time Yoga Pose Detection via Webcam (CNN / MobileNetV2)

Usage:
    python yoga_webcam.py
    python yoga_webcam.py --model yoga_cnn_model.h5 --encoder label_encoder_cnn.pkl
    python yoga_webcam.py --camera 1          # use external camera

Controls:
    Q  — quit
    S  — save current frame as screenshot
"""

import argparse
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np

# Lazy-import TensorFlow so error messages are clear
try:
    import tensorflow as tf
except ImportError:
    sys.exit("[ERROR] TensorFlow not found. Run: pip install tensorflow")

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
SMOOTH_N   = 5          # Majority-vote buffer size
PREDICT_EVERY = 3       # Run CNN every N frames for smooth UI

QUALITY_COLOR = {
    "good": (0,  200,  60),   # Green  (BGR)
    "avg" : (0,  200, 220),   # Yellow (BGR)
    "poor": (0,   50, 220),   # Red    (BGR)
}

TIPS = {
    "balasana": {
        "good": "Great form! Keep hips on heels.",
        "avg" : "Lower forehead closer to the mat.",
        "poor": "Hips should rest on heels, arms extended forward.",
    },
    "bhujangasana": {
        "good": "Excellent! Elbows slightly bent.",
        "avg" : "Open your chest more, lift higher.",
        "poor": "Keep hips on floor, elbows close to body.",
    },
    "padmasana": {
        "good": "Beautiful lotus! Spine is tall.",
        "avg" : "Try to get both feet on thighs.",
        "poor": "Sit on a cushion to help hip flexibility.",
    },
    "parvatasana": {
        "good": "Arms perfectly overhead!",
        "avg" : "Straighten arms fully above head.",
        "poor": "Sit tall, raise arms parallel.",
    },
    "tadasana": {
        "good": "Mountain pose is solid!",
        "avg" : "Stand taller, feet together.",
        "poor": "Stand straight, arms at sides, feet together.",
    },
    "trikonasana": {
        "good": "Triangle is perfect!",
        "avg" : "Extend your top arm higher.",
        "poor": "Keep legs straight, reach sideways more.",
    },
    "vrikshasana": {
        "good": "Excellent balance!",
        "avg" : "Raise hands above head and join palms.",
        "poor": "Fix gaze on a point, press foot to inner thigh.",
    },
}


# ── Image pre-processing ──────────────────────────────────────────────────────
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    return resized.astype(np.float32) / 255.0


# ── HUD overlay ───────────────────────────────────────────────────────────────
def draw_overlay(frame, pose: str, quality: str, confidence: float,
                 tip: str, color: tuple) -> np.ndarray:
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 145), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Pose: {pose.upper()}",
                (14, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Quality: {quality.upper()}   Confidence: {confidence:.0%}",
                (14, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.76, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Tip: {tip}",
                (14, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 210, 210), 1, cv2.LINE_AA)

    # Quality bar at bottom
    cv2.rectangle(frame, (0, h - 8), (w, h), color, -1)
    cv2.putText(frame, "Q to quit  |  S to screenshot",
                (w - 290, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return frame


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Real-time Yoga Pose Detection")
    parser.add_argument("--model",   default="yoga_cnn_model.h5",      help="Keras .h5 model path")
    parser.add_argument("--encoder", default="label_encoder_cnn.pkl",  help="LabelEncoder .pkl path")
    parser.add_argument("--camera",  type=int, default=0,              help="Camera index (default 0)")
    args = parser.parse_args()

    # Load model
    model_path   = Path(args.model)
    encoder_path = Path(args.encoder)
    for p in [model_path, encoder_path]:
        if not p.exists():
            sys.exit(f"[ERROR] File not found: {p}\n"
                     "        Train the model first by running yoga-cnn.ipynb.")

    print("Loading model …")
    model = tf.keras.models.load_model(str(model_path))
    le    = joblib.load(str(encoder_path))
    print(f"Model loaded — {len(le.classes_)} classes\n")

    # Open webcam
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera index {args.camera}. "
                 "Try --camera 1 for an external webcam.")

    print("Webcam opened — press Q to quit, S to save screenshot.\n")

    pred_buffer  = deque(maxlen=SMOOTH_N)
    frame_count  = 0
    last_pose    = "detecting"
    last_quality = ""
    last_conf    = 0.0
    last_color   = (200, 200, 200)
    last_tip     = "Getting ready …"
    screenshot_n = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed — retrying …")
            continue

        frame = cv2.flip(frame, 1)   # mirror

        if frame_count % PREDICT_EVERY == 0:
            tensor = preprocess_frame(frame)[np.newaxis]
            probs  = model.predict(tensor, verbose=0)[0]
            idx    = int(np.argmax(probs))
            pred_buffer.append(idx)

            smooth_idx   = Counter(pred_buffer).most_common(1)[0][0]
            label_str    = le.classes_[smooth_idx]
            parts        = label_str.rsplit("_", 1)
            last_pose    = parts[0] if len(parts) == 2 else label_str
            last_quality = parts[1] if len(parts) == 2 else ""
            last_conf    = float(probs[smooth_idx])
            last_color   = QUALITY_COLOR.get(last_quality, (200, 200, 200))
            last_tip     = TIPS.get(last_pose, {}).get(last_quality, "Keep practising!")

        frame = draw_overlay(frame, last_pose, last_quality,
                             last_conf, last_tip, last_color)

        cv2.imshow("Yoga Pose Detection — CNN", frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"screenshot_{screenshot_n:03d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Screenshot saved: {fname}")
            screenshot_n += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")


if __name__ == "__main__":
    main()
