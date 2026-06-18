"""
predict.py — Run Yoga Pose prediction on a single image or video file.

Usage:
    python predict.py --input path/to/image.jpg
    python predict.py --input path/to/video.mp4
    python predict.py --input image.jpg --model yoga_cnn_model.h5 --encoder label_encoder_cnn.pkl
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    sys.exit("[ERROR] TensorFlow not found. Run: pip install tensorflow")

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)

TIPS = {
    "balasana":     {"good": "Great form! Keep hips on heels.",
                     "avg" : "Lower forehead closer to the mat.",
                     "poor": "Hips should rest on heels, arms extended forward."},
    "bhujangasana": {"good": "Excellent! Elbows slightly bent.",
                     "avg" : "Open your chest more, lift higher.",
                     "poor": "Keep hips on floor, elbows close to body."},
    "padmasana":    {"good": "Beautiful lotus! Spine is tall.",
                     "avg" : "Try to get both feet on thighs.",
                     "poor": "Sit on a cushion to help hip flexibility."},
    "parvatasana":  {"good": "Arms perfectly overhead!",
                     "avg" : "Straighten arms fully above head.",
                     "poor": "Sit tall, raise arms parallel."},
    "tadasana":     {"good": "Mountain pose is solid!",
                     "avg" : "Stand taller, feet together.",
                     "poor": "Stand straight, arms at sides, feet together."},
    "trikonasana":  {"good": "Triangle is perfect!",
                     "avg" : "Extend your top arm higher.",
                     "poor": "Keep legs straight, reach sideways more."},
    "vrikshasana":  {"good": "Excellent balance!",
                     "avg" : "Raise hands above head and join palms.",
                     "poor": "Fix gaze on a point, press foot to inner thigh."},
}


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    return resized.astype(np.float32) / 255.0


def predict_frame(frame, model, le):
    """Return (label, confidence) for a single BGR frame."""
    tensor = preprocess_frame(frame)[np.newaxis]
    probs  = model.predict(tensor, verbose=0)[0]
    idx    = int(np.argmax(probs))
    return le.classes_[idx], float(probs[idx])


def predict_image_file(path: Path, model, le):
    frame = cv2.imread(str(path))
    if frame is None:
        sys.exit(f"[ERROR] Cannot read image: {path}")
    label, conf = predict_frame(frame, model, le)
    return label, conf, 1


def predict_video_file(path: Path, model, le, every_n=3):
    cap    = cv2.VideoCapture(str(path))
    votes  = []
    confs  = {}
    n      = 0
    idx_n  = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx_n % every_n == 0:
            label, conf = predict_frame(frame, model, le)
            votes.append(label)
            confs.setdefault(label, []).append(conf)
            n += 1
        idx_n += 1
    cap.release()
    if not votes:
        return None, 0.0, 0
    best = Counter(votes).most_common(1)[0][0]
    avg_conf = round(sum(confs[best]) / len(confs[best]), 3)
    return best, avg_conf, n


def format_result(label, conf, frames, src_type):
    parts   = label.rsplit("_", 1) if label else ["unknown", ""]
    pose    = parts[0]
    quality = parts[1] if len(parts) == 2 else ""
    tip     = TIPS.get(pose, {}).get(quality, "Keep practising!")

    print("\n" + "=" * 52)
    print(f"  {'Image' if src_type == 'image' else 'Video':<12}: {''}")
    print(f"  {'Pose':<12}: {pose.capitalize()}")
    print(f"  {'Quality':<12}: {quality.capitalize()}")
    print(f"  {'Confidence':<12}: {conf:.1%}")
    if src_type == "video":
        print(f"  {'Frames used':<12}: {frames}")
    print(f"\n  Tip: {tip}")
    print("=" * 52 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Yoga Pose CNN Inference")
    parser.add_argument("--input",   required=True,                        help="Image or video file path")
    parser.add_argument("--model",   default="yoga_cnn_model.h5",          help="Keras .h5 model")
    parser.add_argument("--encoder", default="label_encoder_cnn.pkl",      help="LabelEncoder .pkl")
    args = parser.parse_args()

    for p in [args.model, args.encoder]:
        if not Path(p).exists():
            sys.exit(f"[ERROR] File not found: {p}\n"
                     "        Train the model first by running yoga-cnn.ipynb.")

    print("Loading model …")
    model = tf.keras.models.load_model(args.model)
    le    = joblib.load(args.encoder)
    print(f"Loaded — {len(le.classes_)} classes\n")

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"[ERROR] Input not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        label, conf, frames = predict_image_file(input_path, model, le)
        src_type = "image"
    elif suffix in {".mp4", ".avi", ".mov", ".mkv"}:
        label, conf, frames = predict_video_file(input_path, model, le)
        src_type = "video"
    else:
        sys.exit(f"[ERROR] Unsupported file type: {suffix}")

    if label is None:
        print("[WARN] No predictions could be made — check the input file.")
        return

    format_result(label, conf, frames, src_type)


if __name__ == "__main__":
    main()
