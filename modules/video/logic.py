import os
import json
import time
import uuid
import base64
import io
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from flask import jsonify

from tensorflow.keras import layers, Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# ── Paths ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("🎥 Loading video models...")

# ── Load models ───────────────────────────────────────
cnn_model         = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'shoplifting_cnn.h5'))
lstm_model        = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'shoplifting_lstm.h5'))
eff_model         = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'eff_lstm_best.h5'))
feature_extractor = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'feature_extractor.h5'))

# ── Load config ───────────────────────────────────────
with open(os.path.join(MODEL_DIR, 'model_config.json')) as f:
    config = json.load(f)

# ── Load normalization ────────────────────────────────
FEAT_MEAN = np.load(os.path.join(MODEL_DIR, 'feat_mean.npy'))[0]
FEAT_STD  = np.load(os.path.join(MODEL_DIR, 'feat_std.npy'))[0]
EFF_MEAN  = np.load(os.path.join(MODEL_DIR, 'eff_mean.npy'))[0]
EFF_STD   = np.load(os.path.join(MODEL_DIR, 'eff_std.npy'))[0]

# ── Config ────────────────────────────────────────────
FRAMES_PER_VID = config.get('frames_per_vid', 16)
IMG_SIZE       = tuple(config.get('img_size', [224, 224]))
CLASS_NAMES    = config.get('class_names', ['Normal', 'Shoplifting'])

# ── EfficientNet feature extractor ────────────────────
base_eff = tf.keras.applications.EfficientNetB0(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_eff.trainable = False

eff_extractor = Model(
    inputs=base_eff.input,
    outputs=layers.GlobalAveragePooling2D()(base_eff.output)
)

print("✅ Video models loaded!")


# ─────────────────────────────────────────────────────
# FRAME EXTRACTION
# ─────────────────────────────────────────────────────
def extract_frames(video_path, n_frames=16):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    if total <= 0:
        cap.release()
        return [], 0, 0

    indices = set(np.linspace(0, total - 1, n_frames, dtype=int))
    frames, fi = [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in indices:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, IMG_SIZE)
            frames.append(resized)
        fi += 1

    cap.release()

    if len(frames) < n_frames:
        pad     = [np.zeros((*IMG_SIZE, 3), dtype=np.uint8)] * (n_frames - len(frames))
        frames += pad

    return frames[:n_frames], fps, total


def frames_to_b64(frames, size=(160, 90)):
    result = []
    for f in frames:
        img = Image.fromarray(f).resize(size)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=70)
        result.append(base64.b64encode(buf.getvalue()).decode())
    return result


# ─────────────────────────────────────────────────────
# MODEL RUNNERS
# ─────────────────────────────────────────────────────
def run_cnn(frames):
    """Match app.py exactly: use np.mean over frame predictions."""
    arr   = np.array(frames, dtype=np.float32)
    pp    = mobilenet_preprocess(arr.copy())
    preds = cnn_model.predict(pp, verbose=0).flatten()
    return float(np.mean(preds)), preds.tolist()   # ← FIXED: was np.max


def run_mobilenet_lstm(frames):
    arr      = np.array(frames, dtype=np.float32)
    pp       = mobilenet_preprocess(arr.copy())
    features = feature_extractor.predict(pp, verbose=0)
    features = (features - FEAT_MEAN) / FEAT_STD
    preds    = lstm_model.predict(features[np.newaxis, ...], verbose=0)[0]
    return float(preds[1]), preds.tolist()


def run_efficientnet_lstm(frames):
    arr      = np.array(frames, dtype=np.float32)
    pp       = efficientnet_preprocess(arr.copy())
    features = eff_extractor.predict(pp, verbose=0)
    features = (features - EFF_MEAN) / EFF_STD
    preds    = eff_model.predict(features[np.newaxis, ...], verbose=0)[0]
    return float(preds[1]), preds.tolist()


def build_result(prob, frame_probs=None):
    pred = int(prob > 0.5)
    conf = prob * 100 if pred == 1 else (1 - prob) * 100

    if frame_probs:
        scores = [round(p * 100, 1) for p in frame_probs]
    else:
        scores = [round(prob * 100, 1)] * FRAMES_PER_VID

    return {
        'pred':             pred,
        'label':            CLASS_NAMES[pred],
        'threat':           pred == 1,
        'conf':             round(conf, 1),
        'shoplifting_prob': round(prob * 100, 1),
        'normal_prob':      round((1 - prob) * 100, 1),
        'scores':           scores,
    }


# ─────────────────────────────────────────────────────
# MAIN PREDICT FUNCTION
# ─────────────────────────────────────────────────────
def predict_shoplifting(request):
    filepath = None
    try:
        if "video" not in request.files:
            return jsonify({'success': False, 'error': 'No video uploaded'}), 400

        file   = request.files["video"]
        suffix = os.path.splitext(file.filename)[1] or ".mp4"

        filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{suffix}")
        file.save(filepath)

        start = time.time()

        frames, fps, total_frames = extract_frames(filepath, FRAMES_PER_VID)

        if not frames:
            return jsonify({'success': False, 'error': 'Could not read video'}), 400

        duration = total_frames / fps if fps > 0 else 0

        # ── Model predictions ──────────────────────────
        cnn_prob,  cnn_frame_probs = run_cnn(frames)
        lstm_prob, _               = run_mobilenet_lstm(frames)
        eff_prob,  _               = run_efficientnet_lstm(frames)

        # ── Debug (server console only) ────────────────
        print(f"CNN prob:  {cnn_prob:.4f}  → {'THREAT' if cnn_prob > 0.5 else 'normal'}")
        print(f"LSTM prob: {lstm_prob:.4f} → {'THREAT' if lstm_prob > 0.5 else 'normal'}")
        print(f"EFF prob:  {eff_prob:.4f}  → {'THREAT' if eff_prob > 0.5 else 'normal'}")

        # ── Ensemble majority voting ───────────────────
        cnn_vote  = int(cnn_prob  > 0.5)
        lstm_vote = int(lstm_prob > 0.5)
        eff_vote  = int(eff_prob  > 0.5)
        majority  = int((cnn_vote + lstm_vote + eff_vote) >= 2)

        print(f"Votes: CNN={cnn_vote} LSTM={lstm_vote} EFF={eff_vote} → majority={majority}")

        # ── Ensemble confidence ────────────────────────
        avg_prob   = (cnn_prob + lstm_prob + eff_prob) / 3
        confidence = avg_prob * 100 if majority else (1 - avg_prob) * 100

        final_result = {
            'label':            CLASS_NAMES[majority],
            'threat':           majority == 1,
            'confidence':       round(confidence, 1),
            'shoplifting_prob': round(avg_prob * 100, 1),
            'normal_prob':      round((1 - avg_prob) * 100, 1),
        }

        return jsonify({
            'success':        True,
            'overall_threat': majority,
            'result':         final_result,
            'cnn':            build_result(cnn_prob, cnn_frame_probs),
            'lstm':           build_result(lstm_prob),
            'eff':            build_result(eff_prob),
            'duration':       round(duration, 1),
            'fps':            round(fps, 1),
            'proc_time':      round(time.time() - start, 2),
            'strip_b64':      frames_to_b64(frames),
        })

    except Exception as e:
        print(f"❌ predict() error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass