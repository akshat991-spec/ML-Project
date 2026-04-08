import os
import warnings
import numpy as np
import joblib
import librosa
from flask import jsonify
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load models ───────────────────────────────────────
scaler    = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_rbf.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))

# ── Allowed formats ───────────────────────────────────
ALLOWED_EXT = {"wav", "mp3", "ogg", "flac", "m4a"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ── Feature extraction ────────────────────────────────
def extract_features(file_path, duration=3):
    audio, sr = librosa.load(file_path, duration=duration, res_type="kaiser_fast")

    mfcc     = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma   = librosa.feature.chroma_stft(y=audio, sr=sr)
    zcr      = np.mean(librosa.feature.zero_crossing_rate(audio))
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    rolloff  = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    rms      = np.mean(librosa.feature.rms(y=audio))

    return np.hstack([
        np.mean(mfcc.T, axis=0),
        np.mean(chroma.T, axis=0),
        [zcr, centroid, rolloff, rms]
    ])

# ── MAIN FUNCTION ─────────────────────────────────────
def predict_audio(request):
    filepath = None
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        file = request.files["audio"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": f"Unsupported format. Use: {', '.join(ALLOWED_EXT)}"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        features = extract_features(filepath)
        features_sc = scaler.transform([features])

        # ── SVM ──
        svm_prob = svm_model.predict_proba(features_sc)[0]
        svm_conf = float(round(max(svm_prob) * 100, 2))

        svm_result = {
            "model": "SVM (RBF)",
            "prediction": "Female" if np.argmax(svm_prob) == 1 else "Male",
            "confidence": svm_conf,
            "male_prob": float(round(svm_prob[0] * 100, 2)),
            "female_prob": float(round(svm_prob[1] * 100, 2)),
        }

        # ── XGBoost ──
        xgb_prob = xgb_model.predict_proba(features_sc)[0]
        xgb_conf = float(round(max(xgb_prob) * 100, 2))

        xgb_result = {
            "model": "XGBoost",
            "prediction": "Female" if np.argmax(xgb_prob) == 1 else "Male",
            "confidence": xgb_conf,
            "male_prob": float(round(xgb_prob[0] * 100, 2)),
            "female_prob": float(round(xgb_prob[1] * 100, 2)),
        }

        # ── BEST MODEL ──
        best_result = svm_result if svm_conf >= xgb_conf else xgb_result

        # ── Feature Info ──
        feature_info = {
            "zcr": round(float(features[-4]), 4),
            "centroid": round(float(features[-3]), 2),
            "rolloff": round(float(features[-2]), 2),
            "rms": round(float(features[-1]), 4),
            "mfcc_mean": round(float(np.mean(features[:40])), 4),
        }

        return jsonify({
            "success": True,
            "filename": filename,
            "result": best_result,
            "features": feature_info
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)