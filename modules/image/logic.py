import numpy as np
import json
import os
import uuid
from PIL import Image
import io
import base64
from flask import jsonify
import tensorflow as tf

# ── Paths ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

# global static folder (important fix)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────
model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, 'vegetable_classifier.keras')
)

with open(os.path.join(MODEL_DIR, 'class_indices.json')) as f:
    CLASS_INDICES = json.load(f)

IDX_TO_CLASS = {v: k for k, v in CLASS_INDICES.items()}
IMG_SIZE = (224, 224)

# ── Metadata ──────────────────────────────────────────
VEGGIE_META = {
    'Bean': {'emoji': '🫘', 'color': '#7cb342', 'calories': '31 kcal/100g',
             'benefits': 'High in protein and fibre.', 'uses': 'Curries, salads'},
    # keep rest same...
}

DEFAULT_META = {
    'emoji': '🥬',
    'color': '#4caf50',
    'calories': 'N/A',
    'benefits': 'Healthy vegetable',
    'uses': 'Various dishes'
}

# ── Preprocess ────────────────────────────────────────
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ── MAIN FUNCTION ─────────────────────────────────────
def predict_image(request):
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400

        file = request.files['image']
        img_bytes = file.read()

        # Save image
        fname = f"{uuid.uuid4().hex[:8]}.jpg"
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        save_path = os.path.join(UPLOAD_DIR, fname)
        img_pil.save(save_path)

        # Predict
        arr = preprocess_image(img_bytes)
        preds = model.predict(arr, verbose=0)[0]

        top3_idx = np.argsort(preds)[::-1][:3]
        top3 = [(IDX_TO_CLASS[i], float(preds[i]) * 100) for i in top3_idx]

        pred_class = top3[0][0]
        confidence = top3[0][1]
        meta = VEGGIE_META.get(pred_class, DEFAULT_META)

        # Base64 preview
        buffered = io.BytesIO()
        img_pil.thumbnail((400, 400))
        img_pil.save(buffered, format='JPEG', quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'success': True,
            'prediction': pred_class,
            'confidence': round(confidence, 1),
            'top3': top3,
            'emoji': meta['emoji'],
            'color': meta['color'],
            'calories': meta['calories'],
            'benefits': meta['benefits'],
            'uses': meta['uses'],
            'image_b64': img_b64,
            'image_path': f"/static/uploads/{fname}"  # optional
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500