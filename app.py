from flask import Flask, render_template, request

# ── Import all modules ─────────────────────────────
from modules.text.logic import predict as text_predict
from modules.audio.logic import predict_audio as audio_predict
from modules.image.logic import predict_image as image_predict
from modules.numeric.logic import predict_aqi as numeric_predict
from modules.video.logic import predict_shoplifting as video_predict

app = Flask(__name__)

# ── ROUTES ─────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text')
def text_page():
    return render_template('text.html')

@app.route('/audio')
def audio_page():
    return render_template('audio.html')

@app.route('/image')
def image_page():
    return render_template('image.html')

@app.route('/numeric')
def numeric_page():
    return render_template('numeric.html')

@app.route('/video')
def video_page():
    return render_template('video.html')

# ── API ROUTES ─────────────────────────────────────

@app.route('/predict/text', methods=['POST'])
def predict_toxic():
    return text_predict(request)

@app.route('/predict/audio', methods=['POST'])
def predict_audio():
    return audio_predict(request)

@app.route('/predict/image', methods=['POST'])
def predict_image():
    return image_predict(request)

@app.route('/predict/numeric', methods=['POST'])
def predict_numeric():
    return numeric_predict(request)

@app.route('/predict/video', methods=['POST'])
def predict_video():
    return video_predict(request)

# ── RUN ────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)