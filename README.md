# Machine Learning Project

A multi-modal AI system built with Flask that provides predictions across five different data types — numeric, text, audio, image, and video — through a unified web dashboard.

---

## Modules

| Module | Task | Model Type |
|--------|------|------------|
| Air Quality (AQI) | Predict air quality index from sensor data | Regression |
| Text Analysis | Toxicity detection in natural language | NLP / Transformer |
| Audio Recognition | Speaker gender classification from audio | Deep Learning |
| Image Classification | Vegetable identification from images | CNN |
| Video Detection | Shoplifting detection in CCTV footage | Ensemble (CNN + LSTM + EfficientNet) |

---

## Project Structure

```
Machine-Learning-Project/
│
├── app.py                        # Main Flask application entry point
├── requirements.txt              # Python dependencies
├── .gitignore
├── README.md
│
├── modules/
│   ├── numeric/                  # AQI prediction module
│   ├── text/                     # Toxicity detection module
│   │   └── models/
│   │       └── bert_model/       # BERT model (not included, see below)
│   ├── audio/                    # Gender recognition module
│   ├── image/                    # Vegetable classification module
│   └── video/                    # Shoplifting detection module
│       ├── logic.py
│       └── models/
│           ├── shoplifting_cnn.h5
│           ├── shoplifting_lstm.h5
│           ├── eff_lstm_best.h5
│           └── feature_extractor.h5
│
├── static/
│   └── uploads/                  # Temporary file storage (auto-created)
│
└── templates/
    ├── dashboard.html
    ├── numeric.html
    ├── text.html
    ├── audio.html
    ├── image.html
    └── video.html
```

---

## Requirements

- Python 3.8 or higher
- pip

Install all dependencies:

```bash
pip install -r requirements.txt
```

Key packages used:

- Flask
- TensorFlow / Keras
- OpenCV (`cv2`)
- NumPy, Pandas
- Scikit-learn
- Pillow
- Librosa (audio processing)
- Transformers (HuggingFace — for BERT)

---

## Large Model Files

Certain model files exceed GitHub's 100MB file size limit and are not included in this repository. You must download them manually and place them in the correct directories before running the application.

### BERT Model (Text Module)

The file `model.safetensors` (~418 MB) is required for the text toxicity module.

Download link: _[Add your Google Drive / HuggingFace link here]_

Place the downloaded file at:

```
modules/text/models/bert_model/model.safetensors
```

### Video Module Models

The `.h5` model files for the video shoplifting detection module are also excluded.

Download link: _[Add your Google Drive link here]_

Place them at:

```
modules/video/models/shoplifting_cnn.h5
modules/video/models/shoplifting_lstm.h5
modules/video/models/eff_lstm_best.h5
modules/video/models/feature_extractor.h5
modules/video/models/feat_mean.npy
modules/video/models/feat_std.npy
modules/video/models/eff_mean.npy
modules/video/models/eff_std.npy
modules/video/models/model_config.json
```

---

## Running the Application

```bash
python app.py
```

The application will start on `http://localhost:5000` by default.

Open your browser and navigate to `http://localhost:5000` to access the dashboard.

---

## API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Main dashboard |
| `/numeric` | GET | AQI prediction page |
| `/text` | GET | Text analysis page |
| `/audio` | GET | Audio recognition page |
| `/image` | GET | Image classification page |
| `/video` | GET | Video detection page |
| `/predict/video` | POST | Video inference endpoint |

---

## Video Detection — How It Works

The video module uses an ensemble of three models:

1. **CNN (MobileNetV2-based)** — frame-level classification
2. **MobileNet + LSTM** — temporal sequence modeling
3. **EfficientNet + LSTM** — temporal sequence modeling with EfficientNet features

Each model produces an independent prediction. A **majority voting** strategy is applied — if 2 or more models flag a video as a threat, the final result is classified as shoplifting detected.

16 frames are uniformly sampled from each uploaded video for inference.

---

## Notes

- Uploaded files are temporarily stored in `static/uploads/` and deleted automatically after inference.
- The application is intended for local/demo use. For production deployment, configure a proper WSGI server (e.g. Gunicorn) and disable Flask debug mode.
- Model accuracy figures are stored in `model_config.json` per module.

---

## Author

Aastha  [github.com/Aastha0625](https://github.com/Aastha0625)
Aanchal [github.com/Aanchal86](https://github.com/Aanchal86)
Akshat  [github.com/akshat991-spec](https://github.com/akshat991-spec)