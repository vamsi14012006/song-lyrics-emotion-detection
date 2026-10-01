import json
import os
import re

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from preprocess import clean_lyrics

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "emotion_model.joblib")
METRICS_PATH = os.path.join(BASE, "model", "metrics.json")
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
metrics = json.load(open(METRICS_PATH)) if os.path.exists(METRICS_PATH) else {}


def score(text):
    """Return (probabilities, cleaned text) for one piece of text."""
    cleaned = clean_lyrics(text)
    return (model.predict_proba([cleaned])[0] if cleaned else None), cleaned


def explain(cleaned, class_index, n=6):
    """Words in the text that pushed hardest toward the predicted emotion."""
    vec = model["tfidf"].transform([cleaned]).toarray()[0]
    contrib = vec * model["clf"].coef_[class_index]
    vocab = model["tfidf"].get_feature_names_out()
    idx = [i for i in np.argsort(contrib)[::-1][:n] if contrib[i] > 0]
    return [str(vocab[i]) for i in idx]


def sections(text):
    """Split pasted lyrics into verses: blank-line separated, or chunks of 4 lines."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) < 2:
        lines = [l for l in text.splitlines() if l.strip()]
        parts = ["\n".join(lines[i:i + 4]) for i in range(0, len(lines), 4)]
    return [p for p in parts if len(p.split()) >= 8]


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/metrics")
def get_metrics():
    return jsonify(metrics)


@app.post("/predict")
def predict():
    if model is None:
        return jsonify(error="Model not trained yet. Run: python train.py"), 503
    lyrics = (request.get_json(silent=True) or {}).get("lyrics", "").strip()
    if len(lyrics.split()) < 5:
        return jsonify(error="Paste at least a few lines of lyrics."), 400
    probs, cleaned = score(lyrics)
    if probs is None:
        return jsonify(error="No usable words found in that text."), 400

    classes = list(model.classes_)
    top = int(np.argmax(probs))
    ranked = sorted(zip(classes, probs), key=lambda p: -p[1])

    timeline = []
    secs = sections(lyrics)
    if len(secs) >= 3:
        for n, sec in enumerate(secs, 1):
            p, _ = score(sec)
            if p is not None:
                timeline.append({"part": n, "emotion": classes[int(np.argmax(p))], "confidence": round(float(p.max()), 3)})

    return jsonify(
        emotion=classes[top],
        scores=[{"emotion": e, "probability": round(float(p), 4)} for e, p in ranked],
        top_words=explain(cleaned, top),
        timeline=timeline,
    )


if __name__ == "__main__":
    app.run(debug=True)
