import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app, model  # noqa: E402
from preprocess import clean_lyrics  # noqa: E402

pytestmark = pytest.mark.skipif(model is None, reason="run python train.py first")
client = app.test_client()
SAD = "I sit alone in the rain tonight\nYou left and the house is empty and cold\nTears keep falling and I miss you so"


def test_home_page():
    assert client.get("/").status_code == 200


def test_predict_ok():
    r = client.post("/predict", json={"lyrics": SAD})
    data = r.get_json()
    assert r.status_code == 200
    assert data["emotion"] in {s["emotion"] for s in data["scores"]}
    assert abs(sum(s["probability"] for s in data["scores"]) - 1) < 0.01


def test_predict_too_short():
    assert client.post("/predict", json={"lyrics": "hi"}).status_code == 400


def test_predict_missing_body():
    assert client.post("/predict").status_code == 400


def test_metrics_endpoint():
    assert "cv_results" in client.get("/metrics").get_json()


def test_negation_kept_and_labels_stripped():
    assert "not" in clean_lyrics("I am not happy")
    assert "chorus" not in clean_lyrics("[Chorus] love love")
