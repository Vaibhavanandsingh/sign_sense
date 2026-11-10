import pytest
import torch
from PIL import Image

from ml.config import CHECKPOINT_PATH, CLASS_NAMES, IMG_SIZE, NUM_CLASSES, SAMPLES_DIR
from ml.model import SignNet
from ml.signs import describe


def test_model_output_shape():
    logits = SignNet(NUM_CLASSES, IMG_SIZE)(torch.randn(2, 3, IMG_SIZE, IMG_SIZE))
    assert logits.shape == (2, NUM_CLASSES)


def test_every_class_has_a_known_category():
    assert len(CLASS_NAMES) == NUM_CLASSES
    for class_id in range(NUM_CLASSES):
        assert describe(class_id)["category"] != "other", CLASS_NAMES[class_id]


@pytest.mark.parametrize("class_id, limit", [(0, 20), (1, 30), (2, 50), (7, 100), (8, 120)])
def test_speed_limits(class_id, limit):
    info = describe(class_id)
    assert info["category"] == "speed_limit"
    assert info["speed_limit_kmh"] == limit


def test_only_speed_limit_signs_set_a_speed():
    for class_id in range(NUM_CLASSES):
        info = describe(class_id)
        assert (info["speed_limit_kmh"] is not None) == (info["category"] == "speed_limit")


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="trained checkpoint not available")
@pytest.mark.parametrize("sample, expected", [("028_class14.png", "Stop"),
                                              ("004_class02.png", "Speed limit (50km/h)")])
def test_trained_model_on_demo_samples(monkeypatch, sample, expected):
    from backend.inference import SignClassifier

    monkeypatch.setenv("DEVICE", "cpu")
    classifier = SignClassifier(CHECKPOINT_PATH)
    class_id, confidence = classifier.predict([Image.open(SAMPLES_DIR / sample).convert("RGB")], top_k=1)[0][0]
    assert classifier.class_names[class_id] == expected
    assert confidence > 0.9
