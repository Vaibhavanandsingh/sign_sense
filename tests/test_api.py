import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from ml.config import CLASS_NAMES, IMG_SIZE, MEAN, STD
from ml.model import SignNet


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    checkpoint = tmp_path_factory.mktemp("model") / "dummy.pt"
    torch.save(
        {
            "state_dict": SignNet(len(CLASS_NAMES), IMG_SIZE).state_dict(),
            "class_names": CLASS_NAMES,
            "img_size": IMG_SIZE,
            "mean": MEAN,
            "std": STD,
            "val_acc": None,
        },
        checkpoint,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODEL_PATH", str(checkpoint))
        mp.setenv("DEVICE", "cpu")
        from backend.main import app

        with TestClient(app) as test_client:
            yield test_client


def png_bytes(color=(200, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["device"] == "cpu"


def test_classes(client):
    classes = client.get("/classes").json()
    assert len(classes) == 43
    assert classes[14]["label"] == "Stop"
    assert classes[14]["category"] == "stop"


def test_model_info(client):
    info = client.get("/model/info").json()
    assert info["num_classes"] == 43
    assert info["img_size"] == IMG_SIZE
    assert info["num_parameters"] > 0


def test_predict_returns_ranked_top_k(client):
    response = client.post("/predict", files={"file": ("red.png", png_bytes(), "image/png")},
                           params={"top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "red.png"
    assert len(body["top_k"]) == 3
    confidences = [item["confidence"] for item in body["top_k"]]
    assert confidences == sorted(confidences, reverse=True)
    assert body["class_id"] == body["top_k"][0]["class_id"]
    assert 0.0 <= body["confidence"] <= 1.0


def test_untrained_model_is_not_confident(client):
    body = client.post("/predict", files={"file": ("red.png", png_bytes(), "image/png")}).json()
    assert body["is_confident"] is False


def test_predict_rejects_non_image(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "not a valid image" in response.json()["detail"]


def test_predict_rejects_empty_file(client):
    response = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400


@pytest.mark.parametrize("top_k", [0, 11])
def test_predict_rejects_out_of_range_top_k(client, top_k):
    response = client.post("/predict", files={"file": ("red.png", png_bytes(), "image/png")},
                           params={"top_k": top_k})
    assert response.status_code == 422


def test_batch(client):
    files = [("files", (f"{i}.png", png_bytes((i * 40, 80, 120)), "image/png")) for i in range(4)]
    body = client.post("/predict/batch", files=files).json()
    assert [p["filename"] for p in body["predictions"]] == ["0.png", "1.png", "2.png", "3.png"]


def test_batch_rejects_more_than_32_images(client):
    files = [("files", (f"{i}.png", png_bytes(), "image/png")) for i in range(33)]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 400
