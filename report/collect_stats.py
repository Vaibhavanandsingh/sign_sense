"""Collect the numbers used in the project report into report/figures/stats.json."""
import json
import statistics
from collections import Counter
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from torch import nn
from torchvision import datasets

from ml.config import CHECKPOINT_PATH, CLASS_NAMES, DATA_DIR, IMG_SIZE, SAMPLES_DIR
from ml.data import get_test_loader
from ml.model import SignNet
from ml.utils import get_device

OUT = Path(__file__).resolve().parent / "figures" / "stats.json"


def class_counts() -> dict:
    train = Counter(label for _, label in datasets.GTSRB(DATA_DIR, split="train")._samples)
    test = Counter(label for _, label in datasets.GTSRB(DATA_DIR, split="test")._samples)
    return {"train": [train[i] for i in range(43)], "test": [test[i] for i in range(43)]}


def layer_table() -> list[dict]:
    model = SignNet().eval()
    rows, x = [], torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

    def walk(module, x, prefix):
        for name, child in module.named_children():
            if isinstance(child, nn.Sequential):
                x = walk(child, x, f"{prefix}{name}.")
            else:
                x = child(x)
                rows.append({"name": f"{prefix}{name}", "type": type(child).__name__,
                             "shape": list(x.shape[1:]),
                             "params": sum(p.numel() for p in child.parameters())})
        return x

    with torch.no_grad():
        walk(model, x, "")
    return rows


def top_confusions(n: int = 8) -> list[dict]:
    device = get_device()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model = SignNet().to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    pairs = Counter()
    with torch.inference_mode():
        for images, labels in get_test_loader():
            preds = model(images.to(device)).argmax(dim=1).cpu()
            for t, p in zip(labels.tolist(), preds.tolist()):
                if t != p:
                    pairs[(t, p)] += 1
    return [{"true": CLASS_NAMES[t], "true_id": t, "pred": CLASS_NAMES[p], "pred_id": p, "count": c}
            for (t, p), c in pairs.most_common(n)]


def api_benchmark() -> dict:
    from backend.main import app

    samples = sorted(SAMPLES_DIR.glob("*.png"))
    latencies, misses, correct = [], [], 0
    with TestClient(app) as client:
        for path in samples:
            body = client.post("/predict", files={"file": (path.name, path.read_bytes())}).json()
            truth = int(path.stem.split("class")[1])
            latencies.append(body["inference_ms"])
            if body["class_id"] == truth:
                correct += 1
            else:
                misses.append({"file": path.name, "truth": CLASS_NAMES[truth], "pred": body["label"],
                               "confidence": body["confidence"], "is_confident": body["is_confident"]})
        files = [("files", (p.name, p.read_bytes())) for p in samples[:12]]
        batch = client.post("/predict/batch", files=files).json()
    latencies.sort()
    return {"num_samples": len(samples), "correct": correct, "misses": misses,
            "median_ms": statistics.median(latencies),
            "p95_ms": latencies[int(0.95 * (len(latencies) - 1))],
            "max_ms": latencies[-1], "batch12_total_ms": batch["total_ms"]}


if __name__ == "__main__":
    stats = {"class_counts": class_counts(), "layers": layer_table(),
             "top_confusions": top_confusions(), "api": api_benchmark()}
    OUT.write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k in ("top_confusions", "api")}, indent=2))
