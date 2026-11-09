import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from backend.inference import SignClassifier
from backend.schemas import (BatchPrediction, ClassInfo, HealthResponse, ModelInfo,
                             Prediction, TopKItem)
from ml.config import CHECKPOINT_PATH, REPORTS_DIR
from ml.signs import describe

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_BATCH = 32

state: dict[str, SignClassifier] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = Path(os.getenv("MODEL_PATH", CHECKPOINT_PATH))
    if not checkpoint.exists():
        raise RuntimeError(f"No model at {checkpoint}. Run `python -m ml.train` first.")
    state["classifier"] = SignClassifier(checkpoint)
    yield
    state.clear()


app = FastAPI(
    title="SignSense API",
    description="Traffic sign recognition for autonomous vehicle simulations",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def classifier() -> SignClassifier:
    return state["classifier"]


async def read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if not data:
        raise HTTPException(400, f"{file.filename} is empty")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"{file.filename} is larger than 5 MB")
    try:
        return SignClassifier.load_image(data)
    except OSError:
        raise HTTPException(400, f"{file.filename} is not a valid image")


def to_prediction(filename: str | None, ranked: list[tuple[int, float]], elapsed_ms: float) -> Prediction:
    names = classifier().class_names
    class_id, confidence = ranked[0]
    return Prediction(
        filename=filename,
        class_id=class_id,
        label=names[class_id],
        confidence=round(confidence, 4),
        is_confident=confidence >= CONFIDENCE_THRESHOLD,
        top_k=[TopKItem(class_id=c, label=names[c], confidence=round(p, 4)) for c, p in ranked],
        inference_ms=round(elapsed_ms, 2),
        **describe(class_id),
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded="classifier" in state, device=str(classifier().device))


@app.get("/classes", response_model=list[ClassInfo])
def classes():
    return [
        ClassInfo(class_id=i, label=name, category=describe(i)["category"])
        for i, name in enumerate(classifier().class_names)
    ]


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    clf = classifier()
    metrics_path = REPORTS_DIR / "metrics.json"
    return ModelInfo(
        architecture="SignNet (3-block CNN)",
        num_classes=len(clf.class_names),
        num_parameters=clf.num_parameters,
        img_size=clf.img_size,
        device=str(clf.device),
        val_accuracy=clf.val_acc,
        test_metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else None,
    )


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=10)):
    image = await read_image(file)
    start = time.perf_counter()
    ranked = (await run_in_threadpool(classifier().predict, [image], top_k))[0]
    return to_prediction(file.filename, ranked, (time.perf_counter() - start) * 1000)


@app.post("/predict/batch", response_model=BatchPrediction)
async def predict_batch(files: list[UploadFile] = File(...), top_k: int = Query(3, ge=1, le=10)):
    if len(files) > MAX_BATCH:
        raise HTTPException(400, f"Send at most {MAX_BATCH} images per batch")
    images = [await read_image(f) for f in files]
    start = time.perf_counter()
    ranked = await run_in_threadpool(classifier().predict, images, top_k)
    total_ms = (time.perf_counter() - start) * 1000
    per_image_ms = total_ms / len(files)
    return BatchPrediction(
        predictions=[to_prediction(f.filename, r, per_image_ms) for f, r in zip(files, ranked)],
        total_ms=round(total_ms, 2),
    )
