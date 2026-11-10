import io
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"
REPORTS_DIR = ROOT / "reports"
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
IMAGE_TYPES = ["png", "jpg", "jpeg", "ppm", "bmp", "webp"]

START_LIMIT = 50
OPEN_ROAD_LIMIT = 100
SPEED_FACTOR = {"stop": 0.0, "no_entry": 0.0, "yield": 0.5, "warning": 0.7}

st.set_page_config(page_title="SignSense", layout="wide")


class ApiError(Exception):
    pass


def call_api(method: str, path: str, **kwargs) -> dict | list:
    try:
        response = requests.request(method, f"{api_url}{path}", timeout=30, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the API at {api_url}") from exc
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(f"API error {response.status_code}: {detail}")
    return response.json()


def predict_image(name: str, data: bytes, top_k: int) -> dict:
    return call_api("POST", "/predict", files={"file": (name, data)}, params={"top_k": top_k})


def to_display(data: bytes, size: int) -> Image.Image:
    return ImageOps.contain(Image.open(io.BytesIO(data)).convert("RGB"), (size, size))


def ground_truth(filename: str) -> int | None:
    match = re.search(r"class(\d+)", filename)
    return int(match.group(1)) if match else None


def plan(limit: int, result: dict) -> tuple[int, int, str]:
    """Rule-based planner: returns (new speed limit, commanded speed, decision text)."""
    if not result["is_confident"]:
        return limit, limit, f"Uncertain sign ({result['confidence']:.0%}), keeping current limit"
    if result["speed_limit_kmh"]:
        limit = result["speed_limit_kmh"]
    elif result["class_id"] in (6, 32):
        limit = OPEN_ROAD_LIMIT
    speed = round(limit * SPEED_FACTOR.get(result["category"], 1.0))
    return limit, speed, result["action"]


def render_prediction(data: bytes, result: dict) -> None:
    left, right = st.columns([1, 2])
    left.image(to_display(data, 220), caption=result["filename"])
    with right:
        headline = f"**{result['label']}** ({result['confidence']:.1%})"
        if result["is_confident"]:
            st.success(headline)
            st.info(f"Vehicle action: {result['action']}")
        else:
            st.warning(f"Low confidence: {headline}. The vehicle should treat this as an unknown sign.")
            st.info("Vehicle action: keep current behaviour until the sign is confirmed")
        st.caption(f"Category: {result['category']} | Inference: {result['inference_ms']:.1f} ms")
        st.dataframe(
            pd.DataFrame(result["top_k"]),
            hide_index=True,
            column_config={
                "confidence": st.column_config.ProgressColumn(
                    "confidence", min_value=0.0, max_value=1.0, format="%.3f"
                )
            },
        )


with st.sidebar:
    st.title("SignSense")
    st.caption("Traffic sign recognition for autonomous vehicles")
    api_url = st.text_input("API URL", DEFAULT_API_URL).rstrip("/")
    top_k = st.slider("Top-k predictions", 1, 10, 5)
    try:
        health = call_api("GET", "/health")
        class_names = {c["class_id"]: c["label"] for c in call_api("GET", "/classes")}
        st.success(f"API online, model on {health['device']}")
    except ApiError as exc:
        st.error(f"{exc}. Start it with `uvicorn backend.main:app --reload`.")
        st.stop()

samples = sorted(SAMPLES_DIR.glob("*.png"))
classify_tab, batch_tab, drive_tab, model_tab = st.tabs(
    ["Classify", "Batch", "Drive simulation", "Model report"]
)

with classify_tab:
    source = st.radio("Image source", ["Upload", "Test-set sample", "Camera"], horizontal=True)
    name, data = None, None
    if source == "Upload":
        upload = st.file_uploader("Traffic sign image", type=IMAGE_TYPES)
        if upload:
            name, data = upload.name, upload.getvalue()
    elif source == "Test-set sample":
        if samples:
            path = st.selectbox("Sample", samples, format_func=lambda p: p.name)
            name, data = path.name, path.read_bytes()
        else:
            st.info("No samples yet. Run `python -m ml.export_samples`.")
    else:
        shot = st.camera_input("Hold a sign (or a picture of one) up to the camera")
        if shot:
            name, data = "camera.jpg", shot.getvalue()

    if data:
        try:
            result = predict_image(name, data, top_k)
        except ApiError as exc:
            st.error(str(exc))
        else:
            render_prediction(data, result)
            truth = ground_truth(name)
            if truth is not None:
                verdict = "correct" if truth == result["class_id"] else "wrong"
                st.caption(f"Ground truth: {class_names[truth]} ({verdict})")

with batch_tab:
    uploads = st.file_uploader("Upload up to 32 images", type=IMAGE_TYPES,
                               accept_multiple_files=True, key="batch_upload")
    use_samples = st.button("Or classify 12 random test samples", disabled=not samples)

    items = [(u.name, u.getvalue()) for u in uploads] if uploads else []
    if use_samples:
        items = [(p.name, p.read_bytes()) for p in random.sample(samples, min(12, len(samples)))]
    items = items[:32]

    if items:
        try:
            body = call_api("POST", "/predict/batch",
                            files=[("files", item) for item in items], params={"top_k": 3})
        except ApiError as exc:
            st.error(str(exc))
        else:
            predictions = body["predictions"]
            st.caption(f"{len(predictions)} images classified in {body['total_ms']:.1f} ms")
            columns = st.columns(6)
            for i, ((_, img), pred) in enumerate(zip(items, predictions)):
                columns[i % 6].image(to_display(img, 120),
                                     caption=f"{pred['label']} ({pred['confidence']:.0%})")
            rows = []
            for pred in predictions:
                truth = ground_truth(pred["filename"] or "")
                rows.append({
                    "file": pred["filename"],
                    "prediction": pred["label"],
                    "confidence": round(pred["confidence"], 3),
                    "action": pred["action"],
                    "correct": None if truth is None else truth == pred["class_id"],
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True)

with drive_tab:
    st.write(
        "A simulated car drives past a random sequence of signs from the test set. "
        "Each frame is sent to the API, and a rule-based planner turns the prediction "
        "into a speed command."
    )
    if len(samples) < 5:
        st.info("Export at least 5 samples with `python -m ml.export_samples`.")
    else:
        col_a, col_b = st.columns(2)
        n_signs = col_a.slider("Signs on the route", 5, min(40, len(samples)), min(12, len(samples)))
        delay = col_b.slider("Seconds per sign", 0.2, 2.0, 0.8, step=0.1)

        if st.button("Start drive", type="primary"):
            route = random.sample(samples, n_signs)
            limit, log = START_LIMIT, []
            frame = st.empty()
            progress = st.progress(0.0)
            for step, path in enumerate(route, start=1):
                data = path.read_bytes()
                try:
                    result = predict_image(path.name, data, 3)
                except ApiError as exc:
                    st.error(str(exc))
                    break
                limit, speed, decision = plan(limit, result)

                with frame.container():
                    img_col, sign_col, speed_col, limit_col = st.columns([1, 2, 1, 1])
                    img_col.image(to_display(data, 160))
                    sign_col.metric("Detected sign", result["label"])
                    sign_col.caption(f"{result['confidence']:.0%} confidence")
                    speed_col.metric("Speed", f"{speed} km/h")
                    limit_col.metric("Limit", f"{limit} km/h")
                    st.info(f"Decision: {decision}")

                truth = ground_truth(path.name)
                log.append({
                    "step": step,
                    "prediction": result["label"],
                    "confidence": round(result["confidence"], 3),
                    "truth": class_names.get(truth, "?"),
                    "correct": truth == result["class_id"],
                    "speed_kmh": speed,
                    "limit_kmh": limit,
                    "decision": decision,
                })
                progress.progress(step / len(route), text=f"Sign {step}/{len(route)}")
                time.sleep(delay)

            if log:
                df = pd.DataFrame(log)
                st.metric("Route accuracy", f"{df['correct'].mean():.0%}")
                st.line_chart(df.set_index("step")[["speed_kmh", "limit_kmh"]])
                st.dataframe(df, hide_index=True)

with model_tab:
    try:
        info = call_api("GET", "/model/info")
    except ApiError as exc:
        st.error(str(exc))
    else:
        metrics = info.get("test_metrics") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Architecture", info["architecture"].split(" (")[0], help=info["architecture"])
        c2.metric("Parameters", f"{info['num_parameters']:,}")
        c3.metric("Val accuracy", f"{info['val_accuracy']:.2%}" if info["val_accuracy"] else "n/a")
        c4.metric("Test accuracy", f"{metrics['test_accuracy']:.2%}" if metrics else "run ml.evaluate")

        for filename, caption in [("training_curves.png", "Training curves"),
                                  ("confusion_matrix.png", "Confusion matrix (test set)")]:
            if (REPORTS_DIR / filename).exists():
                st.image(str(REPORTS_DIR / filename), caption=caption)

        if metrics.get("worst_classes"):
            st.subheader("Hardest classes")
            st.dataframe(pd.DataFrame(metrics["worst_classes"]), hide_index=True)

        st.subheader("All classes")
        st.dataframe(pd.DataFrame(call_api("GET", "/classes")), hide_index=True)
