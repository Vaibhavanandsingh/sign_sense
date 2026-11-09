from pydantic import BaseModel


class TopKItem(BaseModel):
    class_id: int
    label: str
    confidence: float


class Prediction(BaseModel):
    filename: str | None
    class_id: int
    label: str
    confidence: float
    is_confident: bool
    category: str
    action: str
    speed_limit_kmh: int | None
    top_k: list[TopKItem]
    inference_ms: float


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
    total_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class ClassInfo(BaseModel):
    class_id: int
    label: str
    category: str


class ModelInfo(BaseModel):
    architecture: str
    num_classes: int
    num_parameters: int
    img_size: int
    device: str
    val_accuracy: float | None
    test_metrics: dict | None
