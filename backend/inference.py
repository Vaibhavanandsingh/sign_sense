import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from ml.model import SignNet
from ml.utils import get_device


class SignClassifier:
    def __init__(self, checkpoint_path: Path):
        self.device = get_device()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.class_names: list[str] = list(checkpoint["class_names"])
        self.img_size: int = checkpoint["img_size"]
        self.val_acc: float | None = checkpoint.get("val_acc")

        self.model = SignNet(len(self.class_names), self.img_size)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.num_parameters = sum(p.numel() for p in self.model.parameters())

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
        ])
        self.predict([Image.new("RGB", (self.img_size, self.img_size))])  # warm-up

    @staticmethod
    def load_image(data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(data)).convert("RGB")

    @torch.inference_mode()
    def predict(self, images: list[Image.Image], top_k: int = 5) -> list[list[tuple[int, float]]]:
        batch = torch.stack([self.transform(image) for image in images]).to(self.device)
        probabilities = torch.softmax(self.model(batch), dim=1)
        confidences, indices = probabilities.topk(min(top_k, len(self.class_names)), dim=1)
        return [
            list(zip(idx.tolist(), conf.tolist()))
            for idx, conf in zip(indices.cpu(), confidences.cpu())
        ]
