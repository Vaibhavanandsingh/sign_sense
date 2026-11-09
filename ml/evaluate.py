import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from ml.config import CHECKPOINT_PATH, CLASS_NAMES, NUM_CLASSES, REPORTS_DIR
from ml.data import get_test_loader
from ml.model import SignNet
from ml.utils import get_device


@torch.inference_mode()
def collect_predictions(model, loader, device):
    preds, targets = [], []
    for images, labels in tqdm(loader, desc="test"):
        preds.append(model(images.to(device)).argmax(dim=1).cpu())
        targets.append(labels)
    return torch.cat(targets).numpy(), torch.cat(preds).numpy()


def plot_confusion(y_true, y_pred, path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)), normalize="true")
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Predicted class id")
    ax.set_ylabel("True class id")
    ax.set_title("Normalized confusion matrix (GTSRB test set)")
    fig.colorbar(image)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model = SignNet(NUM_CLASSES, checkpoint["img_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    y_true, y_pred = collect_predictions(model, get_test_loader(), device)
    report = classification_report(
        y_true, y_pred, labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    per_class_f1 = {name: float(report[name]["f1-score"]) for name in CLASS_NAMES}
    worst = sorted(per_class_f1.items(), key=lambda item: item[1])[:5]

    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "num_test_images": int(len(y_true)),
        "worst_classes": [{"class": name, "f1": f1} for name, f1 in worst],
        "per_class_f1": per_class_f1,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(y_true, y_pred, REPORTS_DIR / "confusion_matrix.png")

    print(f"Test accuracy: {metrics['test_accuracy']:.4f}  Macro F1: {metrics['macro_f1']:.4f}")
    for row in metrics["worst_classes"]:
        print(f"  weakest: {row['class']:<45} F1 {row['f1']:.3f}")


if __name__ == "__main__":
    main()
