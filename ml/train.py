import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from ml.config import (CHECKPOINT_PATH, CLASS_NAMES, IMG_SIZE, MEAN, MODELS_DIR,
                       NUM_CLASSES, REPORTS_DIR, STD)
from ml.data import get_dataloaders
from ml.model import SignNet
from ml.utils import get_device


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, leave=False, desc="train" if training else "val"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)
    return total_loss / seen, correct / seen


def plot_history(history: dict, path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()
    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SignNet on GTSRB")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(42)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Training on {device}")
    train_loader, val_loader = get_dataloaders(args.batch_size, num_workers=args.num_workers)

    model = SignNet(NUM_CLASSES, IMG_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "class_names": CLASS_NAMES,
                    "img_size": IMG_SIZE,
                    "mean": MEAN,
                    "std": STD,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                CHECKPOINT_PATH,
            )
            print(f"  saved new best model (val acc {val_acc:.4f})")

    (REPORTS_DIR / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, REPORTS_DIR / "training_curves.png")
    print(f"Best val accuracy {best_acc:.4f}, checkpoint at {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
