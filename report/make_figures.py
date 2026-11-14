"""Render the matplotlib figures used in the project report into report/figures/."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

from ml.config import CLASS_NAMES, REPORTS_DIR, SAMPLES_DIR
from ml.signs import describe

FIG_DIR = Path(__file__).resolve().parent / "figures"
STATS = json.loads((FIG_DIR / "stats.json").read_text())

CATEGORY_COLORS = {
    "speed_limit": "#1f4e79",
    "end_of_restriction": "#7f7f7f",
    "prohibitory": "#c00000",
    "priority": "#bf8f00",
    "yield": "#ed7d31",
    "stop": "#8b0000",
    "no_entry": "#d9534f",
    "warning": "#f0ad4e",
    "mandatory": "#2e75b6",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})


def class_distribution() -> None:
    counts = STATS["class_counts"]["train"]
    colors = [CATEGORY_COLORS[describe(i)["category"]] for i in range(43)]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(range(43), counts, color=colors, width=0.8)
    ax.set_xticks(range(43))
    ax.set_xticklabels(range(43), fontsize=7)
    ax.set_xlim(-0.8, 42.8)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Training images")
    used = []
    for i in range(43):
        category = describe(i)["category"]
        if category not in used:
            used.append(category)
    ax.legend(handles=[Patch(color=CATEGORY_COLORS[c], label=c.replace("_", " ")) for c in used],
              ncol=9, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.2, columnspacing=1.0)
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG_DIR / "class_distribution.png")
    plt.close(fig)


def sample_grid() -> None:
    first = {}
    for path in sorted(SAMPLES_DIR.glob("*.png")):
        class_id = int(path.stem.split("class")[1])
        first.setdefault(class_id, path)
    cols, rows = 11, 4
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.3))
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx < 43:
            ax.imshow(Image.open(first[idx]).convert("RGB").resize((64, 64)))
            ax.set_title(str(idx), fontsize=9, pad=2)
    fig.subplots_adjust(wspace=0.08, hspace=0.3)
    fig.savefig(FIG_DIR / "sample_grid.png")
    plt.close(fig)


def training_curves() -> None:
    history = json.loads((REPORTS_DIR / "history.json").read_text())
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.4))
    ax_loss.plot(epochs, history["train_loss"], "o-", ms=3, label="Training")
    ax_loss.plot(epochs, history["val_loss"], "s-", ms=3, label="Validation")
    ax_loss.set(title="(a) Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    ax_acc.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", ms=3, label="Training")
    ax_acc.plot(epochs, [a * 100 for a in history["val_acc"]], "s-", ms=3, label="Validation")
    ax_acc.set(title="(b) Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in (ax_loss, ax_acc):
        ax.set_xticks(list(epochs))
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png")
    plt.close(fig)


def per_class_f1() -> None:
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text())
    f1 = [metrics["per_class_f1"][name] for name in CLASS_NAMES]
    colors = ["#c00000" if v < 0.96 else "#1f4e79" for v in f1]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(range(43), f1, color=colors, width=0.8)
    ax.axhline(metrics["macro_f1"], color="#555555", ls="--", lw=1,
               label=f"Macro F1 = {metrics['macro_f1']:.4f}")
    ax.set_ylim(0.85, 1.005)
    ax.set_xticks(range(43))
    ax.set_xticklabels(range(43), fontsize=7)
    ax.set_xlim(-0.8, 42.8)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("F1 score")
    ax.legend(handles=[Patch(color="#1f4e79", label="F1 >= 0.96"), Patch(color="#c00000", label="F1 < 0.96"),
                       ax.get_lines()[0]], frameon=False, fontsize=8, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG_DIR / "per_class_f1.png")
    plt.close(fig)


if __name__ == "__main__":
    class_distribution()
    sample_grid()
    training_curves()
    per_class_f1()
    print("figures written to", FIG_DIR)
