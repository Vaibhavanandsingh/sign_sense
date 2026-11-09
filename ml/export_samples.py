import argparse
import random
from collections import defaultdict

from torchvision import datasets

from ml.config import DATA_DIR, SAMPLES_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GTSRB test images for the demo")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    test_ds = datasets.GTSRB(DATA_DIR, split="test", download=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for old in SAMPLES_DIR.glob("*.png"):
        old.unlink()

    by_class = defaultdict(list)
    for idx, (_, label) in enumerate(test_ds._samples):
        by_class[label].append(idx)

    rng = random.Random(args.seed)
    indices = [idx for label in sorted(by_class) for idx in rng.sample(by_class[label], args.per_class)]
    for i, idx in enumerate(indices):
        image, label = test_ds[idx]
        image.save(SAMPLES_DIR / f"{i:03d}_class{label:02d}.png")
    print(f"Saved {len(indices)} samples ({args.per_class} per class) to {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
