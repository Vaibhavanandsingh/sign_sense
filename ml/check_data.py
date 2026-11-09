from collections import Counter

from ml.config import IMG_SIZE, NUM_CLASSES
from ml.data import get_dataloaders, get_test_loader
from ml.utils import get_device


def main() -> None:
    print(f"device: {get_device()}")
    train_loader, val_loader = get_dataloaders()
    test_loader = get_test_loader()

    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        images, labels = next(iter(loader))
        print(f"{name:<5} {len(loader.dataset):>6} images | batch {tuple(images.shape)} | "
              f"labels {labels.min().item()}-{labels.max().item()} | "
              f"pixel mean {images.mean():+.3f} std {images.std():.3f}")
        assert images.shape[1:] == (3, IMG_SIZE, IMG_SIZE)

    counts = Counter(label for _, label in train_loader.dataset.dataset._samples)
    assert len(counts) == NUM_CLASSES
    print(f"classes {len(counts)} | smallest {min(counts.values())} | largest {max(counts.values())} images")


if __name__ == "__main__":
    main()
