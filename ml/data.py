import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ml.config import DATA_DIR, IMG_SIZE, MEAN, STD


def build_transform(train: bool) -> transforms.Compose:
    steps = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        steps += [
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


def _loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def get_dataloaders(batch_size: int = 128, val_fraction: float = 0.1,
                    num_workers: int = 2, seed: int = 42) -> tuple[DataLoader, DataLoader]:
    # Two views of the same training set so validation images are not augmented.
    train_view = datasets.GTSRB(DATA_DIR, split="train", transform=build_transform(True), download=True)
    val_view = datasets.GTSRB(DATA_DIR, split="train", transform=build_transform(False), download=True)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_view), generator=generator).tolist()
    n_val = int(len(indices) * val_fraction)

    train_ds = Subset(train_view, indices[n_val:])
    val_ds = Subset(val_view, indices[:n_val])
    return (
        _loader(train_ds, batch_size, True, num_workers),
        _loader(val_ds, batch_size, False, num_workers),
    )


def get_test_loader(batch_size: int = 256, num_workers: int = 2) -> DataLoader:
    test_ds = datasets.GTSRB(DATA_DIR, split="test", transform=build_transform(False), download=True)
    return _loader(test_ds, batch_size, False, num_workers)
