import os

import torch


def get_device() -> torch.device:
    if override := os.getenv("DEVICE"):
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
