from __future__ import annotations

import numpy as np
import torch

# Reproducibility
SEED = 0


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def print_device() -> None:
    """Print the device currently used by PyTorch."""
    print(f"Using device: {DEVICE}")