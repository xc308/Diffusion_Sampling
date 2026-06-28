from __future__ import annotations

import torch
from torch import Tensor

import math

def sample_ring_of_gaussians(
    num_samples: int,
    num_modes: int = 8,
    radius: float = 4.0,
    mode_std: float = 0.15,
) -> Tensor:
    """Sample from an isotropic mixture of Gaussians arranged on a ring.

    Args:
        num_samples: Total number of points to draw.
        num_modes: Number of Gaussian components on the ring.
        radius: Distance from the origin to each component mean.
        mode_std: Standard deviation of each component.

    Returns:
        Tensor of shape (num_samples, 2).
    """
    mode_indices = torch.randint(low = 0, high = num_modes, size = (num_samples,)) # create a 1D tensor from range 0, to num_modes-1 int
    angles = mode_indices.float() * (2 * math.pi / num_modes) # divide 360 deg into num_modes to get angles for each mode
    means = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1) # radius * torch.cos(angles) is the x-coords, pair each coods x and y,return a tensor of num_samples rows, each row has two cols for x, y coords
    return means + mode_std * torch.randn(num_samples, 2) # tensors with each sample from a gaussian mixture

