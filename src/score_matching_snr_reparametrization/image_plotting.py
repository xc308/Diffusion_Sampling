from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def unnormalize_images(images: Tensor) -> Tensor:
    """Convert images from [-1, 1] back to [0, 1] for plotting."""
    return (images + 1.0) / 2.0


def plot_image_grid(
    images: Tensor,
    *,
    num_rows: int = 4,
    num_cols: int = 8,
    title: str = "Image samples",
    save_path: str | Path | None = None,
) -> None:
    """Plot a grid of grayscale images.

    Args:
        images: Tensor of shape (B, 1, H, W), usually in [-1, 1].
        num_rows: Number of rows in the image grid.
        num_cols: Number of columns in the image grid.
        title: Figure title.
        save_path: Optional path for saving the figure.
    """
    images = images.detach().cpu()    # detech from gradient computation graph, move numbers to cpu
    images = unnormalize_images(images)
    images = images.clamp(0.0, 1.0)   # matplot.lib expect value [0, 1]

    num_images = min(images.shape[0], num_rows * num_cols) # images in a batch maybe fewer than num_rows * num_cols

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols, num_rows),
    )

    axes = axes.flatten() # axes was a 2d array, but turn it into 1D [ax00,..., ax37], for indexing axes[i]

    for i in range(num_rows * num_cols):
        ax = axes[i]     # pick the ith subplot box
        ax.axis("off")   # no x-/y-axis ticks

        if i < num_images: # only draw image that exists
            image = images[i, 0] # image shape (B, 1, H, W), select ith image, and 1st chanel, result (H, W)
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.show()