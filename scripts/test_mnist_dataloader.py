from __future__ import annotations

from pathlib import Path

from score_matching_snr_reparametrization.image_data import get_mnist_dataloader
from score_matching_snr_reparametrization.image_plotting import plot_image_grid


FIGURE_DIR = Path("outputs/figures")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    dataloader = get_mnist_dataloader(batch_size=32)

    images, labels = next(iter(dataloader))

    print("images shape:", images.shape)
    print("labels shape:", labels.shape)
    print("labels:", labels)
    print("image min/max:", images.min().item(), images.max().item())

    plot_image_grid(
        images,
        num_rows=4,
        num_cols=8,
        title="Real MNIST training images",
        save_path=FIGURE_DIR / "real_mnist_grid.png",
    )

if __name__ == "__main__":
    main()