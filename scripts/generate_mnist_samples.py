from __future__ import annotations

from pathlib import Path

import torch

from score_matching_snr_reparametrization.device import (DEVICE, set_seed, print_device)
from score_matching_snr_reparametrization.cos_schedule import CosineNoiseSchedule
from score_matching_snr_reparametrization.unet import ImageScoreUNet
from score_matching_snr_reparametrization.sampling import generate_samples
from score_matching_snr_reparametrization.image_plotting import plot_image_grid


CHECKPOINT_PATH = Path("outputs/checkpoints/mnist_unet_v.pt")
FIGURE_DIR = Path("outputs/figures")


def main() -> None:
    set_seed()
    print_device()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    schedule = CosineNoiseSchedule()

    diffusion_network = ImageScoreUNet(
        image_channels=1,
        base_channels=32,
        time_embedding_dim=64,
    ).to(DEVICE)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    diffusion_network.load_state_dict(state_dict)
    diffusion_network.eval()

    print(f"Loaded checkpoint from: {CHECKPOINT_PATH}")
    print("Generating MNIST samples...")

    generated_samples = generate_samples(
        score_network=diffusion_network,
        schedule=schedule,
        sample_shape=(32, 1, 32, 32),
        num_steps=100,
        parameterisation="v",
        device=DEVICE,
    )

    plot_image_grid(
        generated_samples,
        num_rows=4,
        num_cols=8,
        title="Generated MNIST samples",
        save_path=FIGURE_DIR / "mnist_generated_samples.png",
    )

    print(f"Saved generated samples to: {FIGURE_DIR / 'mnist_generated_samples.png'}")


if __name__ == "__main__":
    main()