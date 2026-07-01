from __future__ import annotations

from pathlib import Path
import torch


from score_matching_snr_reparametrization.device import (DEVICE, set_seed, print_device)
from score_matching_snr_reparametrization.image_data import get_mnist_dataloader
from score_matching_snr_reparametrization.cos_schedule import CosineNoiseSchedule
from score_matching_snr_reparametrization.unet import ImageScoreUNet
from score_matching_snr_reparametrization.image_training import train_image_diffusion_model


CHECKPOINT_DIR = Path("outputs/checkpoints")


def main() -> None:
    set_seed()
    print_device()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    dataloader = get_mnist_dataloader(
        batch_size=128,
        image_size=32,
        data_dir="data",
        num_workers=0,
        shuffle=True,
    )

    schedule = CosineNoiseSchedule()

    diffusion_network = ImageScoreUNet(
        image_channels=1,
        base_channels=32,
        time_embedding_dim=64,
    ).to(DEVICE)

    print("Training MNIST U-Net diffusion model:")

    loss_history = train_image_diffusion_model(
        score_network=diffusion_network,
        schedule=schedule,
        dataloader=dataloader,
        num_training_steps=10_000,
        learning_rate=2e-4,
        parameterisation="v",
        log_every=500,
    )

    checkpoint_path = CHECKPOINT_DIR / "mnist_unet_v.pt"
    torch.save(diffusion_network.state_dict(), checkpoint_path)

    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Final loss: {loss_history[-1]:.4f}")


if __name__ == "__main__":
    main()