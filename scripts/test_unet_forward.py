from __future__ import annotations

import torch

from score_matching_snr_reparametrization.unet import ImageScoreUNet


def main() -> None:
    model = ImageScoreUNet(
        image_channels=1,
        base_channels=32,
        time_embedding_dim=64,
    )

    noisy_images = torch.randn(8, 1, 32, 32)
    log_snr_t = torch.randn(8)

    output = model(noisy_images, log_snr_t)

    print("input shape: ", noisy_images.shape)
    print("time shape:  ", log_snr_t.shape)
    print("output shape:", output.shape)


if __name__ == "__main__":
    main()