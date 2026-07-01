from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from score_matching_snr_reparametrization.embeddings import FourierFeatureEmbedding


class ResBlock(nn.Module):
    """A simple residual block with time embedding injection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
    ) -> None:
        super().__init__()

        self.act = nn.SiLU()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_proj = nn.Linear(time_dim, out_channels)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(
        self,
        x: Tensor,
        time_emb: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Feature map of shape (B, C, H, W).
            time_emb: Time embedding of shape (B, time_dim).

        Returns:
            Tensor of shape (B, out_channels, H, W).
        """
        h = self.conv1(self.act(x))  # (B, out_channels, H, W)

        # Project time embedding to channel dimension, then broadcast over H and W.
        time_bias = self.time_proj(self.act(time_emb))[:, :, None, None] # from (B, out_channels) to (B, out_channels, 1, 1)
                                                                        # so can be added to h shape (B, out_channels, H, W)
        h = h + time_bias

        h = self.conv2(self.act(h))

        return h + self.skip(x) # original image x + processed with time jected image h



class ImageScoreUNet(nn.Module):
    """A small U-Net for image diffusion on MNIST-like images.
        Consists of time embedding, down path, bottleneck, up path, final image prediction
    """

    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 32,
        time_embedding_dim: int = 64,
    ) -> None:
        super().__init__()

        self.time_embedding = FourierFeatureEmbedding(time_embedding_dim)

        time_features_dim = 2 * time_embedding_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_features_dim, base_channels * 4), 
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )

        time_dim = base_channels * 4

        # Initial projection from image channels to feature channels.
        self.init_conv = nn.Conv2d(
            image_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        # Down path.
        self.down1 = ResBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.down2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # Bottleneck.
        self.middle = ResBlock(base_channels * 4, base_channels * 4, time_dim)

        # Up path.
        self.upsample2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up2 = ResBlock(base_channels * 4, base_channels * 2, time_dim)

        self.upsample1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up1 = ResBlock(base_channels * 2, base_channels, time_dim)

        self.final_conv = nn.Conv2d(
            base_channels,
            image_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        noisy_image: Tensor,
        log_snr_t: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the U-Net.

        Args:
            noisy_image: Tensor of shape (B, 1, H, W).
            log_snr_t: Tensor of shape (B,).
            condition: Unused here; included for interface compatibility.

        Returns:
            Tensor of shape (B, 1, H, W), matching noisy_image.
        """
        del condition  # unused, but kept for compatibility with diffusion_loss

        time_features = self.time_embedding(log_snr_t)
        time_emb = self.time_mlp(time_features)

        x0 = self.init_conv(noisy_image)               # (B, 32, 32, 32)

        x1 = self.down1(x0, time_emb)                  # (B, 32, 32, 32)
        skip1 = x1

        x2 = self.downsample1(x1)                      # (B, 64, 16, 16)
        x3 = self.down2(x2, time_emb)                  # (B, 64, 16, 16)
        skip2 = x3

        x4 = self.downsample2(x3)                      # (B, 128, 8, 8)
        x4 = self.middle(x4, time_emb)                 # (B, 128, 8, 8)

        x = self.upsample2(x4)                         # (B, 64, 16, 16)
        x = torch.cat([x, skip2], dim=1)               # (B, 128, 16, 16)
        x = self.up2(x, time_emb)                      # (B, 64, 16, 16)

        x = self.upsample1(x)                          # (B, 32, 32, 32)
        x = torch.cat([x, skip1], dim=1)               # (B, 64, 32, 32)
        x = self.up1(x, time_emb)                      # (B, 32, 32, 32)

        return self.final_conv(x)                      # (B, 1, 32, 32)
    


    ## Summary:
    # Image size change:
        # 32 * 32         32 * 32
        #    |               |
        # 16 * 16         16 * 16
            # -- 8 * 8 -- 

    # Chanel size change:
        # 1             1
        # |             |
        # 32            32
        # |             |
        # 64            64
            # -- 128 --