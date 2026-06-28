from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn

import math




class FourierFeatureEmbedding(nn.Module):
    """Random Fourier features for embedding a continuous scalar (e.g. log-SNR).

    Output is [sin(2*pi*x*W), cos(2*pi*x*W)] where W is a fixed random vector.
    """

    def __init__(self, num_frequencies: int = 64, scale: float = 16.0) -> None:
        super().__init__() # set up a PyTorch NN machinary for this class

        # Buffer (not a Parameter): not trained, but moves with .to(device).
        # create a 1D tensor called self.frequencies with shape (num_frequencies, ), whose numbers are from stdN * scale
        # store this self.frequencies tensor in the model, on device,
        # but it's not trainable parameters
        self.register_buffer("frequencies", torch.randn(num_frequencies) * scale) 


    def forward(self, scalar_input: Tensor) -> Tensor:
        # scalar_input has shape (batch,); produce (batch, 2*num_frequencies).
        # scalar_input[:, None] has shape (batch, 1)
        # self.frequencies[None, :] has shape (1, 64)
        # element-wise mulitiply scalar input with every frequency
        # projected has shape (B, 64)
        projected = scalar_input[:, None] * self.frequencies[None, :] * 2 * math.pi

        # projected.sin(), projected.cos() each has shape (B, 64), concatenate (B, 128)
        return torch.cat([projected.sin(), projected.cos()], dim=-1)
