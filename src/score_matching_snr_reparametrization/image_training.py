from __future__ import annotations

from itertools import cycle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from score_matching_snr_reparametrization.cos_schedule import CosineNoiseSchedule
from score_matching_snr_reparametrization.loss import diffusion_loss
from score_matching_snr_reparametrization.parameterisation import Parameterisation


def train_image_diffusion_model(
    score_network: nn.Module,
    schedule: CosineNoiseSchedule,
    dataloader: DataLoader,
    *,
    num_training_steps: int = 2_000,
    learning_rate: float = 2e-4,
    parameterisation: Parameterisation = "v",
    log_every: int = 100,
) -> list[float]:
    """Train an image diffusion model.

    Args:
        score_network: U-Net model that predicts v or eps.
        schedule: Noise schedule used to compute log-SNR, alpha, and sigma.
        dataloader: DataLoader yielding image batches.
        num_training_steps: Number of optimisation steps.
        learning_rate: AdamW learning rate.
        parameterisation: Either "v" or "eps".
        log_every: Print average loss every log_every steps.

    Returns:
        List of loss values, one per training step.
    """
    optimizer = torch.optim.AdamW(score_network.parameters(), lr=learning_rate)

    loss_history: list[float] = []

    score_network.train()

    data_iterator = cycle(dataloader) # if dataloader reaches the end, start from the beginning, gives an endless stream of batch

    device = next(score_network.parameters()).device

    for step in range(num_training_steps):
        clean_batch, _ = next(data_iterator) # get MNIST batch, (B, 1, 32, 32)
        clean_batch = clean_batch.to(device) # moves the clean batch to the same device where the model lives

        loss = diffusion_loss(
            score_network,
            clean_batch,
            schedule,
            parameterisation=parameterisation,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (step + 1) % log_every == 0:
            recent_loss = sum(loss_history[-log_every:]) / log_every
            print(
                f"  step {step + 1:5d}/{num_training_steps} "
                f"| loss {recent_loss:.4f}"
            )

    return loss_history