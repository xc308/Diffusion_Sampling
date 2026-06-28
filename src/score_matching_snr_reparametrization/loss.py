from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from score_matching_snr_reparametrization.cos_schedule import CosineNoiseSchedule
from score_matching_snr_reparametrization.parameterisation import Parameterisation




def add_noise(
    clean_sample: Tensor,
    alpha_t: Tensor,
    sigma_t: Tensor,
) -> tuple[Tensor, Tensor]:
    """Forward diffusion step: z_t = alpha_t * x + sigma_t * noise.

    Args:
        clean_sample: Clean data x, shape (batch, ...features).
        alpha_t: Per-sample signal coefficient, broadcastable to clean_sample.
        sigma_t: Per-sample noise coefficient, broadcastable to clean_sample.

    Returns:
        noisy_sample: z_t with the same shape as clean_sample.
        noise: The standard-normal noise that was added.
    """
    noise = torch.randn_like(clean_sample)
    noisy_sample = alpha_t * clean_sample + sigma_t * noise
    return noisy_sample, noise



def diffusion_loss( # per sample loss
    score_network: nn.Module,
    clean_batch: Tensor,
    schedule: CosineNoiseSchedule,
    parameterisation: Parameterisation = "v",
    condition: Tensor | None = None,
) -> Tensor:
    """Denoising score-matching loss for a diffusion model.

    For each sample in the batch we draw an independent time t ~ U[0, 1],
    add the corresponding noise, ask the network for either the noise or
    the velocity, convert to a noise prediction, and compute MSE against
    the true noise.

    Args:
        score_network: A module mapping (noisy_sample, log_snr_t, condition) ->
            either predicted noise (parameterisation="eps") or predicted
            velocity (parameterisation="v"), with the same shape as
            noisy_sample.
        clean_batch: Real data x, shape (batch, ...features).
        schedule: Noise schedule providing log_snr and alpha_sigma.
        parameterisation: Which target the network outputs.
        condition: Optional conditioning info passed to the network.

    Returns:
        Scalar MSE loss.
    """

    batch_size = clean_batch.shape[0]
    device = clean_batch.device

    # Sample one t per example, then look up alpha_t, sigma_t, lambda_t.
    t = torch.rand(batch_size, device=device) # a tensor [B, ]
    log_snr_t = schedule.log_snr(t)           # a tensor [B, ]
    alpha_t, sigma_t = schedule.alpha_sigma(log_snr_t)

    # Reshape the schedule scalars so they broadcast over the feature dims.
    feature_shape = (-1,) + (1,) * (clean_batch.ndim - 1)
    alpha_t = alpha_t.view(feature_shape)
    sigma_t = sigma_t.view(feature_shape)

    # Forward noising.
    noisy_sample, true_noise = add_noise(clean_batch, alpha_t, sigma_t)

    # Network forward pass. call the nn function, run its foward part by feeding required input
    network_output = score_network(noisy_sample, log_snr_t, condition)

    # Convert whatever the network predicts into a noise prediction so we can
    # compare against `true_noise` with the same loss in both parameterisations.
    if parameterisation == "v":
        predicted_noise = sigma_t * noisy_sample + alpha_t * network_output
    elif parameterisation == "eps":
        predicted_noise = network_output
    else:
        raise ValueError(f"Unknown parameterisation: {parameterisation!r}")

    return F.mse_loss(predicted_noise, true_noise)
