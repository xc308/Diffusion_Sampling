from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from score_matching_snr_reparametrization.parameterisation import Parameterisation
from score_matching_snr_reparametrization.cos_schedule import CosineNoiseSchedule
from score_matching_snr_reparametrization.device import DEVICE


@torch.no_grad()
def ancestral_sampling_step(
    noisy_sample: Tensor,
    network_output: Tensor,
    log_snr_t: Tensor, # scalar tensor
    log_snr_s: Tensor, # scalar tensor
    parameterisation: Parameterisation,
    clip_to: tuple[float, float] | None = (-1.0, 1.0),
) -> tuple[Tensor, Tensor]:
    """One reverse step of ancestral sampling, from noise level t down to s < t.

    Args:
        noisy_sample: Current iterate z_t, shape (batch, ...features).
        network_output: The network's prediction at (z_t, log_snr_t).
        log_snr_t: Current (more noisy) log-SNR, scalar tensor.
        log_snr_s: Target (less noisy) log-SNR, scalar tensor.
        parameterisation: Whether the network predicts "v" or "eps".
        clip_to: Optional clamp range for the predicted clean sample, useful
            when data is normalised to [-1, 1]. Set to None to disable.

    Returns:
        posterior_mean: The mean of the reverse Gaussian.
        posterior_variance: The (scalar) variance of the reverse Gaussian.
    """

    # -expm1(x) = 1 - exp(x); positive when x < 0 (i.e. log_snr_t < log_snr_s).
    c = -torch.expm1(log_snr_t - log_snr_s)

    alpha_t = torch.sqrt(torch.sigmoid(log_snr_t))
    alpha_s = torch.sqrt(torch.sigmoid(log_snr_s))
    sigma_t = torch.sqrt(torch.sigmoid(-log_snr_t))
    sigma_s = torch.sqrt(torch.sigmoid(-log_snr_s))

    # Decode the network output into a prediction of the clean sample x.
    if parameterisation == "v":
        predicted_clean = alpha_t * noisy_sample - sigma_t * network_output
    elif parameterisation == "eps":
        predicted_clean = (noisy_sample - sigma_t * network_output) / alpha_t
    else:
        raise ValueError(f"Unknown parameterisation: {parameterisation!r}")

    if clip_to is not None:
        predicted_clean = predicted_clean.clamp(*clip_to)

    posterior_mean = alpha_s * (
        noisy_sample * (1.0 - c) / alpha_t + c * predicted_clean
    )
    posterior_variance = (sigma_s ** 2) * c
    return posterior_mean, posterior_variance



@torch.no_grad()
def generate_samples(
    score_network: nn.Module,    # trained NN
    schedule: CosineNoiseSchedule,
    sample_shape: tuple[int, ...],
    num_steps: int = 100,
    parameterisation: Parameterisation = "v",
    condition: Tensor | None = None,
    clip_to: tuple[float, float] | None = None,
    device: torch.device = DEVICE,
) -> Tensor:
    """Generate a batch of samples by running ancestral sampling end-to-end.

    Args:
        score_network: Trained denoising network.
        schedule: Noise schedule.
        sample_shape: Shape of the batch to generate, e.g. (1024, 2).
        num_steps: Number of denoising steps. More steps => more accurate.
        parameterisation: Which target the network outputs.
        condition: Optional conditioning info.
        clip_to: Optional clamp on intermediate clean predictions.
        device: Device to draw the initial noise on.

    Returns:
        Tensor of shape sample_shape, drawn (approximately) from p_d.
    """
    # Initialise from pure Gaussian noise.
    sample = torch.randn(sample_shape, device=device)

    # We integrate from t=1 (pure noise) down to t=0 (clean data) on a uniform grid.
    for step in reversed(range(1, num_steps + 1)):
        t_now = torch.tensor(step / num_steps, device=device)
        t_next = torch.tensor((step - 1) / num_steps, device=device)

        log_snr_t = schedule.log_snr(t_now)
        log_snr_s = schedule.log_snr(t_next)

        # The network expects a per-sample log-SNR; broadcast scalar to batch.
        log_snr_t_batched = log_snr_t.expand(sample.shape[0])
        network_output = score_network(sample, log_snr_t_batched, condition) # now run the forward in FourierFeatureEmbedding and ScoreMLP 

        mean, variance = ancestral_sampling_step(
            noisy_sample=sample,
            network_output=network_output,
            log_snr_t=log_snr_t,
            log_snr_s=log_snr_s,
            parameterisation=parameterisation,
            clip_to=clip_to,
        )

        if step > 1:
            # Stochastic step: add noise scaled by sqrt(variance).
            sample = mean + torch.randn_like(mean) * torch.sqrt(variance)
        else:
            # Last step: just take the mean (deterministic) to avoid noisy outputs.
            sample = mean

    return sample