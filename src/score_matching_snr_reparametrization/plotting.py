from __future__ import annotations

import torch
from torch import Tensor

from pathlib import Path

import matplotlib.pyplot as plt

from score_matching_snr_reparametrization.cos_schedule import CosineNoiseSchedule



def plot_2d_samples(
    samples: Tensor,
    title: str,
    save_path: str | Path | None = None,
) -> None:
    """Scatter plot of 2D samples."""
    samples = samples.detach().cpu()

    plt.figure(figsize=(4, 4))
    plt.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.5)
    plt.title(title)
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.show()



def plot_noise_schedule(
    schedule: CosineNoiseSchedule,
    save_path: str | Path | None = None,
) -> None:
    """Visualise log-SNR, alpha_t, and sigma_t."""

    t_grid = torch.linspace(0.001, 0.999, 200)

    log_snr_grid = schedule.log_snr(t_grid)
    alpha_grid, sigma_grid = schedule.alpha_sigma(log_snr_grid)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axes[0].plot(t_grid, log_snr_grid)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"$\lambda_t$")
    axes[0].set_title("Log-SNR")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_grid, alpha_grid, label=r"$\alpha_t$")
    axes[1].plot(t_grid, sigma_grid, label=r"$\sigma_t$")
    axes[1].set_xlabel("t")
    axes[1].legend()
    axes[1].set_title("Schedule coefficients")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.show()



def plot_real_vs_generated(
    real_samples: Tensor,
    generated_samples: Tensor,
    save_path: str | Path | None = None,
) -> None:
    """Compare real samples and generated samples."""
    real_samples = real_samples.detach().cpu()
    generated_samples = generated_samples.detach().cpu()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].scatter(real_samples[:, 0], real_samples[:, 1], s=4, alpha=0.5)
    axes[0].set_title(r"Real $p_d(\mathbf{x})$")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(generated_samples[:, 0], generated_samples[:, 1], s=4, alpha=0.5)
    axes[1].set_title("Diffusion samples")
    axes[1].axis("equal")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.show()

