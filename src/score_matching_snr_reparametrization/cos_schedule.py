from __future__ import annotations

import math

import torch
from torch import Tensor




class CosineNoiseSchedule:
    """Cosine log-SNR schedule (Nichol & Dhariwal, 2021), with optional shift.

    Maps a time variable t in [0, 1] to a log signal-to-noise ratio lambda_t,
    and from there to the diffusion coefficients alpha_t and sigma_t.

    The optional `shift` argument implements the "shifted cosine" from the
    Simple Diffusion paper, which adapts the schedule to the spatial resolution
    of the data when training on images.
    """

    def __init__(
        self,
        log_snr_min: float = -15.0,
        log_snr_max: float = 15.0,
        shift: float = 0.0,
    ) -> None:
        self.log_snr_min = log_snr_min
        self.log_snr_max = log_snr_max
        self.shift = shift
        # Pre-compute the t-range bounds so log_snr is in [log_snr_min, log_snr_max].
        self._t_lo = math.atan(math.exp(-0.5 * log_snr_max))
        self._t_hi = math.atan(math.exp(-0.5 * log_snr_min))


    def log_snr(self, t: Tensor) -> Tensor:
        """Compute lambda_t = log(alpha_t^2 / sigma_t^2) for t in [0, 1].

        Args:
            t: Tensor of any shape with values in [0, 1].

        Returns:
            Tensor of the same shape, the (clipped) log-SNR.
        """
        if torch.any((t < 0) | (t > 1)):
            raise ValueError("t must be in [0, 1].")
        clipped_t = self._t_lo + t * (self._t_hi - self._t_lo)
        cosine_log_snr = -2.0 * torch.log(torch.tan(clipped_t))
        return cosine_log_snr + self.shift


    def alpha_sigma(self, log_snr_t: Tensor) -> tuple[Tensor, Tensor]:
        """Convert log-SNR to (alpha_t, sigma_t).

        Uses the numerically stable identities
            alpha_t^2 = sigmoid( log_snr_t ),
            sigma_t^2 = sigmoid(-log_snr_t).
        """
        alpha_t = torch.sqrt(torch.sigmoid(log_snr_t))
        sigma_t = torch.sqrt(torch.sigmoid(-log_snr_t))
        return alpha_t, sigma_t
