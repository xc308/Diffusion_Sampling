from __future__ import annotations

import torch
from pathlib import Path

from score_matching_snr_reparametrization.cos_schedule import CosineNoiseSchedule
from score_matching_snr_reparametrization.model import ScoreMLP
from score_matching_snr_reparametrization.sampling import generate_samples
from score_matching_snr_reparametrization.data import sample_ring_of_gaussians
from score_matching_snr_reparametrization.device import (DEVICE, print_device, set_seed)
from score_matching_snr_reparametrization.plotting import plot_real_vs_generated


FIGURE_DIR = Path("outputs/figures")
CHECKPOINT_PATH = Path("outputs/checkpoints/score_mlp_v.pt")


def main() -> None:
    set_seed()
    print_device()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Re-create the same schedule and model architecture.
    schedule = CosineNoiseSchedule()
    diffusion_network = ScoreMLP(data_dim=2).to(DEVICE)

    # 2. Load trained weights.
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    diffusion_network.load_state_dict(state_dict)

    diffusion_network.eval()

    # 3. Generate samples.
    generated_samples = generate_samples(
        score_network=diffusion_network,
        schedule=schedule,
        sample_shape=(2000, 2),
        num_steps=100,
        parameterisation="v",
        device=DEVICE,
    ).cpu()


    # 4. Compare against real samples.
    real_samples = sample_ring_of_gaussians(2000)

    plot_real_vs_generated(
        real_samples,
        generated_samples,
        save_path=FIGURE_DIR / "real_vs_generated.png",
    )
    


if __name__ == "__main__": # only run main() when asked to execute
    main()