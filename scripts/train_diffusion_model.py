from __future__ import annotations

import torch
from pathlib import Path

from score_matching_snr_reparametrization.cos_schedule import CosineNoiseSchedule
from score_matching_snr_reparametrization.model import ScoreMLP
from score_matching_snr_reparametrization.training import train_diffusion_model
from score_matching_snr_reparametrization.data import sample_ring_of_gaussians
from score_matching_snr_reparametrization.device import (DEVICE, set_seed, print_device)
from score_matching_snr_reparametrization.plotting import (plot_2d_samples, plot_noise_schedule)


FIGURE_DIR = Path("outputs/figures")
CHECKPOINT_DIR = Path("outputs/checkpoints")

def main() -> None:
    set_seed()
    print_device()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Plot real target data.
    data_samples = sample_ring_of_gaussians(1000)
    plot_2d_samples(
        data_samples,
        title=r"True target distribution $p_d(\mathbf{x})$",
        save_path=FIGURE_DIR / "true_distribution.png",
    )

    # 2. Create and plot schedule.
    schedule = CosineNoiseSchedule()
    plot_noise_schedule(
        schedule,
        save_path=FIGURE_DIR / "noise_schedule.png",
    )

    # 3. Create model.
    diffusion_network = ScoreMLP(data_dim=2).to(DEVICE)


    # 4. Train model.
    print("Training diffusion model (v-parameterisation):")
    loss_history = train_diffusion_model(
        score_network=diffusion_network,
        schedule=schedule,
        num_training_steps=5000,
        parameterisation="v",
    )

    # 5. Save trained model.
    checkpoint_path = CHECKPOINT_DIR / "score_mlp_v.pt"
    torch.save(diffusion_network.state_dict(), checkpoint_path)

    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Final loss: {loss_history[-1]:.4f}")

if __name__ == "__main__": # only run the main() when directly called to do so
    main() 


