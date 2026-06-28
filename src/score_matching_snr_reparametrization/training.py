from __future__ import annotations

import torch
import torch.nn as nn

from score_matching_snr_reparametrization.cos_schedule import CosineNoiseSchedule
from score_matching_snr_reparametrization.parameterisation import Parameterisation
from score_matching_snr_reparametrization.data import sample_ring_of_gaussians
from score_matching_snr_reparametrization.device import DEVICE
from score_matching_snr_reparametrization.loss import diffusion_loss

def train_diffusion_model(
    score_network: nn.Module,
    schedule: CosineNoiseSchedule,
    *, # the following must state arguments 
    num_training_steps: int = 5_000,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    parameterisation: Parameterisation = "v",
    log_every: int = 500,
) -> list[float]:

    """Train `score_network` by minimising the diffusion loss on the toy ring."""
    optimizer = torch.optim.Adam(score_network.parameters(), lr=learning_rate) # update score_network.parameters()
    loss_history: list[float] = []

    score_network.train()   # turn on training mode, can allow random drop, opposite to eval() mode
    
    for step in range(num_training_steps):
        clean_batch = sample_ring_of_gaussians(batch_size).to(DEVICE) # a batch of samples of clean data

        loss = diffusion_loss(
            score_network=score_network, # state argument explictly
            clean_batch=clean_batch,
            schedule=schedule,
            parameterisation=parameterisation,
        )

        optimizer.zero_grad(set_to_none=True) # clear gradient, set_to_none=True to save memory by setting grad to NONE
        loss.backward()  # compute the gradients of loss wrt the trainable weights
        optimizer.step() # update the gradients

        loss_history.append(loss.item()) # append the loss scalar not tensor
        
        if (step + 1) % log_every == 0:  # 
            recent = sum(loss_history[-log_every:]) / log_every # averaged loss over the lastest 500 steps, more smooth loss
            print(f"  step {step + 1:5d}/{num_training_steps} | loss {recent:.4f}")

    return loss_history
