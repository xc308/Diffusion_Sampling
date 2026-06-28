from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from score_matching_snr_reparametrization.embeddings import FourierFeatureEmbedding

class ScoreMLP(nn.Module):
    """A small MLP that maps (noisy_sample, log_snr, condition) -> prediction.

    The condition is optional; when None, the model is unconditional.
    """

    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embedding_dim: int = 64,
        condition_dim: int = 0,
    ) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.time_embedding = FourierFeatureEmbedding(time_embedding_dim) # creates a FourierFeatureEmbedding obj, just the first part is run

        time_features_dim = 2 * time_embedding_dim
        input_dim = data_dim + time_features_dim + condition_dim

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()] # add elements nn.Linear(hidden_dim, hidden_dim), nn.SiLU() into layers list, like layers.extend(), not append
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.network = nn.Sequential(*layers) # * unpack the nn.modules in the layers list, and pass in Sequential module by module, 
                                              # ** upacks both the key and value of a dictionary as arguments for another function who asks for two args


    def forward(
        self,
        noisy_sample: Tensor, # (B, 2)
        log_snr_t: Tensor, # (B, )
        condition: Tensor | None = None,
    ) -> Tensor:
        time_features = self.time_embedding(log_snr_t) # log_snr tensor [B, ], runs the foward part of FourierFeatureEmbedding, creates the features
        components: list[Tensor] = [noisy_sample, time_features]
        if self.condition_dim > 0:
            assert condition is not None, "condition_dim > 0 but condition was None"
            components.append(condition)
        return self.network(torch.cat(components, dim=-1)) # a forward pass of input into the above defined MLP
