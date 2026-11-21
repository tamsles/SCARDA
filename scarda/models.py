"""Model wrapper (SPEC §7): the original SCARCE MLP plus a `features` accessor."""
from __future__ import annotations

import torch

from models import mlp_model


class MLP(mlp_model):
    """784 -> 500 -> ReLU -> K, identical forward to models.mlp_model."""

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = x.view(-1, self.num_flat_features(x))
        return self.relu1(self.fc1(out))
