"""Model wrapper (SPEC §7): the original SCARCE MLP plus a `features` accessor."""
from __future__ import annotations

import torch

from models import mlp_model


class MLP(mlp_model):
    """784 -> 500 -> ReLU -> K, identical forward to models.mlp_model."""

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = x.view(-1, self.num_flat_features(x))
        return self.relu1(self.fc1(out))


def build_model(input_shape, cfg) -> MLP:
    """Seeded initialization (init_seed = seed, SPEC §4.5)."""
    torch.manual_seed(cfg.seed)
    return MLP(input_dim=int(torch.Size(input_shape).numel()), hidden_dim=cfg.hidden, output_dim=cfg.K)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x: torch.Tensor, device, bs: int = 4096) -> torch.Tensor:
    """Logits of `x` in eval mode, in chunks; restores the model's train/eval mode."""
    was_training = model.training
    model.eval()
    out = torch.cat([model(x[i:i + bs].to(device)) for i in range(0, len(x), bs)])
    model.train(was_training)
    return out
