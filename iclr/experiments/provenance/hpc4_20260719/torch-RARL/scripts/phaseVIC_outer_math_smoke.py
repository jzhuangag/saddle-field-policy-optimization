from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.ppo_composite_outer import FlatActorGame


class Policy(nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.mlp_extractor = SimpleNamespace(policy_net=nn.Identity())
        self.action_net = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.action_net.weight.fill_(weight)


class BilinearCritic(nn.Module):
    def forward(self, states, u, w):
        return (u * w).sum(dim=-1)


def agent(weight: float):
    return SimpleNamespace(
        policy=Policy(weight),
        action_space=SimpleNamespace(
            low=np.asarray([-10.0], dtype=np.float32),
            high=np.asarray([10.0], dtype=np.float32),
        ),
    )


def main() -> None:
    model = SimpleNamespace(protagonist=agent(2.0), adversary=agent(3.0))
    game = FlatActorGame(model, BilinearCritic(), torch.device("cpu"))
    states = torch.ones((16, 1))
    z = game.current_z().requires_grad_(True)
    field = game.field(z, states, create_graph=True)
    expected = torch.tensor([-3.0, 2.0])
    if not torch.allclose(field, expected, atol=1e-6):
        raise AssertionError(f"bilinear field mismatch: {field} != {expected}")
    _, g = torch.autograd.functional.jvp(
        lambda value: game.field(value, states, create_graph=True),
        z,
        field.detach(),
        create_graph=False,
    )
    expected_g = torch.tensor([-2.0, -3.0])
    if not torch.allclose(g, expected_g, atol=1e-6):
        raise AssertionError(f"G=JF mismatch: {g} != {expected_g}")
    geometry = game.geometry(states, probes=8)
    if geometry["WF_over_SF"] < 1e6:
        raise AssertionError(f"bilinear game should be skew dominated: {geometry}")
    if geometry["cross_to_same_ratio"] < 1e6:
        raise AssertionError(f"bilinear game should be cross-player dominated: {geometry}")
    print({"field": field.tolist(), "G": g.tolist(), **geometry})


if __name__ == "__main__":
    main()
