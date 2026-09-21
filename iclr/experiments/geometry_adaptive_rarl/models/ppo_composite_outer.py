from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from utils.callbacks import JointTransitionRecorderCallback


EPS = 1e-12


class JointActionCritic(nn.Module):
    def __init__(self, state_dim: int, protagonist_dim: int, adversary_dim: int):
        super().__init__()
        width = 128
        self.net = nn.Sequential(
            nn.Linear(state_dim + protagonist_dim + adversary_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, states: torch.Tensor, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((states, u, w), dim=-1)).squeeze(-1)


class ActorMeanView(nn.Module):
    """Differentiable deterministic-mean view of an SB3 MLP actor."""

    def __init__(self, policy, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.policy_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        mean = self.action_net(self.policy_net(observations))
        return torch.maximum(torch.minimum(mean, self.action_high), self.action_low)


class FlatActorGame:
    def __init__(self, model, critic: JointActionCritic, device: torch.device):
        self.device = device
        self.critic = critic
        self.protagonist = ActorMeanView(
            model.protagonist.policy,
            model.protagonist.action_space.low,
            model.protagonist.action_space.high,
        ).to(device)
        self.adversary = ActorMeanView(
            model.adversary.policy,
            model.adversary.action_space.low,
            model.adversary.action_space.high,
        ).to(device)
        self.p_names, self.p_shapes, self.p_sizes = self._layout(self.protagonist)
        self.a_names, self.a_shapes, self.a_sizes = self._layout(self.adversary)
        self.p_dim = sum(self.p_sizes)

    @staticmethod
    def _layout(module: nn.Module):
        named = list(module.named_parameters())
        return (
            [name for name, _ in named],
            [parameter.shape for _, parameter in named],
            [parameter.numel() for _, parameter in named],
        )

    @staticmethod
    def _flat(module: nn.Module) -> torch.Tensor:
        return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])

    @staticmethod
    def _mapping(flat: torch.Tensor, names, shapes, sizes):
        chunks = torch.split(flat, sizes)
        return {name: chunk.reshape(shape) for name, shape, chunk in zip(names, shapes, chunks)}

    def current_z(self) -> torch.Tensor:
        return torch.cat((self._flat(self.protagonist), self._flat(self.adversary))).to(self.device)

    def actions(self, z: torch.Tensor, states: torch.Tensor):
        p_flat, a_flat = z[: self.p_dim], z[self.p_dim :]
        u = functional_call(
            self.protagonist,
            self._mapping(p_flat, self.p_names, self.p_shapes, self.p_sizes),
            (states,),
        )
        w = functional_call(
            self.adversary,
            self._mapping(a_flat, self.a_names, self.a_shapes, self.a_sizes),
            (states,),
        )
        return u, w

    def objective(self, z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        u, w = self.actions(z, states)
        return self.critic(states, u, w).mean()

    def field(self, z: torch.Tensor, states: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        objective = self.objective(z, states)
        gradient = torch.autograd.grad(objective, z, create_graph=create_graph)[0]
        return torch.cat((-gradient[: self.p_dim], gradient[self.p_dim :]))

    def geometry(self, states: torch.Tensor, probes: int = 4) -> dict[str, float]:
        z = self.current_z().requires_grad_(True)
        field = self.field(z, states, create_graph=True)
        _, g = torch.autograd.functional.jvp(
            lambda value: self.field(value, states, create_graph=True),
            z,
            field.detach(),
            create_graph=False,
            strict=False,
        )
        grad_energy = torch.autograd.grad(0.5 * torch.dot(field, field), z)[0]
        sf = 0.5 * (g + grad_energy)
        wf = 0.5 * (g - grad_energy)
        same_sq = 0.0
        cross_sq = 0.0
        generator = torch.Generator(device=z.device).manual_seed(1701)
        for block in (0, 1):
            for _ in range(probes):
                direction = torch.zeros_like(z)
                sl = slice(0, self.p_dim) if block == 0 else slice(self.p_dim, None)
                signs = torch.randint(0, 2, direction[sl].shape, generator=generator, device=z.device)
                direction[sl] = signs.to(z.dtype).mul_(2).sub_(1)
                direction[sl] /= torch.linalg.norm(direction[sl]) + EPS
                _, response = torch.autograd.functional.jvp(
                    lambda value: self.field(value, states, create_graph=True),
                    z,
                    direction,
                    create_graph=False,
                    strict=False,
                )
                own = response[: self.p_dim] if block == 0 else response[self.p_dim :]
                other = response[self.p_dim :] if block == 0 else response[: self.p_dim]
                same_sq += float(torch.dot(own, own).item())
                cross_sq += float(torch.dot(other, other).item())
        return {
            "F_norm": float(torch.linalg.norm(field).item()),
            "G_norm": float(torch.linalg.norm(g).item()),
            "SF_norm": float(torch.linalg.norm(sf).item()),
            "WF_norm": float(torch.linalg.norm(wf).item()),
            "WF_over_SF": float(torch.linalg.norm(wf).item() / (torch.linalg.norm(sf).item() + EPS)),
            "cross_to_same_ratio": float(np.sqrt(cross_sq / max(same_sq, EPS))),
        }


class PPOCompositeOuterDiagnostic:
    """Train a joint critic and audit the synchronized PPO actor field without QP."""

    def __init__(
        self,
        output: str | Path,
        device: str = "cpu",
        warmup_outer: int = 10,
        critic_updates: int = 100,
        batch_size: int = 256,
        state_batch_size: int = 256,
        mode: str = "diagnostic",
        actor_lr: float = 1e-6,
    ):
        if mode not in {"diagnostic", "gda"}:
            raise ValueError("PPO outer bridge initially permits only diagnostic or gda mode")
        self.output = Path(output)
        self.device = torch.device(device)
        self.warmup_outer = int(warmup_outer)
        self.critic_updates = int(critic_updates)
        self.batch_size = int(batch_size)
        self.state_batch_size = int(state_batch_size)
        self.mode = mode
        self.actor_lr = float(actor_lr)
        self.rng = np.random.default_rng(982_451_653)
        self.protagonist_callback = JointTransitionRecorderCallback()
        self.critic = None
        self.critic_optimizer = None
        self.rows: list[dict[str, float | int | str]] = []

    def _ensure_critic(self, arrays: dict[str, np.ndarray]) -> None:
        if self.critic is not None:
            return
        devices = [] if self.device.type == "cpu" else [self.device.index or 0]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(982_451_653)
            self.critic = JointActionCritic(
                arrays["obs"].shape[1], arrays["u"].shape[1], arrays["w"].shape[1]
            ).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _train_critic(self, arrays: dict[str, np.ndarray]) -> float:
        count = len(arrays["obs"])
        losses = []
        for _ in range(self.critic_updates):
            indices = self.rng.integers(0, count, size=min(self.batch_size, count))
            prediction = self.critic(
                self._tensor(arrays["obs"][indices]),
                self._tensor(arrays["u"][indices]),
                self._tensor(arrays["w"][indices]),
            )
            target = self._tensor(arrays["mc_return"][indices]).reshape(-1)
            loss = torch.nn.functional.smooth_l1_loss(prediction, target)
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.critic_optimizer.step()
            losses.append(float(loss.detach().item()))
        return float(np.mean(losses))

    def _correlation(self, arrays: dict[str, np.ndarray]) -> float:
        with torch.no_grad():
            prediction = self.critic(
                self._tensor(arrays["obs"]), self._tensor(arrays["u"]), self._tensor(arrays["w"])
            ).cpu().numpy()
        target = arrays["mc_return"].reshape(-1)
        if np.std(prediction) < EPS or np.std(target) < EPS:
            return float("nan")
        return float(np.corrcoef(prediction, target)[0, 1])

    @staticmethod
    def _copy_flat(module: nn.Module, flat: torch.Tensor) -> None:
        offset = 0
        with torch.no_grad():
            for parameter in module.parameters():
                size = parameter.numel()
                parameter.copy_(flat[offset : offset + size].reshape_as(parameter))
                offset += size

    def __call__(self, model, iteration: int) -> None:
        all_arrays = self.protagonist_callback.arrays()
        latest = self.protagonist_callback.arrays(latest_only=True)
        if not all_arrays or not latest:
            return
        self._ensure_critic(all_arrays)
        train_arrays = self.protagonist_callback.arrays(exclude_latest=True)
        if not train_arrays:
            train_arrays = all_arrays
        loss = self._train_critic(train_arrays)
        corr = self._correlation(latest)
        row: dict[str, float | int | str] = {
            "outer_iteration": iteration,
            "protagonist_steps": int(model.protagonist.num_timesteps),
            "mode": self.mode,
            "replay_samples": len(all_arrays["obs"]),
            "critic_loss": loss,
            "corr_Q_MC": corr,
            "applied_update": 0,
        }
        if iteration >= self.warmup_outer and np.isfinite(corr) and corr >= 0.6:
            game = FlatActorGame(model, self.critic, self.device)
            states = self._tensor(latest["obs"][: self.state_batch_size])
            geometry = game.geometry(states)
            row.update(geometry)
            if self.mode == "gda":
                z = game.current_z().requires_grad_(True)
                field = game.field(z, states, create_graph=False).detach()
                updated = z.detach() - self.actor_lr * field
                self._copy_flat(game.protagonist, updated[: game.p_dim])
                self._copy_flat(game.adversary, updated[game.p_dim :])
                row["applied_update"] = 1
        self.rows.append(row)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        keys: list[str] = []
        for item in self.rows:
            for key in item:
                if key not in keys:
                    keys.append(key)
        with self.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)
