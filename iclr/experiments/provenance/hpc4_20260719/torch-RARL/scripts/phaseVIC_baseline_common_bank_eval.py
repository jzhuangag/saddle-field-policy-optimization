from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from full_policy_followup_common import (
    SavedRun,
    load_rarl_for_eval,
    load_yaml,
)
from utils.callbacks import normalized_opponent_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Baseline-only common-adversary evaluation")
    parser.add_argument("--target", type=Path, required=True, help="Target RARL model directory")
    parser.add_argument("--attacker", action="append", required=True, help="LABEL=RARL_MODEL_DIR")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def saved_run(label: str, model_dir: Path) -> SavedRun:
    return SavedRun(
        method="adam",
        tag=label,
        run_root=model_dir.parent,
        model_dir=model_dir,
        args_data=load_yaml(model_dir / "args.yml"),
        config_data=load_yaml(model_dir / "config.yml"),
    )


def evaluate(actor, vec_env, episodes: int, seed: int) -> list[float]:
    vec_env.seed(seed)
    obs = vec_env.reset()
    returns: list[float] = []
    running = 0.0
    while len(returns) < episodes:
        action, _ = actor.predict(obs, deterministic=True)
        obs, reward, done, _ = vec_env.step(action)
        running += float(reward[0])
        if bool(done[0]):
            returns.append(running)
            running = 0.0
    return returns


class ConstantOpponent:
    def __init__(self, action: tuple[float, float]) -> None:
        self.action = action

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        value = torch.as_tensor(self.action, dtype=observation.dtype, device=observation.device)
        return value.unsqueeze(0).expand(observation.shape[0], -1)


class RadialRandomOpponent:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        rho = self.rng.uniform(0.0, 1.0, size=observation.shape[0])
        angle = self.rng.uniform(-np.pi, np.pi, size=observation.shape[0])
        action = np.column_stack((rho * np.cos(angle), rho * np.sin(angle)))
        return torch.as_tensor(action, dtype=observation.dtype, device=observation.device)


def main() -> None:
    args = parse_args()
    target = saved_run("target", args.target)
    target_model, target_env = load_rarl_for_eval(
        target, adv_impact="force", adv_strength=args.force_scale, device=args.device
    )
    rows: list[dict[str, float | int | str]] = []

    target_env.set_attr("operating_mode", None)
    for episode, value in enumerate(evaluate(target_model.protagonist, target_env, args.episodes, 710_000)):
        rows.append({"attacker": "clean", "episode": episode, "return": value})

    diagnostic_attackers = {
        "random_radial": RadialRandomOpponent(715_000),
        "fixed_pos_x": ConstantOpponent((1.0, 0.0)),
        "fixed_neg_x": ConstantOpponent((-1.0, 0.0)),
        "fixed_pos_z": ConstantOpponent((0.0, 1.0)),
        "fixed_neg_z": ConstantOpponent((0.0, -1.0)),
    }
    target_env.set_attr("operating_mode", "protagonist")
    for index, (label, attacker) in enumerate(diagnostic_attackers.items()):
        target_env.set_attr("_adv_policy", attacker)
        for episode, value in enumerate(
            evaluate(target_model.protagonist, target_env, args.episodes, 715_000 + index * 1000)
        ):
            rows.append({"attacker": label, "episode": episode, "return": value})

    attacker_resources = []
    for index, spec in enumerate(args.attacker):
        label, raw_dir = spec.split("=", 1)
        source = saved_run(label, Path(raw_dir))
        attacker_model, attacker_env = load_rarl_for_eval(
            source, adv_impact="force", adv_strength=args.force_scale, device=args.device
        )
        attacker_resources.append(attacker_env)
        target_env.set_attr("operating_mode", "protagonist")
        target_env.set_attr(
            "_adv_policy", normalized_opponent_policy(attacker_model.adversary.policy, attacker_env)
        )
        for episode, value in enumerate(
            evaluate(target_model.protagonist, target_env, args.episodes, 720_000 + index * 1000)
        ):
            rows.append({"attacker": label, "episode": episode, "return": value})

    target_env.close()
    for env in attacker_resources:
        env.close()
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    summary = frame.groupby("attacker")["return"].agg(["mean", "std", "min", "max", "count"])
    summary["degradation_from_clean"] = float(summary.loc["clean", "mean"]) - summary["mean"]
    summary.to_csv(args.output.with_name(args.output.stem + "_summary.csv"))


if __name__ == "__main__":
    main()
