from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.algorithms import ALGOS_RARL
from scripts.full_policy_followup_common import SavedRun, load_rarl_for_eval, load_yaml
from utils.callbacks import normalized_opponent_policy


METHODS = ["sgd", "egm", "ppm", "proposed_noG", "proposed_qp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Standard RARL cross-play and dynamics evaluation")
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--target-seed", required=True, type=int)
    parser.add_argument("--target-method", required=True, choices=METHODS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--curve-episodes", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--multipliers", nargs="+", type=float, default=[0.5, 0.75, 1.0, 1.25, 1.5])
    return parser.parse_args()


def find_saved_run(suite_root: Path, seed: int, method: str) -> SavedRun:
    task = suite_root / f"seed_{seed}" / method
    matches = list(task.glob(f"**/HalfCheetah-v4_*/HalfCheetah-v4/metadata.zip"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one saved model for seed={seed}, method={method}; found {len(matches)}")
    model_dir = matches[0].parent
    return SavedRun(
        method=method,
        tag=f"seed_{seed}",
        run_root=model_dir.parent,
        model_dir=model_dir,
        args_data=load_yaml(model_dir / "args.yml"),
        config_data=load_yaml(model_dir / "config.yml"),
    )


def evaluate_episodes(actor, vec_env, episodes: int, seed: int) -> np.ndarray:
    returns = []
    vec_env.seed(seed)
    obs = vec_env.reset()
    episode_return = 0.0
    while len(returns) < episodes:
        action, _ = actor.predict(obs, deterministic=True)
        obs, reward, done, _ = vec_env.step(action)
        episode_return += float(reward[0])
        if bool(done[0]):
            returns.append(episode_return)
            episode_return = 0.0
    return np.asarray(returns, dtype=np.float64)


def set_dynamics(vec_env, body_mass: np.ndarray, geom_friction: np.ndarray) -> None:
    wrapper = vec_env.venv.envs[0]
    wrapper._base_env.model.body_mass[:] = body_mass
    wrapper._base_env.model.geom_friction[:] = geom_friction


def load_adversary_bank(suite_root: Path, device: str):
    bank = []
    for seed in range(5):
        for method in METHODS:
            source = find_saved_run(suite_root, seed, method)
            adversary = ALGOS_RARL["ppo"].load(str(source.model_dir / "adversary.zip"), device=device)
            with (source.model_dir / "vecnormalize.pkl").open("rb") as handle:
                source_normalizer = pickle.load(handle)
            bank.append((seed, method, adversary, source_normalizer))
    return bank


def protagonist_checkpoints(target: SavedRun, device: str):
    checkpoints = []
    for path in sorted(target.run_root.glob("pro_model_*_steps.zip")):
        match = re.fullmatch(r"pro_model_(\d+)_steps\.zip", path.name)
        if match is None:
            continue
        steps = int(match.group(1))
        norm_path = target.run_root / f"pro_model_vecnormalize_{steps}_steps.pkl"
        if not norm_path.exists():
            raise FileNotFoundError(f"Missing checkpoint normalizer {norm_path}")
        actor = ALGOS_RARL["ppo"].load(str(path), device=device)
        with norm_path.open("rb") as handle:
            normalizer = pickle.load(handle)
        checkpoints.append((steps, actor, normalizer))
    if not checkpoints:
        raise FileNotFoundError(f"No protagonist checkpoints under {target.run_root}")
    return checkpoints


def install_active_normalizer(vec_env, source_normalizer) -> None:
    vec_env.obs_rms = source_normalizer.obs_rms
    vec_env.clip_obs = source_normalizer.clip_obs
    vec_env.epsilon = source_normalizer.epsilon


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root)
    target = find_saved_run(suite_root, args.target_seed, args.target_method)
    model, vec_env = load_rarl_for_eval(target, adv_impact="force", adv_strength=1.0, device=args.device)
    wrapper = vec_env.venv.envs[0]
    baseline_mass = wrapper._base_env.model.body_mass.copy()
    baseline_friction = wrapper._base_env.model.geom_friction.copy()
    rows: list[dict[str, object]] = []
    bank = load_adversary_bank(suite_root, args.device)

    # Common-attacker cross-play. Every target faces the exact same final-policy bank.
    set_dynamics(vec_env, baseline_mass, baseline_friction)
    vec_env.set_attr("operating_mode", "protagonist")
    for attacker_seed, attacker_method, adversary, source_normalizer in bank:
        vec_env.set_attr("_adv_policy", normalized_opponent_policy(adversary.policy, source_normalizer))
        values = evaluate_episodes(
            model.protagonist, vec_env, args.episodes,
            seed=700_000 + attacker_seed * 100 + METHODS.index(attacker_method),
        )
        for episode, value in enumerate(values):
            rows.append({
                "target_seed": args.target_seed,
                "target_method": args.target_method,
                "evaluation": "common_attacker_bank",
                "checkpoint_steps": model.protagonist.num_timesteps,
                "parameter": "attacker",
                "multiplier": np.nan,
                "attacker_seed": attacker_seed,
                "attacker_method": attacker_method,
                "episode": episode,
                "return": value,
            })

    # Common-bank convergence uses each checkpoint's own observation statistics.
    for steps, actor, checkpoint_normalizer in protagonist_checkpoints(target, args.device):
        install_active_normalizer(vec_env, checkpoint_normalizer)
        for attacker_seed, attacker_method, adversary, source_normalizer in bank:
            vec_env.set_attr("_adv_policy", normalized_opponent_policy(adversary.policy, source_normalizer))
            values = evaluate_episodes(
                actor, vec_env, args.curve_episodes,
                seed=900_000 + steps + attacker_seed * 100 + METHODS.index(attacker_method),
            )
            for episode, value in enumerate(values):
                rows.append({
                    "target_seed": args.target_seed,
                    "target_method": args.target_method,
                    "evaluation": "common_attacker_convergence",
                    "checkpoint_steps": steps,
                    "parameter": "attacker",
                    "multiplier": np.nan,
                    "attacker_seed": attacker_seed,
                    "attacker_method": attacker_method,
                    "episode": episode,
                    "return": value,
                })

    with (target.model_dir / "vecnormalize.pkl").open("rb") as handle:
        final_normalizer = pickle.load(handle)
    install_active_normalizer(vec_env, final_normalizer)

    # Standard dynamics-generalization tests use no active force adversary.
    vec_env.set_attr("operating_mode", None)
    for parameter in ("body_mass", "geom_friction"):
        for multiplier in args.multipliers:
            mass = baseline_mass.copy()
            friction = baseline_friction.copy()
            if parameter == "body_mass":
                mass[1:] *= multiplier  # Keep the inertialess world body unchanged.
            else:
                friction *= multiplier
            set_dynamics(vec_env, mass, friction)
            values = evaluate_episodes(
                model.protagonist, vec_env, args.episodes,
                seed=800_000 + (0 if parameter == "body_mass" else 10_000) + int(multiplier * 100),
            )
            for episode, value in enumerate(values):
                rows.append({
                    "target_seed": args.target_seed,
                    "target_method": args.target_method,
                    "evaluation": "dynamics_sweep",
                    "checkpoint_steps": model.protagonist.num_timesteps,
                    "parameter": parameter,
                    "multiplier": multiplier,
                    "attacker_seed": np.nan,
                    "attacker_method": "none",
                    "episode": episode,
                    "return": value,
                })

    set_dynamics(vec_env, baseline_mass, baseline_friction)
    vec_env.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
