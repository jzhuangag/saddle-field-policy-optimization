from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_PATH = SCRIPT_DIR / "phaseVIC_exact_joint_actor_game.py"
spec = importlib.util.spec_from_file_location("vic_exact_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
survey = runner.survey


def evaluate(game, mode: str, force_scale: float, episodes: int, seed: int, direction=None):
    env = survey.base.gym.make(game.info.env_name)
    returns, lengths = [], []
    rng = np.random.default_rng(seed + 44000)
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + 9000 + episode)
        total, steps, done = 0.0, 0, False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=survey.DTYPE, device=survey.DEVICE).unsqueeze(0)
            with torch.no_grad():
                u = game.actor_protagonist(game.theta, obs_t).squeeze(0)
                if mode == "clean":
                    w = torch.zeros(2, dtype=survey.DTYPE, device=survey.DEVICE)
                elif mode == "random":
                    w = torch.as_tensor(rng.uniform(-1.0, 1.0, size=2), dtype=survey.DTYPE, device=survey.DEVICE)
                else:
                    w = torch.as_tensor(direction, dtype=survey.DTYPE, device=survey.DEVICE)
            old_scale = game.cfg.strength_value
            object.__setattr__(game.cfg, "strength_value", force_scale)
            next_obs, reward, done, _, _, err = game.step_env(env, u, w)
            object.__setattr__(game.cfg, "strength_value", old_scale)
            if err:
                raise RuntimeError("force calibration environment step failed")
            total += reward
            steps += 1
            obs = next_obs
        returns.append(total)
        lengths.append(steps)
    env.close()
    return float(np.mean(returns)), float(np.std(returns, ddof=1)), float(np.mean(lengths))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env", default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.5, 1.0, 2.5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    survey.SEED = args.seed
    survey.base.seed_everything(args.seed)
    info = runner.build_env_info(args.env)
    cfg = survey.WrapperConfig("mujoco_external_force_xz_radial", 1.0, "calibration", 2, info.main_body_id, info.main_body_name)
    game = survey.SurveyGame(info, cfg, args.seed)
    runner.load_pretrained(game, args.checkpoint)

    clean_mean, clean_std, clean_len = evaluate(game, "clean", 0.0, args.episodes, args.seed)
    directions = [(math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)) for k in range(8)]
    rows = []
    for strength in args.strengths:
        random_mean, random_std, random_len = evaluate(game, "random", strength, args.episodes, args.seed)
        direction_returns = [evaluate(game, "direction", strength, args.episodes, args.seed, d)[0] for d in directions]
        worst = float(min(direction_returns))
        rows.append({
            "force_scale": strength,
            "clean_return": clean_mean,
            "clean_std": clean_std,
            "random_force_return": random_mean,
            "random_force_std": random_std,
            "random_degradation_fraction": (clean_mean - random_mean) / max(abs(clean_mean), 1e-12),
            "worst_direction_return": worst,
            "worst_direction_degradation_fraction": (clean_mean - worst) / max(abs(clean_mean), 1e-12),
            "clean_episode_length": clean_len,
            "random_episode_length": random_len,
        })
    with (args.output / "force_calibration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    eligible = [r for r in rows if 0.05 <= r["worst_direction_degradation_fraction"] <= 0.40 and r["random_episode_length"] >= 0.8 * clean_len]
    decision = min(eligible, key=lambda r: r["force_scale"])["force_scale"] if eligible else None
    (args.output / "force_calibration.json").write_text(json.dumps({
        "selection_rule": "smallest strength with 5-40% worst-direction degradation and >=80% clean episode length",
        "selected_force_scale": decision,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"selected_force_scale": decision, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
