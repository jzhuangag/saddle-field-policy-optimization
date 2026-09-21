from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_PATH = SCRIPT_DIR / "phaseVIC_exact_joint_actor_game.py"
spec = importlib.util.spec_from_file_location("vic_exact_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
survey = runner.survey


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", default="HalfCheetah-v4")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    survey.SEED = args.seed
    survey.base.seed_everything(args.seed)
    info = runner.build_env_info(args.env)
    cfg = survey.WrapperConfig("mujoco_external_force_xz_radial", 0.5, "pretrain_eval", 2, info.main_body_id, info.main_body_name)
    game = survey.SurveyGame(info, cfg, args.seed)

    training = pd.read_csv(args.root / "pretrain_convergence.csv")
    warmup_steps = int(training["total_clean_steps"].iloc[0] - training["step"].iloc[0])
    rows = []
    for checkpoint in sorted(args.root.glob("pretrain_checkpoint_*.pt")):
        match = re.fullmatch(r"pretrain_checkpoint_(\d+)\.pt", checkpoint.name)
        if match is None:
            continue
        step = int(match.group(1))
        payload = torch.load(checkpoint, map_location=survey.DEVICE, weights_only=False)
        theta = payload["theta"].to(survey.DEVICE)
        clean = game.evaluate_actor(theta, None, args.episodes)["return"]
        rows.append({"env": args.env, "actor_training_steps": step, "total_clean_env_steps": warmup_steps + step, "clean_return": clean})
    evaluated = pd.DataFrame(rows)
    evaluated.to_csv(args.output / "pretrain_checkpoint_clean_evaluation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(training.total_clean_steps / 1000.0, training.train_episode_return, color="#2878B5", marker="o")
    axes[0].set_title("Clean protagonist collection return")
    axes[0].set_xlabel("Total clean environment steps (thousands)")
    axes[0].set_ylabel("10-episode moving mean return")
    axes[0].grid(alpha=0.2)
    axes[1].plot(evaluated.total_clean_env_steps / 1000.0, evaluated.clean_return, color="#C53D32", marker="o")
    axes[1].set_title("Deterministic clean evaluation")
    axes[1].set_xlabel("Total clean environment steps (thousands)")
    axes[1].set_ylabel(f"Undiscounted return ({args.episodes} episodes)")
    axes[1].grid(alpha=0.2)
    fig.suptitle(f"{args.env} protagonist-only pretraining")
    fig.savefig(args.output / "pretrain_convergence.png", dpi=200)
    fig.savefig(args.output / "pretrain_convergence.pdf")


if __name__ == "__main__":
    main()
