from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import torch


METHODS = ("gda", "egm", "ppm", "nog", "qpg")
SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_PATH = SCRIPT_DIR / "phaseVIC_exact_joint_actor_game.py"
spec = importlib.util.spec_from_file_location("vic_ant_bank_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
survey = runner.survey


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--force-scale", type=float, default=0.5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protagonist_seed = args.task_id // len(METHODS)
    protagonist_method = METHODS[args.task_id % len(METHODS)]
    survey.SEED = 20260720
    survey.base.seed_everything(20260720)
    info = runner.build_env_info("Ant-v4")
    cfg = survey.WrapperConfig(
        "mujoco_external_force_xz_radial", args.force_scale, "common_nog_bank", 2, info.main_body_id, info.main_body_name
    )
    game = survey.SurveyGame(info, cfg, 20260720)

    attackers: dict[int, torch.Tensor] = {}
    for attacker_seed in range(5):
        payload = torch.load(
            args.root / f"seed_{attacker_seed}" / "nog" / "checkpoint_00200000.pt",
            map_location=survey.DEVICE,
            weights_only=False,
        )
        attackers[attacker_seed] = payload["phi"].to(survey.DEVICE)

    rows: list[dict] = []
    run = args.root / f"seed_{protagonist_seed}" / protagonist_method
    for checkpoint in sorted(run.glob("checkpoint_*.pt")):
        step = int(checkpoint.stem.split("_")[-1])
        payload = torch.load(checkpoint, map_location=survey.DEVICE, weights_only=False)
        theta = payload["theta"].to(survey.DEVICE)
        for attacker_seed, phi in attackers.items():
            robust = game.evaluate_actor(theta, phi, args.episodes)["return"]
            rows.append(
                {
                    "protagonist_method": protagonist_method,
                    "protagonist_seed": protagonist_seed,
                    "checkpoint_step": step,
                    "protagonist_env_steps": 5000 + step,
                    "attacker_method": "nog",
                    "attacker_seed": attacker_seed,
                    "episodes": args.episodes,
                    "robust_return": robust,
                }
            )
    output_csv = args.output / f"{protagonist_method}_seed_{protagonist_seed}.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / f"{protagonist_method}_seed_{protagonist_seed}.json").write_text(
        json.dumps(
            {
                "protagonist_method": protagonist_method,
                "protagonist_seed": protagonist_seed,
                "attacker_bank": "five final noG adversaries, seeds 0-4",
                "reward": "native Gymnasium Ant-v4 undiscounted return",
                "rows": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
