from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Aggregate long-run QP+G optimizer diagnostics in chunks")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--envs", nargs="+", default=["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--chunksize", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    accumulators: dict[tuple[str, int, str], dict[str, float]] = {}
    columns = [
        "role", "gamma", "G_contribution_norm", "G_over_update_norm",
        "zero_update_flag", "cap_active_flag", "boundary_solution_flag",
        "interior_solution_flag", "same_minibatch_total_loss_change",
    ]
    for env_id in args.envs:
        for seed in args.seeds:
            path = root / env_id / f"seed_{seed}" / "proposed_qp" / "proposed_qp_diagnostics.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            for chunk in pd.read_csv(path, usecols=lambda name: name in columns, chunksize=args.chunksize):
                for role, role_rows in chunk.groupby("role"):
                    key = (env_id, seed, str(role))
                    acc = accumulators.setdefault(
                        key,
                        {
                            "rows": 0.0,
                            "gamma_active": 0.0,
                            "gamma_sum": 0.0,
                            "g_contribution_sum": 0.0,
                            "g_over_update_sum": 0.0,
                            "zero_updates": 0.0,
                            "cap_active": 0.0,
                            "boundary": 0.0,
                            "interior": 0.0,
                            "realized_decrease": 0.0,
                        },
                    )
                    n = float(len(role_rows))
                    gamma = role_rows["gamma"].fillna(0.0).to_numpy(dtype=float)
                    acc["rows"] += n
                    acc["gamma_active"] += float((np.abs(gamma) > 1e-12).sum())
                    acc["gamma_sum"] += float(gamma.sum())
                    acc["g_contribution_sum"] += float(role_rows["G_contribution_norm"].fillna(0.0).sum())
                    acc["g_over_update_sum"] += float(role_rows["G_over_update_norm"].fillna(0.0).sum())
                    acc["zero_updates"] += float(role_rows["zero_update_flag"].fillna(0.0).sum())
                    acc["cap_active"] += float(role_rows["cap_active_flag"].fillna(0.0).sum())
                    acc["boundary"] += float(role_rows["boundary_solution_flag"].fillna(0.0).sum())
                    acc["interior"] += float(role_rows["interior_solution_flag"].fillna(0.0).sum())
                    acc["realized_decrease"] += float(
                        (role_rows["same_minibatch_total_loss_change"].fillna(np.inf) < 0.0).sum()
                    )

    rows = []
    for (env_id, seed, role), acc in sorted(accumulators.items()):
        n = acc["rows"]
        rows.append(
            {
                "env_id": env_id,
                "seed": seed,
                "role": role,
                "updates": int(n),
                "gamma_active_fraction": acc["gamma_active"] / n,
                "gamma_mean": acc["gamma_sum"] / n,
                "G_contribution_mean": acc["g_contribution_sum"] / n,
                "G_over_update_mean": acc["g_over_update_sum"] / n,
                "zero_update_fraction": acc["zero_updates"] / n,
                "cap_active_fraction": acc["cap_active"] / n,
                "boundary_fraction": acc["boundary"] / n,
                "interior_fraction": acc["interior"] / n,
                "realized_decrease_fraction": acc["realized_decrease"] / n,
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
