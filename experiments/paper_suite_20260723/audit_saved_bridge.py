"""Audit Proposition 1 on the saved final tabular and neural trajectories."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from scipy import stats


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
from markov_game_suite import ENTROPY_TAU, DISCOUNT, game_catalog, hard_game_value


SAVED = {
    "tabular": HERE / "results" / "tabular-exact-gap-20260723-113009" / "curves.csv",
    "neural": HERE / "results" / "neural-journal-four-20260824" / "curves.csv",
}

REPORTED_ENVIRONMENTS = {
    "tabular": {"RPS", "CyclicControl", "FrequencyHopping"},
    "neural": {
        "CyclicControl",
        "FrequencyHopping",
        "RoutingInterdiction",
        "SecurityPatrol",
    },
}


def main():
    catalog = game_catalog()
    values = {}
    maximum_shapley_residual = 0.0
    for name in set().union(*REPORTED_ENVIRONMENTS.values()):
        value, residual, iterations = hard_game_value(catalog[name])
        values[name] = {"value": value, "residual": residual, "iterations": iterations}
        maximum_shapley_residual = max(maximum_shapley_residual, residual)

    entropy_bias = ENTROPY_TAU * (math.log(3) + math.log(3)) / (1.0 - DISCOUNT)
    audits = {}
    multiple_testing = {}
    for mode, path in SAVED.items():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row["environment"] in REPORTED_ENVIRONMENTS[mode]
            ]
        slacks = []
        for row in rows:
            deficiency = values[row["environment"]]["value"] - float(row["hard_br_return"])
            upper = float(row["regularized_gap"]) + entropy_bias
            slacks.append(upper - deficiency)
        audits[mode] = {
            "source": str(path.relative_to(PROJECT)).replace("\\", "/"),
            "reported_environments": sorted(REPORTED_ENVIRONMENTS[mode]),
            "checkpoint_rows": len(rows),
            "minimum_bridge_slack": min(slacks),
            "maximum_bridge_violation": max(0.0, -min(slacks)),
        }
        saved_summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        decisions = []
        for item in saved_summary["decisions"]:
            if item["environment"] not in REPORTED_ENVIRONMENTS[mode]:
                continue
            win_count = int(item.get("br_wins", item.get("br_win_count")))
            raw_p = float(
                stats.binomtest(
                    win_count, 10, p=0.5, alternative="greater"
                ).pvalue
            )
            decisions.append(
                {
                    "environment": item["environment"],
                    "br_win_count": win_count,
                    "raw_sign_p": raw_p,
                }
            )
        order = sorted(range(len(decisions)), key=lambda index: decisions[index]["raw_sign_p"])
        running = 0.0
        for rank, index in enumerate(order):
            running = max(
                running, (len(decisions) - rank) * decisions[index]["raw_sign_p"]
            )
            decisions[index]["holm_adjusted_sign_p"] = min(1.0, running)
        multiple_testing[mode] = decisions

    report = {
        "game_values": values,
        "entropy_bias": entropy_bias,
        "maximum_shapley_residual": maximum_shapley_residual,
        "audits": audits,
        "multiple_testing": multiple_testing,
        "status": "PASS" if all(item["maximum_bridge_violation"] == 0.0 for item in audits.values()) else "FAIL",
    }
    output = HERE / "results" / ("saved-bridge-audit-" + time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("RESULT_DIR=" + str(output))
    print(json.dumps(report, sort_keys=True))


def environments(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["environment"] for row in csv.DictReader(handle)}


if __name__ == "__main__":
    main()
