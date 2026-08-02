"""Create the compact paper-facing plot from a completed formal run."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(SCRIPT_DIRECTORY / ".mplconfig")
)

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("QP+G", "noG", "EGM")
STYLES = {
    "QP+G": ("black", "-"),
    "noG": ("tab:green", "-"),
    "EGM": ("tab:purple", "--"),
}


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "seed",
            "batch_size",
            "step",
            "hard_br_return",
            "field_norm",
        ):
            row[key] = float(row[key])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("output_pdf", type=Path)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    rows = load_rows(args.result_directory / "curves.csv")
    selected_rows = [
        row for row in rows if row["batch_size"] == args.batch_size
    ]
    seeds = sorted({int(row["seed"]) for row in selected_rows})
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.4,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.4,
            "legend.fontsize": 7.0,
        }
    )
    figure, axis = plt.subplots(1, 1, figsize=(3.45, 1.55))
    for metric, title in (
        ("hard_br_return", "hard-BR return (higher is better)"),
    ):
        for method in METHODS:
            trajectories = []
            x = None
            for seed in seeds:
                curve = sorted(
                    (
                        row
                        for row in selected_rows
                        if row["method"] == method and row["seed"] == seed
                    ),
                    key=lambda row: row["step"],
                )
                x = np.asarray([row["step"] for row in curve])
                trajectories.append([row[metric] for row in curve])
            array = np.asarray(trajectories, dtype=float)
            mean = array.mean(axis=0)
            sem = array.std(axis=0, ddof=1) / math.sqrt(len(seeds))
            color, linestyle = STYLES[method]
            axis.plot(
                x,
                mean,
                color=color,
                linestyle=linestyle,
                linewidth=1.55,
                label=method,
            )
            axis.fill_between(
                x, mean - sem, mean + sem, color=color, alpha=0.12
            )
        axis.set_title(title)
        axis.grid(alpha=0.22)
    axis.set_xlabel("simultaneous stochastic update")
    axis.legend(frameon=False, ncol=3, loc="best")
    figure.tight_layout(pad=0.6)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_pdf, bbox_inches="tight")
    figure.savefig(args.output_pdf.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
