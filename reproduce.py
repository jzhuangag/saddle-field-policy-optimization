#!/usr/bin/env python3
"""Verify and reconstruct the released journal artifacts.

The fast modes operate on frozen CSV/JSON results. Full experiment rerun
commands are documented in REPRODUCIBILITY.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "output" / "data" / "final_experiment_manifest.json"
NEURAL_RESULTS = (
    ROOT
    / "experiments"
    / "paper_suite_20260723"
    / "results"
    / "neural-journal-four-20260824"
)
TABULAR_RESULTS = (
    ROOT
    / "experiments"
    / "paper_suite_20260723"
    / "results"
    / "tabular-exact-gap-20260723-113009"
)
STOCHASTIC_PROTOCOL = (
    ROOT
    / "experiments"
    / "stochastic_oracle_validation_20260727"
    / "results"
    / "formal-CyclicControl-dice-20260727-124224"
    / "protocol.json"
)
STOCHASTIC_TABLE = (
    ROOT / "output" / "data" / "vi_d_finite_trajectory_table.json"
)
NEURAL_ENVIRONMENTS = {
    "CyclicControl",
    "FrequencyHopping",
    "RoutingInterdiction",
    "SecurityPatrol",
}
TABULAR_ENVIRONMENTS = {"RPS", "CyclicControl", "FrequencyHopping"}


class VerificationError(RuntimeError):
    """Raised when a released artifact fails an integrity check."""


def canonical_text_bytes(path: Path) -> bytes:
    """Return text bytes after normalizing CRLF and CR line endings to LF."""

    raw = path.read_bytes()
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def canonical_sha256(path: Path) -> str:
    return hashlib.sha256(canonical_text_bytes(path)).hexdigest()


def load_json(path: Path):
    if not path.is_file():
        raise VerificationError("missing required JSON file: {}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def csv_environments(path: Path):
    if not path.is_file():
        raise VerificationError("missing required CSV file: {}".format(path))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is None or "environment" not in rows.fieldnames:
            raise VerificationError(
                "CSV has no environment column: {}".format(path)
            )
        return {row["environment"] for row in rows}


def require_environment_set(label: str, actual, expected) -> None:
    actual_set = set(actual)
    expected_set = set(expected)
    if actual_set != expected_set:
        raise VerificationError(
            "{} environments are {}; expected {}".format(
                label, sorted(actual_set), sorted(expected_set)
            )
        )


def verify_manifest(manifest) -> None:
    if manifest.get("schema") != "journal-artifact-manifest/1":
        raise VerificationError("unexpected or missing manifest schema")
    frozen_sources = manifest.get("frozen_sources")
    if not isinstance(frozen_sources, dict) or not frozen_sources:
        raise VerificationError("manifest has no frozen_sources entries")

    root_resolved = ROOT.resolve()
    for relative, metadata in frozen_sources.items():
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(root_resolved)
        except ValueError as error:
            raise VerificationError(
                "manifest path escapes repository root: {}".format(relative)
            ) from error
        if not path.is_file():
            raise VerificationError("missing manifest source: {}".format(relative))
        canonical = canonical_text_bytes(path)
        actual_hash = hashlib.sha256(canonical).hexdigest()
        expected_hash = str(metadata.get("sha256", "")).lower()
        if actual_hash != expected_hash:
            raise VerificationError(
                "SHA-256 mismatch for {}: {} != {}".format(
                    relative, actual_hash, expected_hash
                )
            )
        expected_bytes = metadata.get("canonical_bytes")
        if expected_bytes is None or len(canonical) != int(expected_bytes):
            raise VerificationError(
                "canonical byte-count mismatch for {}: {} != {}".format(
                    relative, len(canonical), expected_bytes
                )
            )
        print("[ok] {}".format(relative))


def verify_environment_scope(manifest) -> None:
    require_environment_set(
        "neural curves",
        csv_environments(NEURAL_RESULTS / "curves.csv"),
        NEURAL_ENVIRONMENTS,
    )
    require_environment_set(
        "neural diagnostics",
        csv_environments(NEURAL_RESULTS / "diagnostics.csv"),
        NEURAL_ENVIRONMENTS,
    )
    neural_summary = load_json(NEURAL_RESULTS / "summary.json")
    require_environment_set(
        "neural summary protocol",
        neural_summary.get("protocol", {}).get("environments", []),
        NEURAL_ENVIRONMENTS,
    )
    require_environment_set(
        "manifest neural decisions",
        [row.get("environment") for row in manifest.get("vi_c_decisions", [])],
        NEURAL_ENVIRONMENTS,
    )

    require_environment_set(
        "tabular curves",
        csv_environments(TABULAR_RESULTS / "curves.csv"),
        TABULAR_ENVIRONMENTS,
    )
    tabular_summary = load_json(TABULAR_RESULTS / "summary.json")
    require_environment_set(
        "tabular summary protocol",
        tabular_summary.get("protocol", {}).get("environments", []),
        TABULAR_ENVIRONMENTS,
    )

    protocol = load_json(STOCHASTIC_PROTOCOL)
    stochastic_environment = protocol.get("environment")
    if stochastic_environment not in NEURAL_ENVIRONMENTS:
        raise VerificationError(
            "unexpected finite-trajectory environment: {}".format(
                stochastic_environment
            )
        )
    print("[ok] formal environment sets")


def verify_stochastic_table() -> None:
    table = load_json(STOCHASTIC_TABLE)
    source = table.get("source", {})
    relative = source.get("path")
    expected_hash = str(source.get("sha256", "")).lower()
    if not relative or not expected_hash:
        raise VerificationError("finite-trajectory table has incomplete source metadata")
    path = (ROOT / relative).resolve()
    if not path.is_file():
        raise VerificationError(
            "missing finite-trajectory table source: {}".format(relative)
        )
    actual_hash = canonical_sha256(path)
    if actual_hash != expected_hash:
        raise VerificationError(
            "finite-trajectory source SHA-256 mismatch: {} != {}".format(
                actual_hash, expected_hash
            )
        )
    expected_bytes = source.get("canonical_bytes")
    actual_bytes = len(canonical_text_bytes(path))
    if expected_bytes is None or actual_bytes != int(expected_bytes):
        raise VerificationError(
            "finite-trajectory source canonical byte-count mismatch: {} != {}".format(
                actual_bytes, expected_bytes
            )
        )
    if table.get("all_assertions_passed") is not True:
        raise VerificationError("finite-trajectory table assertions did not pass")
    print("[ok] finite-trajectory table source and assertions")


def verify_required_files() -> None:
    required = [
        ROOT / "main.tex",
        ROOT / "refs.bib",
        ROOT / "RARL_final.pdf",
        ROOT / "experiments" / "paper_suite_20260723" / "journal_games.py",
        ROOT
        / "experiments"
        / "paper_suite_20260723"
        / "motivation_geometry_abd_wide.py",
        ROOT / "output" / "pdf" / "fig_motivation_geometry_abd_wide.pdf",
        ROOT / "output" / "pdf" / "fig_vi_a_geometry.pdf",
        ROOT / "output" / "pdf" / "fig_vi_b_tabular.pdf",
        ROOT / "output" / "pdf" / "fig_vi_c_neural.pdf",
        STOCHASTIC_TABLE,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise VerificationError(
            "missing required release files: {}".format(", ".join(missing))
        )


def verify() -> None:
    verify_required_files()
    manifest = load_json(MANIFEST)
    verify_manifest(manifest)
    verify_environment_scope(manifest)
    verify_stochastic_table()
    print("Verification passed.")


def command_environment():
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    mpl_directory = ROOT / "output" / "build" / "reproduce" / ".mplconfig"
    mpl_directory.mkdir(parents=True, exist_ok=True)
    environment.setdefault("MPLCONFIGDIR", str(mpl_directory))
    return environment


def run(command) -> None:
    print("$ {}".format(" ".join(str(part) for part in command)), flush=True)
    subprocess.run(
        [str(part) for part in command],
        cwd=str(ROOT),
        env=command_environment(),
        check=True,
    )


def figures() -> None:
    verify()
    run(
        [
            sys.executable,
            ROOT
            / "experiments"
            / "paper_suite_20260723"
            / "motivation_geometry_abd_wide.py",
        ]
    )
    run(
        [
            sys.executable,
            ROOT
            / "experiments"
            / "paper_suite_20260723"
            / "assemble_paper_results.py",
        ]
    )
    run(
        [
            sys.executable,
            ROOT
            / "experiments"
            / "stochastic_oracle_validation_20260727"
            / "make_vi_d_table.py",
        ]
    )
    verify()
    print("Figure and table reconstruction passed.")


def paper() -> None:
    verify()
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError(
            "latexmk was not found; install a TeX distribution with IEEEtran"
        )
    output = ROOT / "output" / "build" / "reproduce"
    output.mkdir(parents=True, exist_ok=True)
    run(
        [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-outdir={}".format(output),
            ROOT / "main.tex",
        ]
    )
    compiled = output / "main.pdf"
    if not compiled.is_file():
        raise RuntimeError("latexmk completed without producing {}".format(compiled))
    print("Paper compiled to {}".format(compiled.relative_to(ROOT)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and reconstruct the released journal artifacts."
    )
    parser.add_argument(
        "mode",
        choices=("verify", "figures", "paper", "all"),
        help="artifact operation to perform",
    )
    args = parser.parse_args()

    try:
        if args.mode == "verify":
            verify()
        elif args.mode == "figures":
            figures()
        elif args.mode == "paper":
            paper()
        else:
            figures()
            paper()
    except (VerificationError, RuntimeError, subprocess.CalledProcessError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
