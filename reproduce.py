#!/usr/bin/env python3
"""Verify, reconstruct, or fully rerun the released journal artifacts.

The fast modes operate on frozen CSV/JSON results; ``full`` executes the
experiment-to-paper pipeline documented in REPRODUCIBILITY.md.
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
LINEAR_RESULTS = (
    ROOT
    / "experiments"
    / "paper_suite_20260723"
    / "results"
    / "linear-geometry-20260823-234632"
)
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
STOCHASTIC_TABLE_ROWS = ROOT / "output" / "data" / "vi_d_table_rows.tex"
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


def selected_result_dir(manifest, key: str, default: Path) -> Path:
    relative = manifest.get("selected_results", {}).get(key)
    if relative is None:
        return default
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise VerificationError(
            "selected result path escapes repository root: {}".format(relative)
        ) from error
    return path


def verify_environment_scope(manifest) -> None:
    neural_results = selected_result_dir(manifest, "neural", NEURAL_RESULTS)
    tabular_results = selected_result_dir(manifest, "tabular", TABULAR_RESULTS)
    require_environment_set(
        "neural curves",
        csv_environments(neural_results / "curves.csv"),
        NEURAL_ENVIRONMENTS,
    )
    require_environment_set(
        "neural diagnostics",
        csv_environments(neural_results / "diagnostics.csv"),
        NEURAL_ENVIRONMENTS,
    )
    neural_summary = load_json(neural_results / "summary.json")
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
        csv_environments(tabular_results / "curves.csv"),
        TABULAR_ENVIRONMENTS,
    )
    tabular_summary = load_json(tabular_results / "summary.json")
    require_environment_set(
        "tabular summary protocol",
        tabular_summary.get("protocol", {}).get("environments", []),
        TABULAR_ENVIRONMENTS,
    )

    protocol_path = STOCHASTIC_PROTOCOL
    if STOCHASTIC_TABLE.is_file():
        table = load_json(STOCHASTIC_TABLE)
        source_path = table.get("source", {}).get("path")
        if source_path:
            candidate = (ROOT / source_path).resolve().parent / "protocol.json"
            if candidate.is_file():
                protocol_path = candidate
    protocol = load_json(protocol_path)
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
    latex_rows = table.get("latex_three_decimal_rows", {})
    labels = {
        "hard_br_return": r"Worst-case return $\uparrow$",
        "hard_exploitability": r"Exploitability $\downarrow$",
        "field_norm": r"Population $\norm{\bF}\downarrow$",
    }
    rendered = []
    for metric in ("hard_br_return", "hard_exploitability", "field_norm"):
        row = latex_rows.get(metric)
        if not isinstance(row, dict):
            raise VerificationError(
                "finite-trajectory table has no LaTeX row for {}".format(metric)
            )
        qpg = str(row.get("QP+G_latex", "")).replace(r"\pm ", r"\pm")
        nog = str(row.get("noG_latex", "")).replace(r"\pm ", r"\pm")
        paired = str(row.get("paired_latex", "")).replace("$ $", r"\,")
        if not qpg or not nog or not paired:
            raise VerificationError(
                "finite-trajectory LaTeX row is incomplete for {}".format(metric)
            )
        rendered.append("{} & {} & {} & {}".format(labels[metric], qpg, nog, paired))
    expected_tex = (
        r"\providecommand{\VIDTableRows}{%"
        + "\n"
        + "\\\\\n".join(rendered)
        + "\\\\%\n}\n"
    ).encode("utf-8")
    if not STOCHASTIC_TABLE_ROWS.is_file():
        raise VerificationError(
            "missing finite-trajectory LaTeX rows: {}".format(
                STOCHASTIC_TABLE_ROWS.relative_to(ROOT)
            )
        )
    actual_tex = canonical_text_bytes(STOCHASTIC_TABLE_ROWS)
    if actual_tex != expected_tex:
        raise VerificationError(
            "finite-trajectory LaTeX rows do not match the audited table JSON"
        )
    print("[ok] finite-trajectory table source, assertions, and LaTeX rows")


def require_files(paths, label: str) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise VerificationError(
            "missing required {}: {}".format(label, ", ".join(missing))
        )


def verify_required_inputs() -> None:
    require_files(
        [
        ROOT / "main.tex",
        ROOT / "refs.bib",
        MANIFEST,
        ROOT / "experiments" / "paper_suite_20260723" / "journal_games.py",
        ROOT
        / "experiments"
        / "paper_suite_20260723"
        / "motivation_geometry_abd_wide.py",
        ROOT
        / "experiments"
        / "paper_suite_20260723"
        / "assemble_paper_results.py",
        ROOT
        / "experiments"
        / "stochastic_oracle_validation_20260727"
        / "make_vi_d_table.py",
        STOCHASTIC_PROTOCOL,
        ],
        "reconstruction inputs",
    )


def verify_generated_files(include_release_pdf: bool) -> None:
    required = [
        ROOT / "output" / "pdf" / "fig_motivation_geometry_abd_wide.pdf",
        ROOT / "output" / "pdf" / "fig_vi_a_geometry.pdf",
        ROOT / "output" / "pdf" / "fig_vi_b_tabular.pdf",
        ROOT / "output" / "pdf" / "fig_vi_c_neural.pdf",
        STOCHASTIC_TABLE,
        STOCHASTIC_TABLE_ROWS,
    ]
    if include_release_pdf:
        required.append(ROOT / "RARL_final.pdf")
    require_files(required, "generated release files")


def verify_inputs() -> None:
    verify_required_inputs()
    manifest = load_json(MANIFEST)
    verify_manifest(manifest)
    verify_environment_scope(manifest)


def verify() -> None:
    verify_inputs()
    verify_generated_files(include_release_pdf=True)
    verify_stochastic_table()
    print("Verification passed.")


def command_environment():
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment["PYTHONUTF8"] = "1"
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


def run_for_path(command, marker: str) -> Path:
    """Run a command, stream its output, and return its declared artifact path."""

    print("$ {}".format(" ".join(str(part) for part in command)), flush=True)
    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=str(ROOT),
        env=command_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    declared = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        stripped = line.strip()
        if stripped.startswith(marker):
            declared = stripped.split("=", 1)[1]
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    if not declared:
        raise RuntimeError(
            "command completed without declaring {}".format(marker.rstrip("="))
        )
    path = Path(declared)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(
            "declared artifact lies outside the repository: {}".format(path)
        ) from error
    if not path.is_dir():
        raise RuntimeError("declared artifact directory does not exist: {}".format(path))
    return path


def figures() -> None:
    # Generated PDFs and table files are deliberately not prerequisites here:
    # this path must work after they have been removed from a clean clone.
    verify_inputs()
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
    verify_inputs()
    verify_generated_files(include_release_pdf=False)
    verify_stochastic_table()
    print("Figure and table reconstruction passed.")


def paper() -> None:
    verify_inputs()
    verify_generated_files(include_release_pdf=False)
    verify_stochastic_table()
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


def full_rerun() -> None:
    """Rerun every reported experiment and rebuild the manuscript artifacts."""

    paper_suite = ROOT / "experiments" / "paper_suite_20260723"
    stochastic_suite = (
        ROOT / "experiments" / "stochastic_oracle_validation_20260727"
    )

    linear = run_for_path(
        [sys.executable, paper_suite / "linear_geometry.py"], "RESULT_DIR="
    )
    run_for_path(
        [sys.executable, paper_suite / "theory_sentinels.py"], "RESULT_DIR="
    )
    tabular = run_for_path(
        [
            sys.executable,
            paper_suite / "markov_game_suite.py",
            "--mode",
            "tabular",
            "--environments",
            "RPS",
            "CyclicControl",
            "FrequencyHopping",
            "--seeds",
            "10",
            "--seed-start",
            "200",
            "--steps",
            "100",
        ],
        "RESULT_DIR=",
    )
    controller = run_for_path(
        [
            sys.executable,
            paper_suite / "markov_game_suite.py",
            "--mode",
            "neural",
            "--environments",
            "CyclicControl",
            "FrequencyHopping",
            "RoutingInterdiction",
            "SecurityPatrol",
            "--seeds",
            "10",
            "--seed-start",
            "40",
            "--steps",
            "60",
            "--fixed-lr",
            "0.03",
            "--methods",
            "QP+G",
            "noG",
        ],
        "RESULT_DIR=",
    )
    tuning = run_for_path(
        [sys.executable, paper_suite / "tune_fixed_baselines.py"], "RESULT_DIR="
    )
    neural = run_for_path(
        [
            sys.executable,
            paper_suite / "merge_neural_journal.py",
            "--controller-dir",
            controller,
            "--tuning-summary",
            tuning / "summary.json",
        ],
        "RESULT_DIR=",
    )
    run_for_path(
        [
            sys.executable,
            paper_suite / "audit_saved_bridge.py",
            "--tabular-dir",
            tabular,
            "--neural-dir",
            neural,
        ],
        "RESULT_DIR=",
    )

    run_for_path(
        [sys.executable, stochastic_suite / "sentinels_dice.py"], "OUTPUT="
    )
    stochastic = run_for_path(
        [
            sys.executable,
            stochastic_suite / "stochastic_dice_policy.py",
            "--phase",
            "formal",
        ],
        "OUTPUT=",
    )
    run([sys.executable, stochastic_suite / "audit_formal.py", stochastic])

    run([sys.executable, paper_suite / "motivation_geometry_abd_wide.py"])
    run(
        [
            sys.executable,
            paper_suite / "assemble_paper_results.py",
            "--linear-dir",
            linear,
            "--tabular-dir",
            tabular,
            "--neural-dir",
            neural,
        ]
    )
    run(
        [
            sys.executable,
            stochastic_suite / "make_vi_d_table.py",
            "--source-dir",
            stochastic,
        ]
    )
    verify_inputs()
    verify_generated_files(include_release_pdf=False)
    verify_stochastic_table()
    paper()
    print("Full experiment-to-paper rerun passed.")
    print("LINEAR_RESULT={}".format(linear.relative_to(ROOT)))
    print("TABULAR_RESULT={}".format(tabular.relative_to(ROOT)))
    print("NEURAL_RESULT={}".format(neural.relative_to(ROOT)))
    print("FINITE_TRAJECTORY_RESULT={}".format(stochastic.relative_to(ROOT)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and reconstruct the released journal artifacts."
    )
    parser.add_argument(
        "mode",
        choices=("verify", "figures", "paper", "all", "full"),
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
        elif args.mode == "all":
            figures()
            paper()
        else:
            full_rerun()
    except (VerificationError, RuntimeError, subprocess.CalledProcessError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
