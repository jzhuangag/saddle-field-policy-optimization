import math
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "models" / "heldout_safeguard.py"
SPEC = spec_from_file_location("iclr_heldout_safeguard", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
choose_with_heldout_merit = MODULE.choose_with_heldout_merit


def test_accepts_strict_heldout_improvement():
    merits = {"qp": 0.7, "nog": 1.0}
    decision = choose_with_heldout_merit(
        qp_state="qp",
        field_only_state="nog",
        evaluate_merit=merits.__getitem__,
        tolerance=0.1,
    )
    assert decision.state == "qp"
    assert decision.accepted_curvature
    assert decision.merit_improvement == pytest.approx(0.3)


def test_rejects_improvement_below_tolerance():
    merits = {"qp": 0.95, "nog": 1.0}
    decision = choose_with_heldout_merit(
        qp_state="qp",
        field_only_state="nog",
        evaluate_merit=merits.__getitem__,
        tolerance=0.1,
    )
    assert decision.state == "nog"
    assert not decision.accepted_curvature
    assert decision.reason == "field_only_fallback"


def test_rejects_nonfinite_curvature_candidate():
    merits = {"qp": math.nan, "nog": 1.0}
    decision = choose_with_heldout_merit(
        qp_state="qp",
        field_only_state="nog",
        evaluate_merit=merits.__getitem__,
    )
    assert decision.state == "nog"
    assert decision.reason == "nonfinite_curvature_candidate"


def test_requires_finite_field_only_merit():
    merits = {"qp": 0.0, "nog": math.inf}
    with pytest.raises(ValueError, match="field-only"):
        choose_with_heldout_merit(
            qp_state="qp",
            field_only_state="nog",
            evaluate_merit=merits.__getitem__,
        )


def test_rejects_invalid_tolerance():
    with pytest.raises(ValueError, match="tolerance"):
        choose_with_heldout_merit(
            qp_state="qp",
            field_only_state="nog",
            evaluate_merit=lambda _: 0.0,
            tolerance=-1.0,
        )
