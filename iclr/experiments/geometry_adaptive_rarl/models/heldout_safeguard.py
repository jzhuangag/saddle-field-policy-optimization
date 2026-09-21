"""Held-out acceptance rule for curvature-enabled policy updates.

The quadratic model proposes a field-plus-curvature candidate and a matched
field-only candidate.  This module chooses between them using a trajectory
batch that was not used to estimate either candidate.  Lower merit is better.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Generic, TypeVar


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class HeldoutDecision(Generic[StateT]):
    """Result of the held-out curvature acceptance test."""

    state: StateT
    accepted_curvature: bool
    qp_merit: float
    field_only_merit: float
    merit_improvement: float
    tolerance: float
    reason: str


def choose_with_heldout_merit(
    *,
    qp_state: StateT,
    field_only_state: StateT,
    evaluate_merit: Callable[[StateT], float],
    tolerance: float = 0.0,
) -> HeldoutDecision[StateT]:
    """Accept curvature only after a strict held-out merit improvement.

    Args:
        qp_state: Candidate produced by the two-direction quadratic program.
        field_only_state: Candidate produced by the matched gamma=0 problem.
        evaluate_merit: Deterministic evaluation on one frozen held-out batch.
        tolerance: Required decrease beyond the field-only candidate.

    The same held-out batch must evaluate both candidates, and that batch must
    not have been used to fit the local quadratic model.
    """

    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")

    qp_merit = float(evaluate_merit(qp_state))
    field_only_merit = float(evaluate_merit(field_only_state))
    improvement = field_only_merit - qp_merit

    if not math.isfinite(field_only_merit):
        raise ValueError("field-only held-out merit must be finite")
    if not math.isfinite(qp_merit):
        return HeldoutDecision(
            state=field_only_state,
            accepted_curvature=False,
            qp_merit=qp_merit,
            field_only_merit=field_only_merit,
            merit_improvement=float("-inf"),
            tolerance=tolerance,
            reason="nonfinite_curvature_candidate",
        )

    accept = improvement > tolerance
    return HeldoutDecision(
        state=qp_state if accept else field_only_state,
        accepted_curvature=accept,
        qp_merit=qp_merit,
        field_only_merit=field_only_merit,
        merit_improvement=improvement,
        tolerance=tolerance,
        reason="heldout_improvement" if accept else "field_only_fallback",
    )
