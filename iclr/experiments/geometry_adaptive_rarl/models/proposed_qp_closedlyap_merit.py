from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

from models.proposed_qp_closedlyap import (
    ProposedNoGClosedLyapOptimizer,
    ProposedQPClosedLyapOptimizer,
    ProposedQPClosedMinusGLyapOptimizer,
    ProposedQPClosedSignSelectLyapOptimizer,
    ProposedQPNogSafeSignSelectLyapOptimizer,
    _ClosedLyapunovDriftBase,
)


class _NormalizedMeritMixin(_ClosedLyapunovDriftBase):
    def __init__(
        self,
        *args,
        vf_coef: float = 0.5,
        lambda_KL: float = 0.1,
        lambda_CF: float = 0.1,
        target_kl: float = 0.03,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vf_coef_local = float(vf_coef)
        self.lambda_KL = float(lambda_KL)
        self.lambda_CF = float(lambda_CF)
        self.target_kl = float(target_kl)
        self._field_scale: Optional[float] = None
        self._return_scale: Optional[float] = None
        self._policy_scale: Optional[float] = None
        self._kl_scale: Optional[float] = None
        self._clip_scale: Optional[float] = None

    def _fixed_scale(self, attr_name: str, raw_value: float, default_floor: float = 1.0) -> float:
        current = getattr(self, attr_name)
        if current is None or not math.isfinite(float(current)) or float(current) <= self.qp_eps:
            base = abs(float(raw_value))
            if not math.isfinite(base) or base <= self.qp_eps:
                base = float(default_floor)
            setattr(self, attr_name, float(base))
            return float(base)
        return float(current)

    def _actual_ppo_return_term(self, info: Dict[str, object]) -> float:
        total_loss = float(info["total_loss"])
        value_loss = float(info["value_loss"])
        if self.optimizer_scope == "full_policy":
            return total_loss
        return total_loss - self.vf_coef_local * value_loss

    def _trust_region_return_term(self, info: Dict[str, object]) -> Tuple[float, Dict[str, float]]:
        policy_clip = float(info["policy_loss"])
        approx_kl = float(info["approx_kl"])
        clip_fraction = float(info["clip_fraction"])
        kl_penalty = max(0.0, approx_kl - self.target_kl) ** 2

        policy_scale = self._fixed_scale("_policy_scale", policy_clip)
        kl_scale = self._fixed_scale("_kl_scale", kl_penalty, default_floor=max(self.target_kl * self.target_kl, self.qp_eps))
        clip_scale = self._fixed_scale("_clip_scale", clip_fraction)

        normalized_policy = policy_clip / (policy_scale + self.qp_eps)
        normalized_kl = kl_penalty / (kl_scale + self.qp_eps)
        normalized_clip = clip_fraction / (clip_scale + self.qp_eps)
        return_term = normalized_policy + self.lambda_KL * normalized_kl + self.lambda_CF * normalized_clip
        return return_term, {
            "policy_clip": policy_clip,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "kl_penalty": kl_penalty,
            "normalized_policy": normalized_policy,
            "normalized_kl": normalized_kl,
            "normalized_clip": normalized_clip,
        }


class _ActualPPOMeritBase(_NormalizedMeritMixin):
    def _evaluate_state(
        self,
        *,
        eval_closure: Callable[..., Dict[str, object]],
        theta_state,
        selected_names: Sequence[str],
        candidate_merit_evaluator=None,
    ) -> Dict[str, object]:
        del candidate_merit_evaluator
        info = eval_closure(theta_override=theta_state, backward=True, grad_scope_names=list(selected_names))
        grads_selected = {name: info["grads"][name].detach().clone() for name in selected_names}
        raw_field_term = self._field_term(grads_selected, selected_names)
        raw_return_term = self._actual_ppo_return_term(info)
        field_scale = self._fixed_scale("_field_scale", raw_field_term)
        return_scale = self._fixed_scale("_return_scale", raw_return_term)
        field_term = raw_field_term / (field_scale + self.qp_eps)
        return_term = raw_return_term / (return_scale + self.qp_eps)
        payload = dict(info)
        payload["grads_selected"] = grads_selected
        payload["field_term"] = float(field_term)
        payload["return_term"] = float(return_term)
        payload["raw_field_term"] = float(raw_field_term)
        payload["raw_return_term"] = float(raw_return_term)
        payload["return_source"] = "actual_ppo_scope_matched"
        payload["return_merit_available"] = 1
        payload["V"] = float(self.lambda_F * field_term + self.lambda_R * return_term)
        return payload


class _TrustRegionMeritBase(_NormalizedMeritMixin):
    def _evaluate_state(
        self,
        *,
        eval_closure: Callable[..., Dict[str, object]],
        theta_state,
        selected_names: Sequence[str],
        candidate_merit_evaluator=None,
    ) -> Dict[str, object]:
        del candidate_merit_evaluator
        info = eval_closure(theta_override=theta_state, backward=True, grad_scope_names=list(selected_names))
        grads_selected = {name: info["grads"][name].detach().clone() for name in selected_names}
        raw_field_term = self._field_term(grads_selected, selected_names)
        field_scale = self._fixed_scale("_field_scale", raw_field_term)
        field_term = raw_field_term / (field_scale + self.qp_eps)
        return_term, extra = self._trust_region_return_term(info)

        payload = dict(info)
        payload.update(extra)
        payload["grads_selected"] = grads_selected
        payload["field_term"] = float(field_term)
        payload["return_term"] = float(return_term)
        payload["raw_field_term"] = float(raw_field_term)
        payload["return_source"] = "trust_region_policy_kl_clip"
        payload["return_merit_available"] = 1
        payload["V"] = float(self.lambda_F * field_term + self.lambda_R * return_term)
        return payload


class ProposedNoGClosedActualPPOMeritOptimizer(_ActualPPOMeritBase, ProposedNoGClosedLyapOptimizer):
    @property
    def variant_name(self) -> str:
        return "closed_nog_actual_ppo_merit"


class ProposedNoGClosedTrustRegionMeritOptimizer(_TrustRegionMeritBase, ProposedNoGClosedLyapOptimizer):
    @property
    def variant_name(self) -> str:
        return "closed_nog_trust_region_merit"


class ProposedQPClosedActualPPOMeritOptimizer(_ActualPPOMeritBase, ProposedQPClosedLyapOptimizer):
    @property
    def variant_name(self) -> str:
        return "closed_qp_actual_ppo_merit"


class ProposedQPClosedTrustRegionMeritOptimizer(_TrustRegionMeritBase, ProposedQPClosedLyapOptimizer):
    @property
    def variant_name(self) -> str:
        return "closed_qp_trust_region_merit"


class ProposedQPClosedMinusGTrustRegionMeritOptimizer(_TrustRegionMeritBase, ProposedQPClosedMinusGLyapOptimizer):
    @property
    def variant_name(self) -> str:
        return "closed_qp_minus_g_trust_region_merit"


class ProposedQPClosedSignSelectTrustRegionMeritOptimizer(_TrustRegionMeritBase, ProposedQPClosedSignSelectLyapOptimizer):
    @property
    def variant_name(self) -> str:
        return "closed_qp_sign_select_trust_region_merit"


class ProposedQPNogSafeSignSelectTrustRegionMeritOptimizer(_TrustRegionMeritBase, ProposedQPNogSafeSignSelectLyapOptimizer):
    @property
    def variant_name(self) -> str:
        return "closed_qp_nog_safe_sign_select_trust_region_merit"
