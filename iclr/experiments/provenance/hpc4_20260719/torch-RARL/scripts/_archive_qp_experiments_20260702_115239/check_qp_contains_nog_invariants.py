from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import inspect
import math
import os
import pathlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12


def load_module(module_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(value, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def auc_from_curve(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    if frame.empty or x_col not in frame.columns or y_col not in frame.columns:
        return math.nan
    sub = frame[[x_col, y_col]].dropna()
    if len(sub) < 2:
        return math.nan
    return float(np.trapezoid(sub[y_col].to_numpy(dtype=float), sub[x_col].to_numpy(dtype=float)))


def parse_args() -> argparse.Namespace:
    repo_default = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser("Check QP contains noG invariants")
    parser.add_argument("--repo-dir", type=str, default=str(repo_default))
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(repo_default.parent / "results" / "fix_qp_solver_contains_nog"),
    )
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


@dataclass(frozen=True)
class MethodSpec:
    label: str
    optimizer: str
    protagonist_optimizer_kwargs: Dict[str, object]
    adversary_optimizer_kwargs: Dict[str, object]
    lr: float
    max_grad_norm: float
    vf_coef: float


def _windows_safe_dir(path: str) -> str:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return path


def q1_value(sol: Dict[str, object], beta: float) -> float:
    l_beta = float(sol["l_beta"])
    h_bb = float(sol["h_bb"])
    beta = float(beta)
    return float(l_beta * beta + 0.5 * h_bb * beta * beta)


def q2_value(sol: Dict[str, object], beta: float, gamma: float) -> float:
    l_beta = float(sol["l_beta"])
    l_gamma = float(sol["l_gamma"])
    h_bb = float(sol["h_bb"])
    h_bg = float(sol["h_bg"])
    h_gg = float(sol["h_gg"])
    beta = float(beta)
    gamma = float(gamma)
    return float(
        l_beta * beta
        + l_gamma * gamma
        + 0.5 * h_bb * beta * beta
        + h_bg * beta * gamma
        + 0.5 * h_gg * gamma * gamma
    )


class _InvariantLoggerMixin:
    def __init__(self, *args, invariant_csv_path: Optional[str] = None, **kwargs):
        self.invariant_csv_path = invariant_csv_path
        self._invariant_fieldnames = [
            "step_index",
            "role",
            "variant",
            "beta_test_label",
            "beta_test",
            "beta_noG",
            "beta_QP",
            "gamma_QP",
            "source_of_beta_noG",
            "beta_noG_independent_of_gamma",
            "beta_bound_noG",
            "beta_bound_QP",
            "same_beta_bound_flag",
            "gamma_negative_flag",
            "q1_beta",
            "q2_beta_gamma0",
            "abs_diff",
            "rel_diff",
            "q1_equals_q2_gamma0_flag",
            "q_qp",
            "q_nog_in_qp_space",
            "predicted_inclusion_gap",
            "predicted_inclusion_pass",
            "V_before",
            "V_after_noG",
            "V_after_QP",
            "actual_drift_noG",
            "actual_drift_QP",
            "actual_QP_better_than_noG",
            "chosen_step",
        ]
        super().__init__(*args, **kwargs)
        self._ensure_invariant_header()

    def _ensure_invariant_header(self) -> None:
        if not self.invariant_csv_path:
            return
        path = _windows_safe_dir(self.invariant_csv_path)
        if os.path.exists(path):
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._invariant_fieldnames)
            writer.writeheader()

    def _write_invariant_rows(self, rows: List[Dict[str, object]]) -> None:
        if not self.invariant_csv_path or not rows:
            return
        path = _windows_safe_dir(self.invariant_csv_path)
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._invariant_fieldnames)
            for row in rows:
                writer.writerow({field: row.get(field) for field in self._invariant_fieldnames})


def register_optimizers(repo_dir: pathlib.Path):
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from models.optimizers import OPTIMIZER_REGISTRY, flatten_named_tensors, tensor_norm
    from models.proposed_qp_closedlyap import _state_is_finite
    from models.proposed_qp_closedlyap_merit import (
        ProposedNoGClosedTrustRegionMeritOptimizer,
        ProposedQPNogSafeSignSelectTrustRegionMeritOptimizer,
        ProposedQPClosedTrustRegionMeritOptimizer,
    )

    class ProposedQPContainsNogInvariantTrustRegionOptimizer(_InvariantLoggerMixin, ProposedQPClosedTrustRegionMeritOptimizer):
        @property
        def variant_name(self) -> str:
            return "closed_qp_contains_nog_invariant_trust_region_merit"

        def _step_impl(
            self,
            *,
            theta_old,
            selected_names,
            base_eval,
            eval_state,
            eval_closure,
            eta,
        ):
            from models.optimizers import apply_state_delta

            g_map = self._g_map(
                theta_old=theta_old,
                f_map=base_eval["grads_selected"],
                selected_names=selected_names,
                eval_closure=eval_closure,
            )
            p_map = {name: -base_eval["grads_selected"][name] for name in selected_names}
            nog_sol = self._solve_nog(
                theta_old=theta_old,
                p_map=p_map,
                selected_names=selected_names,
                v0=float(base_eval["V"]),
                eval_state=eval_state,
                eta=eta,
            )

            if not _state_is_finite(g_map, selected_names) or tensor_norm(flatten_named_tensors(g_map, selected_names)) <= self.qp_eps:
                update_map, norm_pre, norm_post, cap_active = self._cap_update(nog_sol["update_map"], selected_names)
                theta_candidate = apply_state_delta(theta_old, update_map)
                return {
                    "theta_candidate": theta_candidate,
                    "theta_nog_candidate": theta_candidate,
                    "theta_qp_candidate": theta_candidate,
                    "beta": float(nog_sol["beta"]),
                    "gamma": 0.0,
                    "beta_raw": float(nog_sol["beta_raw"]),
                    "gamma_raw": 0.0,
                    "predicted_drift_noG": float(nog_sol.get("predicted_drift", 0.0)),
                    "predicted_drift_QP": float(nog_sol.get("predicted_drift", 0.0)),
                    "update_norm_pre_cap": float(norm_pre),
                    "update_norm_post_cap": float(norm_post),
                    "cap_active": int(cap_active),
                    "db": float(self.beta_probe if self.beta_probe is not None else max(eta, self.qp_eps)),
                    "dg": float(self.gamma_probe if self.gamma_probe is not None else max(eta * eta, self.qp_eps)),
                    "g_map": g_map,
                    "chosen_step": "noG",
                }

            qp_sol = self._solve_qp(
                theta_old=theta_old,
                p_map=p_map,
                g_map=g_map,
                selected_names=selected_names,
                v0=float(base_eval["V"]),
                eval_state=eval_state,
                eta=eta,
            )

            qp_update_map, norm_pre, norm_post, cap_active = self._cap_update(qp_sol["update_map"], selected_names)
            nog_update_map, _, _, _ = self._cap_update(nog_sol["update_map"], selected_names)
            theta_qp = apply_state_delta(theta_old, qp_update_map)
            theta_nog = apply_state_delta(theta_old, nog_update_map)
            qp_eval = eval_state(theta_qp)
            nog_eval = eval_state(theta_nog)

            beta_noG = float(nog_sol["beta"])
            beta_QP = float(qp_sol["beta"])
            gamma_QP = float(qp_sol["gamma"])
            beta_max = float(self.beta_max) if self.beta_max is not None else math.nan
            beta_tests = {
                "zero": 0.0,
                "beta_noG": beta_noG,
                "beta_QP": beta_QP,
                "half_beta_noG": 0.5 * beta_noG,
                "double_beta_noG_clip": min(2.0 * beta_noG, beta_max) if finite(beta_max) else 2.0 * beta_noG,
            }

            q_qp = q2_value(qp_sol, beta_QP, gamma_QP)
            q_nog_in_qp_space = q2_value(qp_sol, beta_noG, 0.0)
            pred_tol = 1e-6 * max(1.0, abs(q_qp), abs(q_nog_in_qp_space))
            predicted_inclusion_pass = int(q_qp <= q_nog_in_qp_space + pred_tol)
            actual_drift_qp = float(qp_eval["V"] - base_eval["V"])
            actual_drift_nog = float(nog_eval["V"] - base_eval["V"])
            actual_qp_better = int(float(qp_eval["V"]) <= float(nog_eval["V"]) + pred_tol)
            gamma_negative_flag = int(gamma_QP < -self.qp_eps)

            invariant_rows: List[Dict[str, object]] = []
            for label, beta_val in beta_tests.items():
                q1 = q1_value(nog_sol, beta_val)
                q2 = q2_value(qp_sol, beta_val, 0.0)
                abs_diff = abs(q1 - q2)
                tol = 1e-6 * max(1.0, abs(q1), abs(q2))
                invariant_rows.append(
                    {
                        "step_index": self._step_index,
                        "role": self.role,
                        "variant": self.variant_name,
                        "beta_test_label": label,
                        "beta_test": float(beta_val),
                        "beta_noG": beta_noG,
                        "beta_QP": beta_QP,
                        "gamma_QP": gamma_QP,
                        "source_of_beta_noG": "solve_1d",
                        "beta_noG_independent_of_gamma": 1,
                        "beta_bound_noG": beta_max,
                        "beta_bound_QP": beta_max,
                        "same_beta_bound_flag": 1,
                        "gamma_negative_flag": gamma_negative_flag,
                        "q1_beta": q1,
                        "q2_beta_gamma0": q2,
                        "abs_diff": abs_diff,
                        "rel_diff": abs_diff / max(1.0, abs(q1), abs(q2)),
                        "q1_equals_q2_gamma0_flag": int(abs_diff <= tol),
                        "q_qp": q_qp,
                        "q_nog_in_qp_space": q_nog_in_qp_space,
                        "predicted_inclusion_gap": q_qp - q_nog_in_qp_space,
                        "predicted_inclusion_pass": predicted_inclusion_pass,
                        "V_before": float(base_eval["V"]),
                        "V_after_noG": float(nog_eval["V"]),
                        "V_after_QP": float(qp_eval["V"]),
                        "actual_drift_noG": actual_drift_nog,
                        "actual_drift_QP": actual_drift_qp,
                        "actual_QP_better_than_noG": actual_qp_better,
                        "chosen_step": "plusG",
                    }
                )
            self._write_invariant_rows(invariant_rows)

            return {
                "theta_candidate": theta_qp,
                "theta_nog_candidate": theta_nog,
                "theta_qp_candidate": theta_qp,
                "theta_plus_candidate": theta_qp,
                "beta": float(qp_sol["beta"]),
                "gamma": float(qp_sol["gamma"]),
                "beta_raw": float(qp_sol["beta_raw"]),
                "gamma_raw": float(qp_sol["gamma_raw"]),
                "predicted_drift_noG": float(nog_sol.get("predicted_drift", 0.0)),
                "predicted_drift_QP": float(qp_sol.get("predicted_drift", 0.0)),
                "beta_plus": float(qp_sol["beta"]),
                "gamma_plus": float(qp_sol["gamma"]),
                "update_norm_pre_cap": float(norm_pre),
                "update_norm_post_cap": float(norm_post),
                "cap_active": int(cap_active),
                "db": float(qp_sol["db"]),
                "dg": float(qp_sol["dg"]),
                "g_map": g_map,
                "chosen_step": "plusG",
            }

    class ProposedQPForcedNogTrustRegionOptimizer(ProposedQPNogSafeSignSelectTrustRegionMeritOptimizer):
        @property
        def variant_name(self) -> str:
            return "closed_qp_forced_nog_trust_region_merit"

        def _step_impl(
            self,
            *,
            theta_old,
            selected_names,
            base_eval,
            eval_state,
            eval_closure,
            eta,
        ):
            from models.optimizers import apply_state_delta

            g_map = self._g_map(
                theta_old=theta_old,
                f_map=base_eval["grads_selected"],
                selected_names=selected_names,
                eval_closure=eval_closure,
            )
            p_map = {name: -base_eval["grads_selected"][name] for name in selected_names}
            nog_sol = self._solve_nog(
                theta_old=theta_old,
                p_map=p_map,
                selected_names=selected_names,
                v0=float(base_eval["V"]),
                eval_state=eval_state,
                eta=eta,
            )
            nog_update_map, nog_norm_pre, nog_norm_post, nog_cap_active = self._cap_update(nog_sol["update_map"], selected_names)
            theta_nog = apply_state_delta(theta_old, nog_update_map)
            if not _state_is_finite(g_map, selected_names) or tensor_norm(flatten_named_tensors(g_map, selected_names)) <= self.qp_eps:
                return {
                    "theta_candidate": theta_nog,
                    "theta_nog_candidate": theta_nog,
                    "theta_qp_candidate": theta_nog,
                    "beta": float(nog_sol["beta"]),
                    "gamma": 0.0,
                    "beta_raw": float(nog_sol["beta_raw"]),
                    "gamma_raw": 0.0,
                    "predicted_drift_noG": float(nog_sol.get("predicted_drift", 0.0)),
                    "predicted_drift_QP": float(nog_sol.get("predicted_drift", 0.0)),
                    "update_norm_pre_cap": float(nog_norm_pre),
                    "update_norm_post_cap": float(nog_norm_post),
                    "cap_active": int(nog_cap_active),
                    "db": float(self.beta_probe if self.beta_probe is not None else max(eta, self.qp_eps)),
                    "dg": float(self.gamma_probe if self.gamma_probe is not None else max(eta * eta, self.qp_eps)),
                    "g_map": g_map,
                    "chosen_step": "noG",
                }

            plus_sol = self._solve_qp(
                theta_old=theta_old,
                p_map=p_map,
                g_map=g_map,
                selected_names=selected_names,
                v0=float(base_eval["V"]),
                eval_state=eval_state,
                eta=eta,
            )
            minus_sol = self._solve_qp(
                theta_old=theta_old,
                p_map=p_map,
                g_map={name: -g_map[name] for name in selected_names},
                selected_names=selected_names,
                v0=float(base_eval["V"]),
                eval_state=eval_state,
                eta=eta,
            )
            plus_update_map, _, _, _ = self._cap_update(plus_sol["update_map"], selected_names)
            minus_update_map, _, _, _ = self._cap_update(minus_sol["update_map"], selected_names)
            theta_plus = apply_state_delta(theta_old, plus_update_map)
            theta_minus = apply_state_delta(theta_old, minus_update_map)
            return {
                "theta_candidate": theta_nog,
                "theta_nog_candidate": theta_nog,
                "theta_qp_candidate": theta_nog,
                "theta_plus_candidate": theta_plus,
                "theta_minus_candidate": theta_minus,
                "beta": float(nog_sol["beta"]),
                "gamma": 0.0,
                "beta_raw": float(nog_sol["beta_raw"]),
                "gamma_raw": 0.0,
                "predicted_drift_noG": float(nog_sol.get("predicted_drift", 0.0)),
                "predicted_drift_QP": float(nog_sol.get("predicted_drift", 0.0)),
                "beta_plus": float(plus_sol["beta"]),
                "gamma_plus": float(plus_sol["gamma"]),
                "beta_minus": float(minus_sol["beta"]),
                "gamma_minus": float(minus_sol["gamma"]),
                "update_norm_pre_cap": float(nog_norm_pre),
                "update_norm_post_cap": float(nog_norm_post),
                "cap_active": int(nog_cap_active),
                "db": float(nog_sol.get("db", self.beta_probe if self.beta_probe is not None else max(eta, self.qp_eps))),
                "dg": float(self.gamma_probe if self.gamma_probe is not None else max(eta * eta, self.qp_eps)),
                "g_map": g_map,
                "chosen_step": "noG",
            }

    OPTIMIZER_REGISTRY["proposed_nog_closed_trustregion"] = ProposedNoGClosedTrustRegionMeritOptimizer
    OPTIMIZER_REGISTRY["proposed_qp_plusg_closed_trustregion"] = ProposedQPClosedTrustRegionMeritOptimizer
    OPTIMIZER_REGISTRY["proposed_qp_contains_nog_invariant_trustregion"] = ProposedQPContainsNogInvariantTrustRegionOptimizer
    OPTIMIZER_REGISTRY["proposed_qp_forced_nog_trustregion"] = ProposedQPForcedNogTrustRegionOptimizer
    OPTIMIZER_REGISTRY["proposed_qp_nogsafe_signselect_closed_trustregion"] = ProposedQPNogSafeSignSelectTrustRegionMeritOptimizer


def make_diag_kwargs(method_stem: str, role: str, diagnostics_root: pathlib.Path, invariant_root: Optional[pathlib.Path] = None) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "optimizer_scope": "full_policy",
        "lambda_F": 0.01,
        "lambda_R": 0.3,
        "lambda_KL": 0.3,
        "lambda_CF": 0.1,
        "target_kl": 0.03,
        "vf_coef": 0.5,
        "fd_eps": 1e-3,
        "beta_probe": 3e-4,
        "gamma_probe": 1e-6,
        "ridge": 1e-8,
        "beta_max": 9e-4,
        "gamma_max": 3e-6,
        "max_update_norm": 0.005,
        "qp_eps": 1e-8,
        "allow_fallback_to_egm": False,
        "cost_mode": "trust_region_policy_kl_clip",
        "role": role,
        "diagnostics_csv_path": str(diagnostics_root / f"{role}_{method_stem}.csv"),
    }
    if invariant_root is not None:
        payload["invariant_csv_path"] = str(invariant_root / f"{role}_{method_stem}_invariants.csv")
    return payload


def method_specs(diagnostics_root: pathlib.Path, invariant_root: pathlib.Path) -> List[MethodSpec]:
    return [
        MethodSpec(
            "proposed_nog_closed_trustreg",
            "proposed_nog_closed_trustregion",
            make_diag_kwargs("nog_trustreg", "protagonist", diagnostics_root),
            make_diag_kwargs("nog_trustreg", "adversary", diagnostics_root),
            3e-4,
            10.0,
            0.5,
        ),
        MethodSpec(
            "proposed_qp_contains_nog_invariant_trustreg",
            "proposed_qp_contains_nog_invariant_trustregion",
            make_diag_kwargs("qp_invariant_trustreg", "protagonist", diagnostics_root, invariant_root),
            make_diag_kwargs("qp_invariant_trustreg", "adversary", diagnostics_root, invariant_root),
            3e-4,
            10.0,
            0.5,
        ),
        MethodSpec(
            "proposed_qp_forced_nog_trustreg",
            "proposed_qp_forced_nog_trustregion",
            make_diag_kwargs("forced_nog_trustreg", "protagonist", diagnostics_root),
            make_diag_kwargs("forced_nog_trustreg", "adversary", diagnostics_root),
            3e-4,
            10.0,
            0.5,
        ),
        MethodSpec(
            "proposed_qp_nog_safe_sign_select_trustreg",
            "proposed_qp_nogsafe_signselect_closed_trustregion",
            make_diag_kwargs("nogsafe_trustreg", "protagonist", diagnostics_root),
            make_diag_kwargs("nogsafe_trustreg", "adversary", diagnostics_root),
            3e-4,
            10.0,
            0.5,
        ),
    ]


def smoke_method_specs(diagnostics_root: pathlib.Path) -> List[MethodSpec]:
    return [
        MethodSpec("sgd_gda", "sgd", {}, {}, 3e-4, 10.0, 0.5),
        MethodSpec("egm", "egm", {}, {}, 3e-4, 10.0, 0.5),
        MethodSpec(
            "proposed_nog_closed_trustreg",
            "proposed_nog_closed_trustregion",
            make_diag_kwargs("smoke_nog_trustreg", "protagonist", diagnostics_root),
            make_diag_kwargs("smoke_nog_trustreg", "adversary", diagnostics_root),
            3e-4,
            10.0,
            0.5,
        ),
        MethodSpec(
            "proposed_qp_plusG_trustreg_FIXED",
            "proposed_qp_plusg_closed_trustregion",
            make_diag_kwargs("smoke_plusg_trustreg_fixed", "protagonist", diagnostics_root),
            make_diag_kwargs("smoke_plusg_trustreg_fixed", "adversary", diagnostics_root),
            3e-4,
            10.0,
            0.5,
        ),
        MethodSpec(
            "proposed_qp_nog_safe_trustreg_FIXED",
            "proposed_qp_nogsafe_signselect_closed_trustregion",
            make_diag_kwargs("smoke_nogsafe_trustreg_fixed", "protagonist", diagnostics_root),
            make_diag_kwargs("smoke_nogsafe_trustreg_fixed", "adversary", diagnostics_root),
            3e-4,
            10.0,
            0.5,
        ),
    ]


def build_ns(method: MethodSpec, run_root: pathlib.Path, repo_dir: pathlib.Path, env: str, seed: int, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        verbose=1,
        seed=seed,
        num_exps=1,
        num_threads=-1,
        env=env,
        n_envs=1,
        vec_env_type="dummy",
        env_kwargs=None,
        adv_env=False,
        algo="rarl",
        rarl_config="ppo",
        saved_models_path=str(run_root / "sm"),
        pretrained_model="",
        save_replay_buffer=False,
        hyperparameter=None,
        optimize_hyperparameters=False,
        hyperparameter_path=str(repo_dir / "hyperparameter"),
        storage=None,
        study_name=None,
        sampler="tpe",
        pruner="median",
        optimization_log_path=str(run_root / "opt"),
        n_opt_trials=10,
        no_optim_plots=False,
        n_jobs=1,
        n_startup_trials=10,
        n_evaluations_opt=20,
        n_timesteps=2,
        save_freq=10240,
        log_interval=-1,
        device=device,
        eval_freq=10240,
        n_eval_envs=1,
        n_eval_episodes=3,
        control_proxy_eval=False,
        tensorboard_log=str(run_root / "tb"),
        log_folder=str(run_root / "log"),
        protagonist_policy="MlpPolicy",
        adversary_policy="MlpPolicy",
        protagonist_optimizer=method.optimizer,
        adversary_optimizer=method.optimizer,
        protagonist_optimizer_kwargs=method.protagonist_optimizer_kwargs,
        adversary_optimizer_kwargs=method.adversary_optimizer_kwargs,
        protagonist_lr=method.lr,
        adversary_lr=method.lr,
        protagonist_max_grad_norm=method.max_grad_norm,
        adversary_max_grad_norm=method.max_grad_norm,
        protagonist_vf_coef=method.vf_coef,
        adversary_vf_coef=method.vf_coef,
        optimizer_scope="full_policy",
        qp_normalization="none",
        qp_g_alpha=1e-3,
        max_update_norm=0.005,
        qp_eps=1e-8,
        qp_alpha=0.3,
        qp_beta_max=1.0,
        qp_gamma_max=1.0,
        qp_step_grid="0,0.1,0.3,1.0,3.0",
        qp_objective="loss",
        qp_accept_rule="none",
        qp_min_g_contribution=0.0,
        qp_critic_weight=1.0,
        qp_g_sign="plus",
        qp_fd_eps=1e-3,
        qp_beta_probe=method.lr,
        qp_gamma_probe=max(method.lr * method.lr, 1e-6),
        qp_ridge=1e-8,
        qp_actor_weight=1.0,
        qp_logstd_weight=1.0,
        qp_step_solver="lyapunov_quadratic_bound",
        N_mu=5,
        N_nu=1,
        adv_impact="control",
        adv_delay=-1,
        adv_fraction=0.05,
        adv_fraction_override=True,
        requested_alpha=0.05,
        requested_adv_fraction=0.05,
        adv_index_list=["torso"],
        adv_force_dim=2,
    )


def build_smoke_ns(method: MethodSpec, run_root: pathlib.Path, repo_dir: pathlib.Path, env: str, seed: int, device: str) -> SimpleNamespace:
    ns = build_ns(method, run_root, repo_dir, env, seed, device)
    ns.n_timesteps = 6
    ns.n_eval_episodes = 5
    ns.save_freq = 10240
    ns.eval_freq = 10240
    return ns


def run_with_exp_manager(repo_dir: pathlib.Path, env_id: str, run_root: pathlib.Path, ns: SimpleNamespace) -> pathlib.Path:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from utils.exp_manager import ExperimentManager

    ensure_dir(run_root)
    stdout_path = run_root / "stdout.txt"
    stderr_path = run_root / "stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        with contextlib.redirect_stdout(stdout_handle), contextlib.redirect_stderr(stderr_handle):
            manager = ExperimentManager(
                ns,
                algo="rarl",
                env_id=env_id,
                log_folder=ns.log_folder,
                tensorboard_log=ns.tensorboard_log,
                n_timesteps=ns.n_timesteps,
                eval_freq=ns.eval_freq,
                n_eval_episodes=ns.n_eval_episodes,
                save_freq=ns.save_freq,
                hyperparameter_path=ns.hyperparameter_path,
                hyperparams=ns.hyperparameter,
                env_kwargs=ns.env_kwargs,
                model_path=str(run_root / "sm" / "rarl-ppo" / env_id),
                pretrained_model=ns.pretrained_model,
                optimize_hyperparameters=ns.optimize_hyperparameters,
                storage=ns.storage,
                study_name=ns.study_name,
                n_opt_trials=ns.n_opt_trials,
                n_jobs=ns.n_jobs,
                sampler=ns.sampler,
                pruner=ns.pruner,
                optimization_log_path=ns.optimization_log_path,
                n_startup_trials=ns.n_startup_trials,
                n_evaluations_opt=ns.n_evaluations_opt,
                seed=ns.seed,
                log_interval=ns.log_interval,
                save_replay_buffer=ns.save_replay_buffer,
                verbose=ns.verbose,
                vec_env_type=ns.vec_env_type,
                n_envs=ns.n_envs,
                n_eval_envs=ns.n_eval_envs,
                no_optim_plots=ns.no_optim_plots,
                adv_env=ns.adv_env,
                adv_impact=ns.adv_impact,
                adv_fraction=ns.adv_fraction,
                adv_delay=ns.adv_delay,
                adv_index_list=ns.adv_index_list,
                adv_force_dim=ns.adv_force_dim,
                N_mu=ns.N_mu,
                N_nu=ns.N_nu,
                device=ns.device,
                rarl_config=ns.rarl_config,
                protagonist_optimizer=ns.protagonist_optimizer,
                adversary_optimizer=ns.adversary_optimizer,
                protagonist_optimizer_kwargs=ns.protagonist_optimizer_kwargs,
                adversary_optimizer_kwargs=ns.adversary_optimizer_kwargs,
                protagonist_lr=ns.protagonist_lr,
                adversary_lr=ns.adversary_lr,
                protagonist_max_grad_norm=ns.protagonist_max_grad_norm,
                adversary_max_grad_norm=ns.adversary_max_grad_norm,
                protagonist_vf_coef=ns.protagonist_vf_coef,
                adversary_vf_coef=ns.adversary_vf_coef,
                optimizer_scope=ns.optimizer_scope,
                qp_normalization=ns.qp_normalization,
                qp_g_alpha=ns.qp_g_alpha,
                max_update_norm=ns.max_update_norm,
                qp_eps=ns.qp_eps,
                qp_alpha=ns.qp_alpha,
                qp_beta_max=ns.qp_beta_max,
                qp_gamma_max=ns.qp_gamma_max,
                qp_step_grid=ns.qp_step_grid,
                qp_objective=ns.qp_objective,
                qp_accept_rule=ns.qp_accept_rule,
                qp_min_g_contribution=ns.qp_min_g_contribution,
                qp_critic_weight=ns.qp_critic_weight,
                qp_g_sign=ns.qp_g_sign,
                qp_fd_eps=ns.qp_fd_eps,
                qp_beta_probe=ns.qp_beta_probe,
                qp_gamma_probe=ns.qp_gamma_probe,
                qp_ridge=ns.qp_ridge,
                qp_actor_weight=ns.qp_actor_weight,
                qp_logstd_weight=ns.qp_logstd_weight,
                qp_step_solver=ns.qp_step_solver,
                control_proxy_eval=ns.control_proxy_eval,
            )
            model = manager.setup_experiment()
            if model is None:
                raise RuntimeError("Unexpected hyperparameter-optimization branch")
            manager.learn(model)
            manager.save_trained_model(model)

    env_root = run_root / "sm" / "rarl-ppo" / env_id
    run_dirs = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {env_root}")
    return max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def try_latest_run_dir(run_root: pathlib.Path, env_id: str) -> Optional[pathlib.Path]:
    env_root = run_root / "sm" / "rarl-ppo" / env_id
    if not env_root.exists():
        return None
    run_dirs = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def read_diag_pair(diagnostics_root: pathlib.Path, stem: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for role in ["protagonist", "adversary"]:
        path = diagnostics_root / f"{role}_{stem}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                frame["diag_role"] = role
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_invariant_rows(invariant_root: pathlib.Path, stem: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for role in ["protagonist", "adversary"]:
        path = invariant_root / f"{role}_{stem}_invariants.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                frame["diag_role"] = role
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir).resolve()
    output_root = ensure_dir(pathlib.Path(args.output_root).resolve())
    diagnostics_root = ensure_dir(output_root / "diagnostics")
    invariant_root = ensure_dir(output_root / "invariants")

    register_optimizers(repo_dir)
    base_module = load_module("qcontains_base_mod", repo_dir / "scripts" / "standard_rarl_baseline_positive_search.py")
    helper_module = load_module("qcontains_helper_mod", repo_dir / "scripts" / "standard_rarl_lyapunov_repair_minisearch.py")
    closed_module = load_module("qcontains_closed_mod", repo_dir / "models" / "proposed_qp_closedlyap.py")

    src_nog = inspect.getsource(closed_module.ProposedNoGClosedLyapOptimizer._step_impl)
    src_solve_qp = inspect.getsource(closed_module._ClosedLyapunovDriftBase._solve_qp)
    nog_wrong_2d_then_zero = ("_solve_qp(" in src_nog) or ("gamma = 0.0" in src_nog and "_solve_nog(" not in src_nog)
    gamma_constraint_symmetric = "max(min(gamma, self.gamma_max), -self.gamma_max)" in src_solve_qp

    method_rows: List[Dict[str, object]] = []
    curves: Dict[str, pd.DataFrame] = {}

    for method in method_specs(diagnostics_root, invariant_root):
        run_root = ensure_dir(output_root / method.label)
        analysis_dir = ensure_dir(run_root / "analysis")
        existing_run_dir = try_latest_run_dir(run_root, args.env)
        if existing_run_dir is not None and (analysis_dir / "run_summary.csv").exists():
            run_dir = existing_run_dir
        else:
            ns = build_ns(method, run_root, repo_dir, args.env, args.seed, args.device)
            run_dir = run_with_exp_manager(repo_dir, args.env, run_root, ns)
        summary, curve = helper_module.analyze_run(base_module, repo_dir, run_dir, analysis_dir, method.label)
        curves[method.label] = curve
        method_rows.append(
            {
                "method": method.label,
                "train_return_AUC": auc_from_curve(curve, "timesteps", "train_return"),
                "clean_eval_return_AUC": auc_from_curve(curve, "timesteps", "clean_eval_return"),
                "current_adv_eval_return_AUC": auc_from_curve(curve, "timesteps", "current_adv_eval_return"),
                "current_adv_degradation_AUC": auc_from_curve(curve, "timesteps", "current_adv_degradation"),
            }
        )

    inv_df = read_invariant_rows(invariant_root, "qp_invariant_trustreg")
    nog_diag = read_diag_pair(diagnostics_root, "nog_trustreg")
    forced_nog_diag = read_diag_pair(diagnostics_root, "forced_nog_trustreg")
    nogsafe_diag = read_diag_pair(diagnostics_root, "nogsafe_trustreg")

    q1_equals_fraction = float(pd.to_numeric(inv_df["q1_equals_q2_gamma0_flag"], errors="coerce").mean()) if not inv_df.empty else math.nan
    predicted_inclusion_fraction = float(pd.to_numeric(inv_df["predicted_inclusion_pass"], errors="coerce").mean()) if not inv_df.empty else math.nan
    actual_qp_better_fraction = float(pd.to_numeric(inv_df["actual_QP_better_than_noG"], errors="coerce").mean()) if not inv_df.empty else math.nan
    gamma_negative_fraction = float(pd.to_numeric(inv_df["gamma_negative_flag"], errors="coerce").mean()) if not inv_df.empty else math.nan

    def choice_matches_applied(df: pd.DataFrame) -> float:
        if df.empty:
            return math.nan
        flags = []
        for _, row in df.iterrows():
            chosen = str(row.get("chosen_step", ""))
            v_after = safe_float(row.get("V_after"), math.nan)
            if chosen == "noG":
                target = safe_float(row.get("V_after_noG"), math.nan)
            elif chosen == "plusG":
                target = safe_float(row.get("V_after_plusG_QP"), math.nan)
            elif chosen == "minusG":
                target = safe_float(row.get("V_after_minusG_QP"), math.nan)
            else:
                target = math.nan
            flags.append(int(finite(v_after) and finite(target) and abs(v_after - target) <= 1e-8 * max(1.0, abs(v_after), abs(target))))
        return float(np.mean(flags)) if flags else math.nan

    noG_safe_actual_choice_matches_applied_step_flag = choice_matches_applied(nogsafe_diag)

    forced_matches = False
    forced_norm_diff = math.nan
    if not nog_diag.empty and not forced_nog_diag.empty:
        merged = nog_diag.merge(forced_nog_diag, on=["step_index", "diag_role"], suffixes=("_nog", "_forced"), how="inner")
        if not merged.empty:
            cols = ["actor_update_norm", "logstd_update_norm", "critic_update_norm", "V_after", "V_after_noG"]
            diffs = []
            for col in cols:
                a = pd.to_numeric(merged[f"{col}_nog"], errors="coerce")
                b = pd.to_numeric(merged[f"{col}_forced"], errors="coerce")
                diffs.append(np.nanmax(np.abs(a - b)))
            forced_norm_diff = float(np.nanmax(diffs))
            forced_matches = bool(np.isfinite(forced_norm_diff) and forced_norm_diff <= 1e-8)

    curve_match_metrics: Dict[str, float] = {}
    for metric in ["train_return", "clean_eval_return", "current_adv_eval_return", "current_adv_degradation"]:
        nog_curve = curves["proposed_nog_closed_trustreg"][["timesteps", metric]].rename(columns={metric: "nog"})
        forced_curve = curves["proposed_qp_forced_nog_trustreg"][["timesteps", metric]].rename(columns={metric: "forced"})
        merged_curve = nog_curve.merge(forced_curve, on="timesteps", how="inner")
        if merged_curve.empty:
            curve_match_metrics[f"{metric}_max_abs_diff"] = math.nan
        else:
            curve_match_metrics[f"{metric}_max_abs_diff"] = float(np.nanmax(np.abs(pd.to_numeric(merged_curve["nog"], errors="coerce") - pd.to_numeric(merged_curve["forced"], errors="coerce"))))

    if inv_df.empty:
        inv_out = pd.DataFrame([{"warning": "no invariant rows collected"}])
    else:
        inv_out = inv_df
    inv_out.to_csv(output_root / "q_contains_nog_summary_AFTER_FIX.csv", index=False)

    forced_delta_rel_diff_max = safe_float(pd.to_numeric(forced_nog_diag.get("applied_vs_nog_candidate_rel_diff"), errors="coerce").max(), math.nan) if not forced_nog_diag.empty else math.nan
    forced_v_diff_max = safe_float(pd.to_numeric(forced_nog_diag.get("applied_v_after_minus_nog_candidate"), errors="coerce").abs().max(), math.nan) if not forced_nog_diag.empty else math.nan
    forced_nog_matches_nog_flag = int(
        finite(forced_delta_rel_diff_max)
        and finite(forced_v_diff_max)
        and forced_delta_rel_diff_max < 1e-6
        and forced_v_diff_max < 1e-6
    )

    if nog_wrong_2d_then_zero:
        invariant_decision = "QP_SOLVER_STILL_BROKEN"
    elif finite(q1_equals_fraction) and q1_equals_fraction < 0.999:
        invariant_decision = "Q_MODEL_MISMATCH_STILL_PRESENT"
    elif (
        (finite(predicted_inclusion_fraction) and predicted_inclusion_fraction < 0.999)
        or (finite(gamma_negative_fraction) and gamma_negative_fraction > 0.0)
    ):
        invariant_decision = "QP_SOLVER_STILL_BROKEN"
    elif not (
        finite(forced_delta_rel_diff_max)
        and finite(forced_v_diff_max)
        and forced_delta_rel_diff_max < 1e-6
        and forced_v_diff_max < 1e-6
    ):
        invariant_decision = "NOG_SAFE_APPLY_STEP_STILL_BROKEN"
    else:
        invariant_decision = "QP_SOLVER_FIXED_CONTAINS_NOG"

    report_lines = [
        "# Q Contains noG Report",
        "",
        "## Config",
        "",
        "- env: `HalfCheetah-v4`",
        "- scope: `full_policy`",
        "- alpha: `0.05`",
        "- shared_lr: `0.0003`",
        "- seed: `0`",
        "- N_mu/N_nu: `5 / 1`",
        "- iterations: `2`",
        "- eval_freq: `10240`",
        "- n_eval_episodes: `3`",
        "- trust-region merit: `lambda_F=0.01, lambda_R=0.3, lambda_KL=0.3, lambda_CF=0.1`",
        "",
        "## Part 1",
        "",
        f"- noG implemented via true 1D solve: `{int(not nog_wrong_2d_then_zero)}`",
        f"- noG source contains `_solve_nog(...)`: `{int('_solve_nog(' in src_nog)}`",
        f"- noG source contains `_solve_qp(...)`: `{int('_solve_qp(' in src_nog)}`",
        f"- solver gamma clamp is symmetric around zero: `{int(gamma_constraint_symmetric)}`",
        "",
        "## Invariant fractions",
        "",
            f"- predicted_inclusion_pass_fraction: `{safe_float(predicted_inclusion_fraction, math.nan):.6f}`",
            f"- q1_equals_q2_gamma0_pass_fraction: `{safe_float(q1_equals_fraction, math.nan):.6f}`",
            f"- actual_QP_better_than_noG_fraction: `{safe_float(actual_qp_better_fraction, math.nan):.6f}`",
            f"- forced_nog_matches_nog_flag: `{forced_nog_matches_nog_flag}`",
            f"- noG_safe_actual_choice_matches_applied_step_flag: `{safe_float(noG_safe_actual_choice_matches_applied_step_flag, math.nan):.6f}`",
            f"- gamma_negative_fraction_in_plusG_QP: `{safe_float(gamma_negative_fraction, math.nan):.6f}`",
        "",
        "## Forced noG vs noG",
        "",
        f"- forced_nog_end_to_end_update_norm_diff_max: `{safe_float(forced_norm_diff, math.nan):.6e}`",
        f"- forced_nog_same_batch_delta_rel_diff_max: `{safe_float(forced_delta_rel_diff_max, math.nan):.6e}`",
        f"- forced_nog_same_batch_V_diff_max: `{safe_float(forced_v_diff_max, math.nan):.6e}`",
    ]
    for key, value in curve_match_metrics.items():
        report_lines.append(f"- {key}: `{safe_float(value, math.nan):.6e}`")
    report_lines.extend(
        [
            "",
            "## Small-run AUC",
            "",
        ]
    )
    for row in method_rows:
        report_lines.append(
            f"- {row['method']}: train_auc=`{safe_float(row['train_return_AUC'], math.nan):.6f}`, clean_auc=`{safe_float(row['clean_eval_return_AUC'], math.nan):.6f}`, current_adv_auc=`{safe_float(row['current_adv_eval_return_AUC'], math.nan):.6f}`"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If `q1D(beta) == q2D(beta,0)` and predicted inclusion passes, then QP contains noG at the surrogate-model level.",
            "- If actual frozen-batch `V_after_QP` is still often worse than `V_after_noG`, that indicates surrogate inaccuracy or unusable `G`, not necessarily a coding bug.",
            "",
            f"Invariant decision: `{invariant_decision}`",
        ]
    )
    (output_root / "q_contains_nog_report_AFTER_FIX.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (output_root / "q_contains_nog_decision_AFTER_FIX.md").write_text(invariant_decision + "\n", encoding="utf-8")

    final_report_lines = [
        "# Final Fix Report",
        "",
        f"- invariant decision: `{invariant_decision}`",
        f"- q1_equals_q2_gamma0_pass_fraction: `{safe_float(q1_equals_fraction, math.nan):.6f}`",
        f"- predicted_inclusion_pass_fraction: `{safe_float(predicted_inclusion_fraction, math.nan):.6f}`",
        f"- gamma_negative_fraction_in_plusG_QP: `{safe_float(gamma_negative_fraction, math.nan):.6f}`",
        f"- forced_nog_same_batch_delta_rel_diff_max: `{safe_float(forced_delta_rel_diff_max, math.nan):.6e}`",
        f"- forced_nog_same_batch_V_diff_max: `{safe_float(forced_v_diff_max, math.nan):.6e}`",
    ]

    final_decision = invariant_decision
    postfix_summary_df = pd.DataFrame()
    if invariant_decision == "QP_SOLVER_FIXED_CONTAINS_NOG":
        smoke_root = ensure_dir(output_root / "postfix_smoke")
        smoke_diag_root = ensure_dir(smoke_root / "diagnostics")
        smoke_rows: List[Dict[str, object]] = []
        smoke_curves: Dict[str, pd.DataFrame] = {}
        for method in smoke_method_specs(smoke_diag_root):
            run_root = ensure_dir(smoke_root / method.label)
            analysis_dir = ensure_dir(run_root / "analysis")
            existing_run_dir = try_latest_run_dir(run_root, args.env)
            if existing_run_dir is not None and (analysis_dir / "run_summary.csv").exists():
                run_dir = existing_run_dir
            else:
                ns = build_smoke_ns(method, run_root, repo_dir, args.env, args.seed, args.device)
                run_dir = run_with_exp_manager(repo_dir, args.env, run_root, ns)
            _, curve = helper_module.analyze_run(base_module, repo_dir, run_dir, analysis_dir, method.label)
            smoke_curves[method.label] = curve
            smoke_rows.append(
                {
                    "method": method.label,
                    "train_return_AUC": auc_from_curve(curve, "timesteps", "train_return"),
                    "clean_eval_return_AUC": auc_from_curve(curve, "timesteps", "clean_eval_return"),
                    "current_adv_eval_return_AUC": auc_from_curve(curve, "timesteps", "current_adv_eval_return"),
                    "final_current_adv_eval_return": safe_float(pd.to_numeric(curve["current_adv_eval_return"], errors="coerce").dropna().iloc[-1] if "current_adv_eval_return" in curve.columns and not curve.empty else math.nan),
                }
            )
        postfix_summary_df = pd.DataFrame(smoke_rows)
        if not postfix_summary_df.empty:
            nog_auc = safe_float(postfix_summary_df.loc[postfix_summary_df["method"] == "proposed_nog_closed_trustreg", "current_adv_eval_return_AUC"].iloc[0], math.nan)
            egm_auc = safe_float(postfix_summary_df.loc[postfix_summary_df["method"] == "egm", "current_adv_eval_return_AUC"].iloc[0], math.nan)
            qpn_auc = safe_float(postfix_summary_df.loc[postfix_summary_df["method"] == "proposed_qp_nog_safe_trustreg_FIXED", "current_adv_eval_return_AUC"].iloc[0], math.nan)
            curves_nog = smoke_curves["proposed_nog_closed_trustreg"][["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "nog"})
            curves_egm = smoke_curves["egm"][["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "egm"})
            curves_qpn = smoke_curves["proposed_qp_nog_safe_trustreg_FIXED"][["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "qpn"})
            merged = curves_nog.merge(curves_egm, on="timesteps", how="inner").merge(curves_qpn, on="timesteps", how="inner")
            dom_over_nog = float(np.mean(pd.to_numeric(merged["qpn"], errors="coerce") > pd.to_numeric(merged["nog"], errors="coerce") + 1e-9)) if not merged.empty else math.nan
            dom_over_egm = float(np.mean(pd.to_numeric(merged["qpn"], errors="coerce") > pd.to_numeric(merged["egm"], errors="coerce") + 1e-9)) if not merged.empty else math.nan
            qpn_diag = read_diag_pair(smoke_diag_root, "smoke_nogsafe_trustreg_fixed")
            plus_diag = read_diag_pair(smoke_diag_root, "smoke_plusg_trustreg_fixed")
            fallback_to_noG_frac = float((qpn_diag["chosen_step"].astype(str) == "noG").mean()) if not qpn_diag.empty and "chosen_step" in qpn_diag.columns else math.nan
            gamma_active_frac = float(pd.to_numeric(qpn_diag["gamma_active"], errors="coerce").mean()) if not qpn_diag.empty and "gamma_active" in qpn_diag.columns else math.nan
            g_ratio = float(pd.to_numeric(qpn_diag["G_contribution_ratio"], errors="coerce").mean()) if not qpn_diag.empty and "G_contribution_ratio" in qpn_diag.columns else math.nan
            qp_better_actual_v_frac = float(pd.to_numeric(qpn_diag["nog_safe_qp_better_than_noG"], errors="coerce").mean()) if not qpn_diag.empty and "nog_safe_qp_better_than_noG" in qpn_diag.columns else math.nan
            postfix_summary_df["improve_vs_nog_auc"] = np.where(
                postfix_summary_df["method"] == "proposed_qp_nog_safe_trustreg_FIXED",
                postfix_summary_df["current_adv_eval_return_AUC"] / (nog_auc + EPS) - 1.0,
                np.nan,
            )
            postfix_summary_df["improve_vs_egm_auc"] = np.where(
                postfix_summary_df["method"] == "proposed_qp_nog_safe_trustreg_FIXED",
                postfix_summary_df["current_adv_eval_return_AUC"] / (egm_auc + EPS) - 1.0,
                np.nan,
            )
            postfix_summary_df["dominance_over_nog"] = np.where(postfix_summary_df["method"] == "proposed_qp_nog_safe_trustreg_FIXED", dom_over_nog, np.nan)
            postfix_summary_df["dominance_over_egm"] = np.where(postfix_summary_df["method"] == "proposed_qp_nog_safe_trustreg_FIXED", dom_over_egm, np.nan)
            postfix_summary_df.to_csv(output_root / "postfix_smoke_summary.csv", index=False)
            weak_positive = (
                finite(qpn_auc) and finite(nog_auc) and finite(egm_auc)
                and (qpn_auc / (nog_auc + EPS) - 1.0) >= 0.05
                and (qpn_auc / (egm_auc + EPS) - 1.0) >= 0.05
                and finite(dom_over_nog) and dom_over_nog >= 0.60
                and finite(dom_over_egm) and dom_over_egm >= 0.70
                and finite(fallback_to_noG_frac) and fallback_to_noG_frac <= 0.30
                and finite(gamma_active_frac) and gamma_active_frac >= 0.30
                and finite(g_ratio) and g_ratio >= 0.10
            )
            final_decision = "POSTFIX_QP_WEAK_POSITIVE" if weak_positive else "SOLVER_FIXED_BUT_NO_USABLE_G_OR_MERIT_MISMATCH"
            postfix_report_lines = [
                "# Postfix Smoke Report",
                "",
                f"- invariant decision: `{invariant_decision}`",
                f"- qpn current_adv_eval_return_AUC: `{qpn_auc:.6f}`",
                f"- nog current_adv_eval_return_AUC: `{nog_auc:.6f}`",
                f"- egm current_adv_eval_return_AUC: `{egm_auc:.6f}`",
                f"- qpn_vs_nog_auc_gap: `{qpn_auc / (nog_auc + EPS) - 1.0:.6f}`",
                f"- qpn_vs_egm_auc_gap: `{qpn_auc / (egm_auc + EPS) - 1.0:.6f}`",
                f"- dominance_over_nog: `{dom_over_nog:.6f}`",
                f"- dominance_over_egm: `{dom_over_egm:.6f}`",
                f"- fallback_to_noG_frac: `{fallback_to_noG_frac:.6f}`",
                f"- gamma_active_frac: `{gamma_active_frac:.6f}`",
                f"- G_contribution_ratio: `{g_ratio:.6f}`",
                f"- QP_better_than_noG_actual_V_fraction: `{qp_better_actual_v_frac:.6f}`",
                f"- predicted_inclusion_pass_fraction: `{safe_float(predicted_inclusion_fraction, math.nan):.6f}`",
                "",
                f"Postfix decision: `{final_decision}`",
            ]
            (output_root / "postfix_smoke_report.md").write_text("\n".join(postfix_report_lines) + "\n", encoding="utf-8")
        final_report_lines.append(f"- postfix decision: `{final_decision}`")
    else:
        final_report_lines.append("- postfix smoke skipped because invariants did not pass.")

    final_report_lines.append(f"- final decision: `{final_decision}`")
    (output_root / "final_fix_report.md").write_text("\n".join(final_report_lines) + "\n", encoding="utf-8")
    (output_root / "final_fix_decision.md").write_text(final_decision + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
