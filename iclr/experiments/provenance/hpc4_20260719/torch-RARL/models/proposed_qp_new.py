from __future__ import annotations

import csv
import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.optim import Optimizer


NamedParams = List[Tuple[str, torch.nn.Parameter]]
NamedTensorMap = Dict[str, torch.Tensor]


def classify_parameter_block(name: str) -> str:
    if "log_std" in name:
        return "logstd"
    if "value" in name:
        return "critic"
    return "actor"


def clone_named_state(named_params: NamedParams) -> NamedTensorMap:
    return {name: param.data.detach().clone() for name, param in named_params}


def restore_named_state(named_params: NamedParams, state: NamedTensorMap) -> None:
    with torch.no_grad():
        for name, param in named_params:
            param.data.copy_(state[name])


def flatten_named_tensors(named_tensor_map: NamedTensorMap, selected_names: Optional[Sequence[str]] = None) -> torch.Tensor:
    selected = set(selected_names) if selected_names is not None else None
    tensors = [tensor.reshape(-1) for name, tensor in named_tensor_map.items() if selected is None or name in selected]
    if not tensors:
        return torch.zeros(0)
    return torch.cat(tensors)


def tensor_norm(tensor: torch.Tensor) -> float:
    return float(torch.norm(tensor).item()) if tensor.numel() > 0 else 0.0


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    denom = torch.norm(a) * torch.norm(b)
    if denom.item() == 0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def named_difference(new_state: NamedTensorMap, old_state: NamedTensorMap, selected_names: Optional[Sequence[str]] = None) -> NamedTensorMap:
    selected = set(selected_names) if selected_names is not None else None
    diff: NamedTensorMap = {}
    for name, tensor in new_state.items():
        if selected is not None and name not in selected:
            continue
        diff[name] = tensor - old_state[name]
    return diff


def block_norm(named_tensor_map: NamedTensorMap, selected_names: Sequence[str], block: str) -> float:
    tensors = [tensor.reshape(-1) for name, tensor in named_tensor_map.items() if name in selected_names and classify_parameter_block(name) == block]
    if not tensors:
        return 0.0
    return tensor_norm(torch.cat(tensors))


def apply_state_delta(old_state: NamedTensorMap, delta: NamedTensorMap) -> NamedTensorMap:
    new_state = {name: tensor.clone() for name, tensor in old_state.items()}
    for name, update in delta.items():
        new_state[name] = old_state[name] + update
    return new_state


def parse_step_grid(raw_value: object) -> List[float]:
    if isinstance(raw_value, str):
        parts = [part.strip() for part in raw_value.split(",") if part.strip()]
        values = [float(part) for part in parts]
    elif isinstance(raw_value, (list, tuple)):
        values = [float(part) for part in raw_value]
    else:
        raise ValueError(f"Unsupported qp_step_grid={raw_value!r}")
    if 0.0 not in values:
        values = [0.0] + values
    values = sorted({float(value) for value in values})
    return values


class _ProposedDirectionGridBase(Optimizer):
    requires_eval_closure = True

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1.0,
        optimizer_scope: str = "full_policy",
        qp_normalization: str = "block",
        qp_beta_max: float = 1.0,
        qp_gamma_max: float = 1.0,
        qp_max_update_norm: float = 1.0,
        qp_alpha: float = 0.3,
        qp_objective: str = "loss",
        qp_step_grid: object = "0,0.1,0.3,1.0,3.0",
        qp_accept_rule: str = "none",
        qp_min_g_contribution: float = 0.0,
        qp_critic_weight: float = 1.0,
        qp_g_sign: str = "plus",
        qp_eps: float = 1e-8,
        diagnostics_csv_path: Optional[str] = None,
        role: str = "policy",
        disable_g: bool = False,
    ):
        super().__init__(params, defaults=dict(lr=lr))
        if optimizer_scope not in {"full_policy", "actor_game"}:
            raise ValueError(f"Unsupported optimizer_scope={optimizer_scope!r}")
        if qp_normalization not in {"global", "block"}:
            raise ValueError(f"Unsupported qp_normalization={qp_normalization!r}")
        if qp_objective not in {"loss", "field_energy", "mixed"}:
            raise ValueError(f"Unsupported qp_objective={qp_objective!r}")
        if qp_accept_rule not in {"none", "same_minibatch_loss"}:
            raise ValueError(f"Unsupported qp_accept_rule={qp_accept_rule!r}")
        if qp_g_sign not in {"plus", "minus", "auto_probe"}:
            raise ValueError(f"Unsupported qp_g_sign={qp_g_sign!r}")
        self.optimizer_scope = optimizer_scope
        self.qp_normalization = qp_normalization
        self.qp_beta_max = float(qp_beta_max)
        self.qp_gamma_max = float(qp_gamma_max)
        self.qp_max_update_norm = float(qp_max_update_norm)
        self.qp_alpha = max(float(qp_alpha), 1e-12)
        self.qp_objective = qp_objective
        self.qp_step_grid = parse_step_grid(qp_step_grid)
        self.qp_accept_rule = qp_accept_rule
        self.qp_min_g_contribution = float(qp_min_g_contribution)
        self.qp_critic_weight = float(qp_critic_weight)
        self.qp_g_sign = qp_g_sign
        self.qp_eps = max(float(qp_eps), 1e-12)
        self.diagnostics_csv_path = diagnostics_csv_path
        self.role = role
        self.disable_g = bool(disable_g)
        self.last_step_metrics: Dict[str, float | int | str] = {}
        self._step_index = 0
        self._fieldnames = [
            "step_index",
            "active_role",
            "scope",
            "g_sign_requested",
            "g_sign_used",
            "qp_normalization",
            "qp_objective",
            "qp_accept_rule",
            "F_norm_raw",
            "F_norm_used",
            "G_norm_raw",
            "G_norm_used",
            "cosine_F_G",
            "beta",
            "gamma",
            "beta_selected",
            "gamma_selected",
            "update_norm",
            "actor_update_norm",
            "logstd_update_norm",
            "critic_update_norm",
            "same_minibatch_loss_before",
            "same_minibatch_loss_after",
            "same_minibatch_loss_change",
            "policy_loss_change",
            "value_loss_change",
            "entropy_change",
            "approx_kl_change",
            "clip_fraction_change",
            "G_contribution_norm",
            "G_over_update_norm",
            "selected_candidate_rank",
            "best_noG_candidate_loss",
            "best_qp_candidate_loss",
            "gamma_active_frac",
            "zero_update_flag",
        ]
        self._ensure_diagnostics_header()

    def _ensure_diagnostics_header(self) -> None:
        if not self.diagnostics_csv_path:
            return
        os.makedirs(os.path.dirname(self.diagnostics_csv_path), exist_ok=True)
        if os.path.exists(self.diagnostics_csv_path):
            return
        with open(self.diagnostics_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()

    def _write_diagnostics_row(self, metrics: Dict[str, object]) -> None:
        if not self.diagnostics_csv_path:
            return
        with open(self.diagnostics_csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writerow({field: metrics.get(field) for field in self._fieldnames})

    def _selected_names(self, named_params: NamedParams) -> List[str]:
        if self.optimizer_scope == "full_policy":
            return [name for name, _ in named_params]
        return [name for name, _ in named_params if classify_parameter_block(name) != "critic"]

    def _apply_block_weight(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if classify_parameter_block(name) == "critic":
            return tensor * self.qp_critic_weight
        return tensor

    def _weighted_map(self, tensor_map: NamedTensorMap, selected_names: Sequence[str]) -> NamedTensorMap:
        return {name: self._apply_block_weight(name, tensor_map[name]) for name in selected_names}

    def _normalize_direction(self, tensor_map: NamedTensorMap, selected_names: Sequence[str]) -> NamedTensorMap:
        weighted = self._weighted_map(tensor_map, selected_names)
        if self.qp_normalization == "global":
            norm = max(tensor_norm(flatten_named_tensors(weighted, selected_names)), self.qp_eps)
            return {name: weighted[name] / norm for name in selected_names}

        normalized: NamedTensorMap = {}
        block_norms: Dict[str, float] = {}
        for block in ("actor", "logstd", "critic"):
            block_tensors = [weighted[name].reshape(-1) for name in selected_names if classify_parameter_block(name) == block]
            block_norms[block] = max(tensor_norm(torch.cat(block_tensors)) if block_tensors else 0.0, self.qp_eps)
        for name in selected_names:
            normalized[name] = weighted[name] / block_norms[classify_parameter_block(name)]
        return normalized

    def _field_energy(self, grads: NamedTensorMap, selected_names: Sequence[str]) -> float:
        weighted = self._weighted_map(grads, selected_names)
        vec = flatten_named_tensors(weighted, selected_names)
        return 0.5 * float(torch.dot(vec, vec).item()) if vec.numel() > 0 else 0.0

    def _build_update(
        self,
        *,
        selected_names: Sequence[str],
        f_dir_map: NamedTensorMap,
        g_dir_map: NamedTensorMap,
        beta: float,
        gamma: float,
    ) -> Tuple[NamedTensorMap, float]:
        step_scale = float(self.param_groups[0]["lr"])
        update_map: NamedTensorMap = {}
        for name in selected_names:
            update_map[name] = step_scale * ((-beta * f_dir_map[name]) + (gamma * g_dir_map[name]))
        update_vec = flatten_named_tensors(update_map, selected_names)
        update_norm = tensor_norm(update_vec)
        if math.isfinite(self.qp_max_update_norm) and self.qp_max_update_norm > 0.0 and update_norm > self.qp_max_update_norm:
            scale = self.qp_max_update_norm / max(update_norm, self.qp_eps)
            for name in selected_names:
                update_map[name] = update_map[name] * scale
            update_vec = flatten_named_tensors(update_map, selected_names)
            update_norm = tensor_norm(update_vec)
        return update_map, update_norm

    def _candidate_objective(
        self,
        *,
        base_eval: Dict[str, object],
        candidate_eval: Dict[str, object],
        selected_names: Sequence[str],
    ) -> float:
        if self.qp_objective == "loss":
            return float(candidate_eval["total_loss"])
        base_energy = max(self._field_energy(base_eval["grads"], selected_names), self.qp_eps)
        candidate_energy = self._field_energy(candidate_eval["grads"], selected_names)
        if self.qp_objective == "field_energy":
            return float(candidate_energy)
        base_loss_scale = max(abs(float(base_eval["total_loss"])), 1.0)
        return float(candidate_eval["total_loss"]) / base_loss_scale + float(candidate_energy) / base_energy

    def _grid_values(self, max_value: float) -> List[float]:
        values = [value for value in self.qp_step_grid if value >= 0.0]
        if math.isfinite(max_value):
            values = [value for value in values if value <= max_value + self.qp_eps]
        return values

    def _run_candidate_search(
        self,
        *,
        theta_old: NamedTensorMap,
        named_params: NamedParams,
        eval_closure: Callable[..., Dict[str, object]],
        selected_names: Sequence[str],
        base_eval: Dict[str, object],
        f_dir_map: NamedTensorMap,
        g_dir_map: NamedTensorMap,
        allow_g: bool,
    ) -> Dict[str, object]:
        beta_values = self._grid_values(self.qp_beta_max)
        gamma_values = [0.0] if not allow_g else self._grid_values(self.qp_gamma_max)
        zero_candidate = {
            "beta": 0.0,
            "gamma": 0.0,
            "objective": self._candidate_objective(base_eval=base_eval, candidate_eval=base_eval, selected_names=selected_names),
            "eval": base_eval,
            "update_map": {name: torch.zeros_like(theta_old[name]) for name in selected_names},
            "update_norm": 0.0,
            "g_contribution_norm": 0.0,
        }
        candidates = [zero_candidate]

        for beta in beta_values:
            for gamma in gamma_values:
                if beta == 0.0 and gamma == 0.0:
                    continue
                update_map, update_norm = self._build_update(
                    selected_names=selected_names,
                    f_dir_map=f_dir_map,
                    g_dir_map=g_dir_map,
                    beta=beta,
                    gamma=gamma,
                )
                g_contribution_norm = float(self.param_groups[0]["lr"]) * abs(gamma) * tensor_norm(flatten_named_tensors(g_dir_map, selected_names))
                if gamma > 0.0 and self.qp_min_g_contribution > 0.0:
                    if g_contribution_norm / max(update_norm, self.qp_eps) < self.qp_min_g_contribution:
                        continue
                theta_candidate = apply_state_delta(theta_old, update_map)
                candidate_eval = eval_closure(theta_override=theta_candidate, backward=self.qp_objective != "loss")
                objective = self._candidate_objective(base_eval=base_eval, candidate_eval=candidate_eval, selected_names=selected_names)
                candidates.append(
                    {
                        "beta": beta,
                        "gamma": gamma,
                        "objective": objective,
                        "eval": candidate_eval,
                        "update_map": update_map,
                        "update_norm": update_norm,
                        "g_contribution_norm": g_contribution_norm,
                    }
                )

        candidates = sorted(candidates, key=lambda item: float(item["objective"]))
        selected = candidates[0]
        if self.qp_accept_rule == "same_minibatch_loss" and float(selected["eval"]["total_loss"]) > float(base_eval["total_loss"]):
            selected = zero_candidate
        selected_rank = next(index for index, item in enumerate(candidates, start=1) if item["beta"] == selected["beta"] and item["gamma"] == selected["gamma"])
        best_nog_candidate_loss = min(float(item["eval"]["total_loss"]) for item in candidates if float(item["gamma"]) == 0.0)
        return {
            "selected": selected,
            "selected_rank": selected_rank,
            "best_nog_candidate_loss": best_nog_candidate_loss,
            "best_qp_candidate_loss": float(candidates[0]["eval"]["total_loss"]),
        }

    def _choose_g_raw(
        self,
        *,
        theta_old: NamedTensorMap,
        named_params: NamedParams,
        eval_closure: Callable[..., Dict[str, object]],
        selected_names: Sequence[str],
        base_eval: Dict[str, object],
        f_dir_map: NamedTensorMap,
        old_grads: NamedTensorMap,
    ) -> Tuple[str, NamedTensorMap, Dict[str, object]]:
        theta_half = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            theta_half[name] = theta_old[name] - self.qp_alpha * f_dir_map[name]
        half_eval = eval_closure(theta_override=theta_half, backward=True)
        plus_map = {name: old_grads[name] - half_eval["grads"][name].detach().clone() for name in selected_names}
        minus_map = {name: half_eval["grads"][name].detach().clone() - old_grads[name] for name in selected_names}
        if self.disable_g:
            return "plus", {name: torch.zeros_like(old_grads[name]) for name in selected_names}, half_eval
        if self.qp_g_sign == "plus":
            return "plus", plus_map, half_eval
        if self.qp_g_sign == "minus":
            return "minus", minus_map, half_eval

        plus_dir = self._normalize_direction(plus_map, selected_names)
        minus_dir = self._normalize_direction(minus_map, selected_names)
        plus_search = self._run_candidate_search(
            theta_old=theta_old,
            named_params=named_params,
            eval_closure=eval_closure,
            selected_names=selected_names,
            base_eval=base_eval,
            f_dir_map=f_dir_map,
            g_dir_map=plus_dir,
            allow_g=True,
        )
        minus_search = self._run_candidate_search(
            theta_old=theta_old,
            named_params=named_params,
            eval_closure=eval_closure,
            selected_names=selected_names,
            base_eval=base_eval,
            f_dir_map=f_dir_map,
            g_dir_map=minus_dir,
            allow_g=True,
        )
        if float(plus_search["selected"]["eval"]["total_loss"]) <= float(minus_search["selected"]["eval"]["total_loss"]):
            return "plus", plus_map, half_eval
        return "minus", minus_map, half_eval

    def step(  # type: ignore[override]
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
        *,
        eval_closure: Optional[Callable[..., Dict[str, object]]] = None,
        named_params: Optional[NamedParams] = None,
    ):
        del closure
        if eval_closure is None or named_params is None:
            raise ValueError("proposed_*_new optimizers require eval_closure and named_params")

        selected_names = self._selected_names(named_params)
        theta_old = clone_named_state(named_params)
        base_eval = eval_closure(theta_override=theta_old, backward=True)
        old_grads = {name: base_eval["grads"][name].detach().clone() for name in selected_names}
        f_raw_vec = flatten_named_tensors(old_grads, selected_names)
        f_dir_map = self._normalize_direction(old_grads, selected_names)
        f_dir_vec = flatten_named_tensors(f_dir_map, selected_names)

        g_sign_used, g_raw_map, _ = self._choose_g_raw(
            theta_old=theta_old,
            named_params=named_params,
            eval_closure=eval_closure,
            selected_names=selected_names,
            base_eval=base_eval,
            f_dir_map=f_dir_map,
            old_grads=old_grads,
        )
        g_raw_vec = flatten_named_tensors(g_raw_map, selected_names)
        g_dir_map = self._normalize_direction(g_raw_map, selected_names) if not self.disable_g else {name: torch.zeros_like(old_grads[name]) for name in selected_names}
        g_dir_vec = flatten_named_tensors(g_dir_map, selected_names)

        search = self._run_candidate_search(
            theta_old=theta_old,
            named_params=named_params,
            eval_closure=eval_closure,
            selected_names=selected_names,
            base_eval=base_eval,
            f_dir_map=f_dir_map,
            g_dir_map=g_dir_map,
            allow_g=not self.disable_g,
        )
        selected = search["selected"]
        theta_new = apply_state_delta(theta_old, selected["update_map"])
        restore_named_state(named_params, theta_new)
        diff = named_difference(theta_new, theta_old, selected_names)

        metrics: Dict[str, object] = {
            "step_index": self._step_index,
            "active_role": self.role,
            "scope": self.optimizer_scope,
            "g_sign_requested": self.qp_g_sign,
            "g_sign_used": g_sign_used,
            "qp_normalization": self.qp_normalization,
            "qp_objective": self.qp_objective,
            "qp_accept_rule": self.qp_accept_rule,
            "F_norm_raw": tensor_norm(f_raw_vec),
            "F_norm_used": tensor_norm(f_dir_vec),
            "G_norm_raw": tensor_norm(g_raw_vec),
            "G_norm_used": tensor_norm(g_dir_vec),
            "cosine_F_G": cosine_similarity(f_raw_vec, g_raw_vec),
            "beta": float(selected["beta"]),
            "gamma": float(selected["gamma"]),
            "beta_selected": float(selected["beta"]),
            "gamma_selected": float(selected["gamma"]),
            "update_norm": float(selected["update_norm"]),
            "actor_update_norm": block_norm(diff, selected_names, "actor"),
            "logstd_update_norm": block_norm(diff, selected_names, "logstd"),
            "critic_update_norm": block_norm(diff, selected_names, "critic"),
            "same_minibatch_loss_before": float(base_eval["total_loss"]),
            "same_minibatch_loss_after": float(selected["eval"]["total_loss"]),
            "same_minibatch_loss_change": float(selected["eval"]["total_loss"]) - float(base_eval["total_loss"]),
            "policy_loss_change": float(selected["eval"]["policy_loss"]) - float(base_eval["policy_loss"]),
            "value_loss_change": float(selected["eval"]["value_loss"]) - float(base_eval["value_loss"]),
            "entropy_change": float(selected["eval"]["entropy_loss"]) - float(base_eval["entropy_loss"]),
            "approx_kl_change": float(selected["eval"]["approx_kl"]) - float(base_eval["approx_kl"]),
            "clip_fraction_change": float(selected["eval"]["clip_fraction"]) - float(base_eval["clip_fraction"]),
            "G_contribution_norm": float(selected["g_contribution_norm"]),
            "G_over_update_norm": float(selected["g_contribution_norm"]) / max(float(selected["update_norm"]), self.qp_eps),
            "selected_candidate_rank": int(search["selected_rank"]),
            "best_noG_candidate_loss": float(search["best_nog_candidate_loss"]),
            "best_qp_candidate_loss": float(search["best_qp_candidate_loss"]),
            "gamma_active_frac": float(float(selected["gamma"]) > self.qp_eps),
            "zero_update_flag": int(float(selected["update_norm"]) <= self.qp_eps),
        }
        self.last_step_metrics = metrics
        self._write_diagnostics_row(metrics)
        self._step_index += 1
        return selected["eval"]["loss_tensor"]


class ProposedNoGNewOptimizer(_ProposedDirectionGridBase):
    def __init__(self, params: Iterable[torch.nn.Parameter], **kwargs):
        kwargs = dict(kwargs)
        kwargs["disable_g"] = True
        super().__init__(params, **kwargs)


class ProposedQPNewOptimizer(_ProposedDirectionGridBase):
    pass


def _is_finite_named_map(tensor_map: NamedTensorMap, selected_names: Sequence[str]) -> bool:
    for name in selected_names:
        tensor = tensor_map[name]
        if not torch.isfinite(tensor).all():
            return False
    return True


def _block_mean_square(tensor_map: NamedTensorMap, selected_names: Sequence[str], block: str) -> float:
    tensors = [tensor_map[name].reshape(-1) for name in selected_names if classify_parameter_block(name) == block]
    if not tensors:
        return 0.0
    vec = torch.cat(tensors)
    return float(torch.mean(vec * vec).item())


class _LyapunovQPBaseV2(Optimizer):
    requires_eval_closure = True

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1.0,
        optimizer_scope: str = "full_policy",
        qp_normalization: str = "block",
        qp_fd_eps: float = 1e-3,
        qp_beta_probe: float = 1e-3,
        qp_gamma_probe: float = 1e-3,
        qp_ridge: float = 1e-8,
        qp_actor_weight: float = 1.0,
        qp_logstd_weight: float = 1.0,
        qp_critic_weight: float = 0.3,
        qp_beta_max: float = 0.3,
        qp_gamma_max: float = 0.3,
        qp_max_update_norm: float = float("inf"),
        qp_eps: float = 1e-8,
        qp_step_solver: str = "lyapunov_quadratic_bound",
        diagnostics_csv_path: Optional[str] = None,
        role: str = "policy",
        disable_g: bool = False,
    ):
        super().__init__(params, defaults=dict(lr=lr))
        if optimizer_scope not in {"full_policy", "actor_game"}:
            raise ValueError(f"Unsupported optimizer_scope={optimizer_scope!r}")
        if qp_normalization not in {"global", "block"}:
            raise ValueError(f"Unsupported qp_normalization={qp_normalization!r}")
        if qp_step_solver != "lyapunov_quadratic_bound":
            raise ValueError(
                "proposed_*_new_v2 only supports qp_step_solver='lyapunov_quadratic_bound'; "
                "legacy loss-grid logic remains in proposed_*_new"
            )
        self.optimizer_scope = optimizer_scope
        self.qp_normalization = qp_normalization
        self.qp_fd_eps = max(float(qp_fd_eps), 1e-12)
        self.qp_beta_probe = max(float(qp_beta_probe), 1e-12)
        self.qp_gamma_probe = max(float(qp_gamma_probe), 1e-12)
        self.qp_ridge = max(float(qp_ridge), 0.0)
        self.qp_actor_weight = float(qp_actor_weight)
        self.qp_logstd_weight = float(qp_logstd_weight)
        self.qp_critic_weight = float(qp_critic_weight)
        self.qp_beta_max = max(float(qp_beta_max), 0.0)
        self.qp_gamma_max = max(float(qp_gamma_max), 0.0)
        self.qp_max_update_norm = float(qp_max_update_norm)
        self.qp_eps = max(float(qp_eps), 1e-12)
        self.qp_step_solver = qp_step_solver
        self.diagnostics_csv_path = diagnostics_csv_path
        self.role = role
        self.disable_g = bool(disable_g)
        self.last_step_metrics: Dict[str, float | int | str] = {}
        self._step_index = 0
        self._fieldnames = [
            "step_index",
            "active_role",
            "scope",
            "qp_step_solver",
            "qp_normalization",
            "eta",
            "finite_difference_valid",
            "fd_eps",
            "F_raw_norm",
            "F_dir_norm",
            "G_raw_norm",
            "G_dir_norm",
            "cosine_F_G",
            "a",
            "b",
            "c",
            "h",
            "k",
            "H_det",
            "ridge_used",
            "q_condition_status",
            "beta_star_unclipped",
            "gamma_star_unclipped",
            "beta",
            "gamma",
            "beta_eff",
            "gamma_eff",
            "selected_case",
            "beta_at_bound",
            "gamma_at_bound",
            "gamma_active",
            "q_pred",
            "V_before",
            "V_after_actual",
            "actual_V_change",
            "loss_before",
            "loss_after_actual",
            "actual_loss_change",
            "approx_kl_after",
            "clip_fraction_after",
            "F_contribution_norm",
            "G_contribution_norm",
            "G_over_update_norm",
            "update_norm_pre_cap",
            "update_norm_post_cap",
            "cap_active",
            "actor_update_norm",
            "logstd_update_norm",
            "critic_update_norm",
            "zero_update_flag",
        ]
        self._ensure_diagnostics_header()

    def _ensure_diagnostics_header(self) -> None:
        if not self.diagnostics_csv_path:
            return
        os.makedirs(os.path.dirname(self.diagnostics_csv_path), exist_ok=True)
        if os.path.exists(self.diagnostics_csv_path):
            return
        with open(self.diagnostics_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()

    def _write_diagnostics_row(self, metrics: Dict[str, object]) -> None:
        if not self.diagnostics_csv_path:
            return
        with open(self.diagnostics_csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writerow({field: metrics.get(field) for field in self._fieldnames})

    def _selected_names(self, named_params: NamedParams) -> List[str]:
        if self.optimizer_scope == "full_policy":
            return [name for name, _ in named_params]
        return [name for name, _ in named_params if classify_parameter_block(name) != "critic"]

    def _normalize_direction(self, tensor_map: NamedTensorMap, selected_names: Sequence[str]) -> NamedTensorMap:
        if self.qp_normalization == "global":
            norm = max(tensor_norm(flatten_named_tensors(tensor_map, selected_names)), self.qp_eps)
            return {name: tensor_map[name] / norm for name in selected_names}

        normalized: NamedTensorMap = {}
        block_norms: Dict[str, float] = {}
        for block in ("actor", "logstd", "critic"):
            block_tensors = [tensor_map[name].reshape(-1) for name in selected_names if classify_parameter_block(name) == block]
            block_norms[block] = max(tensor_norm(torch.cat(block_tensors)) if block_tensors else 0.0, self.qp_eps)
        for name in selected_names:
            normalized[name] = tensor_map[name] / block_norms[classify_parameter_block(name)]
        return normalized

    def _lyapunov_weights(self) -> Dict[str, float]:
        return {
            "actor": self.qp_actor_weight,
            "logstd": self.qp_logstd_weight,
            "critic": self.qp_critic_weight,
        }

    def _lyapunov_value(self, grads: NamedTensorMap, selected_names: Sequence[str]) -> float:
        weights = self._lyapunov_weights()
        total = 0.0
        for block, weight in weights.items():
            total += weight * _block_mean_square(grads, selected_names, block)
        return 0.5 * total

    def _evaluate_state(
        self,
        *,
        eval_closure: Callable[..., Dict[str, object]],
        theta_state: NamedTensorMap,
        selected_names: Sequence[str],
        backward: bool,
    ) -> Dict[str, object]:
        eval_info = eval_closure(theta_override=theta_state, backward=backward)
        grads = {name: eval_info["grads"][name].detach().clone() for name in selected_names}
        eval_info = dict(eval_info)
        eval_info["grads_selected"] = grads
        eval_info["V"] = self._lyapunov_value(grads, selected_names)
        return eval_info

    def _state_plus_direction(
        self,
        theta_state: NamedTensorMap,
        direction_map: NamedTensorMap,
        scale: float,
        selected_names: Sequence[str],
    ) -> NamedTensorMap:
        theta_new = {name: tensor.clone() for name, tensor in theta_state.items()}
        for name in selected_names:
            theta_new[name] = theta_state[name] + scale * direction_map[name]
        return theta_new

    def _compute_g_direction(
        self,
        *,
        theta_old: NamedTensorMap,
        eval_closure: Callable[..., Dict[str, object]],
        selected_names: Sequence[str],
        f_dir_map: NamedTensorMap,
        base_eval: Dict[str, object],
    ) -> Tuple[NamedTensorMap, NamedTensorMap, bool]:
        if self.disable_g:
            zero_map = {name: torch.zeros_like(base_eval["grads_selected"][name]) for name in selected_names}
            return zero_map, zero_map, False
        theta_plus = self._state_plus_direction(theta_old, f_dir_map, self.qp_fd_eps, selected_names)
        plus_eval = self._evaluate_state(eval_closure=eval_closure, theta_state=theta_plus, selected_names=selected_names, backward=True)
        g_raw_map = {
            name: (plus_eval["grads_selected"][name] - base_eval["grads_selected"][name]) / self.qp_fd_eps
            for name in selected_names
        }
        valid = _is_finite_named_map(g_raw_map, selected_names) and tensor_norm(flatten_named_tensors(g_raw_map, selected_names)) > self.qp_eps
        if not valid:
            zero_map = {name: torch.zeros_like(base_eval["grads_selected"][name]) for name in selected_names}
            return zero_map, zero_map, False
        g_dir_map = self._normalize_direction(g_raw_map, selected_names)
        return g_raw_map, g_dir_map, True

    def _estimate_quadratic_coefficients(
        self,
        *,
        theta_old: NamedTensorMap,
        eval_closure: Callable[..., Dict[str, object]],
        selected_names: Sequence[str],
        f_dir_map: NamedTensorMap,
        g_dir_map: NamedTensorMap,
        v0: float,
        eta: float,
    ) -> Dict[str, float | str]:
        p_map = {name: -f_dir_map[name] for name in selected_names}
        r_map = {name: g_dir_map[name] for name in selected_names}
        db = self.qp_beta_probe
        dg = self.qp_gamma_probe

        def V_at(beta_scale: float, gamma_scale: float) -> float:
            theta_tmp = {name: tensor.clone() for name, tensor in theta_old.items()}
            for name in selected_names:
                theta_tmp[name] = theta_old[name] + eta * beta_scale * p_map[name] + eta * gamma_scale * r_map[name]
            eval_info = self._evaluate_state(eval_closure=eval_closure, theta_state=theta_tmp, selected_names=selected_names, backward=True)
            return float(eval_info["V"])

        vp_plus = V_at(db, 0.0)
        vp_minus = V_at(-db, 0.0)
        vr_plus = V_at(0.0, dg)
        vr_minus = V_at(0.0, -dg)
        vpp = V_at(db, dg)
        vpm = V_at(db, -dg)
        vmp = V_at(-db, dg)
        vmm = V_at(-db, -dg)

        a = (vp_plus - vp_minus) / (2.0 * db)
        c = (vp_plus - 2.0 * v0 + vp_minus) / (db * db)
        b = (vr_plus - vr_minus) / (2.0 * dg)
        k = (vr_plus - 2.0 * v0 + vr_minus) / (dg * dg)
        h = (vpp - vpm - vmp + vmm) / (4.0 * db * dg)

        ridge_used = 0.0
        det = c * k - h * h
        condition = "pd"
        if (c <= self.qp_eps) or (k <= self.qp_eps) or (det <= self.qp_eps) or not math.isfinite(det):
            ridge_used = max(self.qp_ridge, 1e-12)
            c_r = c
            k_r = k
            det_r = det
            for _ in range(8):
                c_r = c + ridge_used
                k_r = k + ridge_used
                det_r = c_r * k_r - h * h
                if (c_r > self.qp_eps) and (k_r > self.qp_eps) and (det_r > self.qp_eps) and math.isfinite(det_r):
                    c = c_r
                    k = k_r
                    det = det_r
                    condition = "ridge_pd"
                    break
                ridge_used *= 10.0
            else:
                c = c_r
                k = k_r
                det = det_r
                condition = "indefinite_after_ridge"

        return {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "h": float(h),
            "k": float(k),
            "H_det": float(det),
            "ridge_used": float(ridge_used),
            "q_condition_status": condition,
        }

    @staticmethod
    def _q_value(beta: float, gamma: float, a: float, b: float, c: float, h: float, k: float) -> float:
        return a * beta + b * gamma + 0.5 * c * beta * beta + h * beta * gamma + 0.5 * k * gamma * gamma

    def _solve_no_g(
        self,
        *,
        a: float,
        c: float,
    ) -> Dict[str, float | str]:
        beta_star = 0.0
        if c > self.qp_eps and math.isfinite(c):
            beta_star = -a / c
            beta = min(max(beta_star, 0.0), self.qp_beta_max)
            candidates = [("interior" if 0.0 <= beta_star <= self.qp_beta_max else "beta_bound", beta)]
        else:
            candidates = [("zero", 0.0), ("beta_max", self.qp_beta_max)]
        best_case = "zero"
        best_beta = 0.0
        best_q = float("inf")
        for case, beta in candidates:
            q_val = self._q_value(beta, 0.0, a, 0.0, c, 0.0, 1.0)
            if q_val < best_q:
                best_q = q_val
                best_beta = beta
                best_case = case
        return {
            "beta_star_unclipped": float(beta_star),
            "gamma_star_unclipped": 0.0,
            "beta": float(best_beta),
            "gamma": 0.0,
            "selected_case": best_case,
            "beta_at_bound": int(abs(best_beta - self.qp_beta_max) <= self.qp_eps and self.qp_beta_max > 0.0),
            "gamma_at_bound": 0,
            "gamma_active": 0,
            "q_pred": float(best_q),
        }

    def _solve_two_direction_qp(
        self,
        *,
        a: float,
        b: float,
        c: float,
        h: float,
        k: float,
        det: float,
        condition: str,
    ) -> Dict[str, float | str]:
        candidates: List[Tuple[str, float, float]] = [
            ("corner_00", 0.0, 0.0),
            ("corner_b0", self.qp_beta_max, 0.0),
            ("corner_0g", 0.0, self.qp_gamma_max),
            ("corner_bg", self.qp_beta_max, self.qp_gamma_max),
        ]

        beta_star = 0.0
        gamma_star = 0.0
        interior_valid = False
        if condition in {"pd", "ridge_pd"} and det > self.qp_eps:
            beta_star = (-a * k + h * b) / det
            gamma_star = (-c * b + h * a) / det
            if (
                math.isfinite(beta_star)
                and math.isfinite(gamma_star)
                and 0.0 <= beta_star <= self.qp_beta_max
                and 0.0 <= gamma_star <= self.qp_gamma_max
            ):
                candidates.append(("interior", beta_star, gamma_star))
                interior_valid = True

        if c > self.qp_eps:
            beta_line = min(max(-a / c, 0.0), self.qp_beta_max)
            candidates.append(("gamma0", beta_line, 0.0))
            beta_on_gamma_max = min(max(-(a + h * self.qp_gamma_max) / c, 0.0), self.qp_beta_max)
            candidates.append(("gamma_max", beta_on_gamma_max, self.qp_gamma_max))
        if k > self.qp_eps:
            gamma_line = min(max(-b / k, 0.0), self.qp_gamma_max)
            candidates.append(("beta0", 0.0, gamma_line))
            gamma_on_beta_max = min(max(-(b + h * self.qp_beta_max) / k, 0.0), self.qp_gamma_max)
            candidates.append(("beta_max", self.qp_beta_max, gamma_on_beta_max))

        best_case = "corner_00"
        best_beta = 0.0
        best_gamma = 0.0
        best_q = float("inf")
        for case, beta, gamma in candidates:
            q_val = self._q_value(beta, gamma, a, b, c, h, k)
            if q_val < best_q:
                best_q = q_val
                best_beta = beta
                best_gamma = gamma
                best_case = case
        return {
            "beta_star_unclipped": float(beta_star),
            "gamma_star_unclipped": float(gamma_star),
            "beta": float(best_beta),
            "gamma": float(best_gamma),
            "selected_case": best_case,
            "beta_at_bound": int(abs(best_beta - self.qp_beta_max) <= self.qp_eps and self.qp_beta_max > 0.0),
            "gamma_at_bound": int(abs(best_gamma - self.qp_gamma_max) <= self.qp_eps and self.qp_gamma_max > 0.0),
            "gamma_active": int(best_gamma > self.qp_eps),
            "q_pred": float(best_q),
            "interior_solution_valid": int(interior_valid),
        }

    def step(  # type: ignore[override]
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
        *,
        eval_closure: Optional[Callable[..., Dict[str, object]]] = None,
        named_params: Optional[NamedParams] = None,
    ):
        del closure
        if eval_closure is None or named_params is None:
            raise ValueError("proposed_*_new_v2 optimizers require eval_closure and named_params")

        selected_names = self._selected_names(named_params)
        theta_old = clone_named_state(named_params)
        eta = float(self.param_groups[0]["lr"])
        base_eval = self._evaluate_state(eval_closure=eval_closure, theta_state=theta_old, selected_names=selected_names, backward=True)
        f_raw_map = {name: base_eval["grads_selected"][name].detach().clone() for name in selected_names}
        f_dir_map = self._normalize_direction(f_raw_map, selected_names)
        f_raw_vec = flatten_named_tensors(f_raw_map, selected_names)
        f_dir_vec = flatten_named_tensors(f_dir_map, selected_names)

        g_raw_map, g_dir_map, finite_difference_valid = self._compute_g_direction(
            theta_old=theta_old,
            eval_closure=eval_closure,
            selected_names=selected_names,
            f_dir_map=f_dir_map,
            base_eval=base_eval,
        )
        g_raw_vec = flatten_named_tensors(g_raw_map, selected_names)
        g_dir_vec = flatten_named_tensors(g_dir_map, selected_names)

        coeffs = self._estimate_quadratic_coefficients(
            theta_old=theta_old,
            eval_closure=eval_closure,
            selected_names=selected_names,
            f_dir_map=f_dir_map,
            g_dir_map=g_dir_map,
            v0=float(base_eval["V"]),
            eta=eta,
        )
        if self.disable_g or (not finite_difference_valid):
            solution = self._solve_no_g(a=float(coeffs["a"]), c=float(coeffs["c"]))
        else:
            solution = self._solve_two_direction_qp(
                a=float(coeffs["a"]),
                b=float(coeffs["b"]),
                c=float(coeffs["c"]),
                h=float(coeffs["h"]),
                k=float(coeffs["k"]),
                det=float(coeffs["H_det"]),
                condition=str(coeffs["q_condition_status"]),
            )

        beta = float(solution["beta"])
        gamma = float(solution["gamma"])
        beta_eff = eta * beta
        gamma_eff = eta * gamma
        update_map: NamedTensorMap = {name: (-beta_eff * f_dir_map[name]) + (gamma_eff * g_dir_map[name]) for name in selected_names}
        update_vec_pre = flatten_named_tensors(update_map, selected_names)
        update_norm_pre = tensor_norm(update_vec_pre)
        update_norm_post = update_norm_pre
        cap_active = 0
        if math.isfinite(self.qp_max_update_norm) and self.qp_max_update_norm > 0.0 and update_norm_pre > self.qp_max_update_norm:
            scale = self.qp_max_update_norm / max(update_norm_pre, self.qp_eps)
            for name in selected_names:
                update_map[name] = update_map[name] * scale
            update_vec_post = flatten_named_tensors(update_map, selected_names)
            update_norm_post = tensor_norm(update_vec_post)
            cap_active = 1
        else:
            update_vec_post = update_vec_pre

        theta_new = apply_state_delta(theta_old, update_map)
        new_eval = self._evaluate_state(eval_closure=eval_closure, theta_state=theta_new, selected_names=selected_names, backward=True)
        restore_named_state(named_params, theta_new)
        diff = named_difference(theta_new, theta_old, selected_names)

        f_contribution_norm = abs(beta_eff) * tensor_norm(f_dir_vec)
        g_contribution_norm = abs(gamma_eff) * tensor_norm(g_dir_vec)
        metrics: Dict[str, object] = {
            "step_index": self._step_index,
            "active_role": self.role,
            "scope": self.optimizer_scope,
            "qp_step_solver": self.qp_step_solver,
            "qp_normalization": self.qp_normalization,
            "eta": eta,
            "finite_difference_valid": int(finite_difference_valid),
            "fd_eps": self.qp_fd_eps,
            "F_raw_norm": tensor_norm(f_raw_vec),
            "F_dir_norm": tensor_norm(f_dir_vec),
            "G_raw_norm": tensor_norm(g_raw_vec),
            "G_dir_norm": tensor_norm(g_dir_vec),
            "cosine_F_G": cosine_similarity(f_raw_vec, g_raw_vec),
            "a": float(coeffs["a"]),
            "b": float(coeffs["b"]),
            "c": float(coeffs["c"]),
            "h": float(coeffs["h"]),
            "k": float(coeffs["k"]),
            "H_det": float(coeffs["H_det"]),
            "ridge_used": float(coeffs["ridge_used"]),
            "q_condition_status": str(coeffs["q_condition_status"]),
            "beta_star_unclipped": float(solution["beta_star_unclipped"]),
            "gamma_star_unclipped": float(solution["gamma_star_unclipped"]),
            "beta": beta,
            "gamma": gamma,
            "beta_eff": beta_eff,
            "gamma_eff": gamma_eff,
            "selected_case": str(solution["selected_case"]),
            "beta_at_bound": int(solution["beta_at_bound"]),
            "gamma_at_bound": int(solution["gamma_at_bound"]),
            "gamma_active": int(solution["gamma_active"]),
            "q_pred": float(solution["q_pred"]),
            "V_before": float(base_eval["V"]),
            "V_after_actual": float(new_eval["V"]),
            "actual_V_change": float(new_eval["V"] - base_eval["V"]),
            "loss_before": float(base_eval["total_loss"]),
            "loss_after_actual": float(new_eval["total_loss"]),
            "actual_loss_change": float(new_eval["total_loss"] - base_eval["total_loss"]),
            "approx_kl_after": float(new_eval["approx_kl"]),
            "clip_fraction_after": float(new_eval["clip_fraction"]),
            "F_contribution_norm": f_contribution_norm,
            "G_contribution_norm": g_contribution_norm,
            "G_over_update_norm": g_contribution_norm / max(update_norm_post, self.qp_eps),
            "update_norm_pre_cap": update_norm_pre,
            "update_norm_post_cap": update_norm_post,
            "cap_active": cap_active,
            "actor_update_norm": block_norm(diff, selected_names, "actor"),
            "logstd_update_norm": block_norm(diff, selected_names, "logstd"),
            "critic_update_norm": block_norm(diff, selected_names, "critic"),
            "zero_update_flag": int(update_norm_post <= self.qp_eps),
            "gamma_active_frac": float(gamma > self.qp_eps),
        }
        self.last_step_metrics = metrics
        self._write_diagnostics_row(metrics)
        self._step_index += 1
        return new_eval["loss_tensor"]


class ProposedNoGNewV2Optimizer(_LyapunovQPBaseV2):
    def __init__(self, params: Iterable[torch.nn.Parameter], **kwargs):
        kwargs = dict(kwargs)
        kwargs["disable_g"] = True
        super().__init__(params, **kwargs)


class ProposedQPNewV2Optimizer(_LyapunovQPBaseV2):
    pass
