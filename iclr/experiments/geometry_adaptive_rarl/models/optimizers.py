from __future__ import annotations

import csv
import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import torch
from torch.optim import Optimizer

from models.proposed_qp_new import (
    ProposedNoGNewOptimizer,
    ProposedNoGNewV2Optimizer,
    ProposedQPNewOptimizer,
    ProposedQPNewV2Optimizer,
)
from models.proposed_qp_perflyap import ProposedNoGPerfLyapOptimizer, ProposedQPPerfLyapOptimizer
from models.proposed_qp_rawfg import ProposedNoGRawFGOptimizer, ProposedQPRawFGOptimizer

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


def zero_like_state(named_params: NamedParams) -> NamedTensorMap:
    return {name: torch.zeros_like(param.data) for name, param in named_params}


def named_difference(new_state: NamedTensorMap, old_state: NamedTensorMap, selected_names: Optional[Sequence[str]] = None) -> NamedTensorMap:
    selected = set(selected_names) if selected_names is not None else None
    diff: NamedTensorMap = {}
    for name, tensor in new_state.items():
        if selected is not None and name not in selected:
            continue
        diff[name] = tensor - old_state[name]
    return diff


def flatten_named_tensors(named_tensor_map: NamedTensorMap, selected_names: Optional[Sequence[str]] = None) -> torch.Tensor:
    selected = set(selected_names) if selected_names is not None else None
    tensors = [tensor.reshape(-1) for name, tensor in named_tensor_map.items() if selected is None or name in selected]
    if not tensors:
        return torch.zeros(0)
    return torch.cat(tensors)


def tensor_norm(tensor: torch.Tensor) -> float:
    return float(torch.norm(tensor).item()) if tensor.numel() > 0 else 0.0


def named_tensor_norm(named_tensor_map: NamedTensorMap, selected_names: Optional[Sequence[str]] = None) -> float:
    return tensor_norm(flatten_named_tensors(named_tensor_map, selected_names))


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    denom = torch.norm(a) * torch.norm(b)
    if denom.item() == 0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


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


class EGM(Optimizer):
    """
    Extrapolation Gradient Method using a closure for the second gradient evaluation.
    """

    requires_closure = True

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3, weight_decay: float = 0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        if closure is None:
            raise ValueError("EGM requires a closure that reevaluates the loss and gradients on the same minibatch.")

        loss = closure()

        param_state = []
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach().clone()
                original = param.data.detach().clone()
                if weight_decay != 0.0:
                    grad = grad + weight_decay * original
                param_state.append((param, original, grad, lr))

        with torch.no_grad():
            for param, original, grad, lr in param_state:
                param.data.copy_(original - lr * grad)

        loss_half = closure()

        with torch.no_grad():
            for group in self.param_groups:
                weight_decay = group["weight_decay"]
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    original_entry = next((entry for entry in param_state if entry[0] is param), None)
                    if original_entry is None:
                        continue
                    _, original, _, lr = original_entry
                    grad = param.grad.detach().clone()
                    if weight_decay != 0.0:
                        grad = grad + weight_decay * param.data.detach()
                    param.data.copy_(original - lr * grad)

        return loss_half if loss_half is not None else loss


class PPM(Optimizer):
    """
    Simple proximal-point / fixed-point optimizer:
        z^(m+1) = z_old - lr * F(z^(m))
    """

    requires_closure = True

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        inner_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if inner_steps < 1:
            raise ValueError(f"inner_steps must be >= 1, got {inner_steps}")
        defaults = dict(lr=lr, inner_steps=inner_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        if closure is None:
            raise ValueError("PPM requires a closure that reevaluates the loss and gradients on the same minibatch.")

        originals: Dict[int, torch.Tensor] = {}
        current_points: Dict[int, torch.Tensor] = {}
        last_loss = None

        for group in self.param_groups:
            for param in group["params"]:
                originals[id(param)] = param.data.detach().clone()
                current_points[id(param)] = param.data.detach().clone()

        max_inner_steps = max(group["inner_steps"] for group in self.param_groups)

        for _ in range(max_inner_steps):
            with torch.no_grad():
                for group in self.param_groups:
                    for param in group["params"]:
                        param.data.copy_(current_points[id(param)])

            last_loss = closure()

            with torch.no_grad():
                for group in self.param_groups:
                    lr = group["lr"]
                    weight_decay = group["weight_decay"]
                    inner_steps = group["inner_steps"]
                    for param in group["params"]:
                        if param.grad is None:
                            continue
                        grad = param.grad.detach().clone()
                        current = current_points[id(param)]
                        if weight_decay != 0.0:
                            grad = grad + weight_decay * current
                        if inner_steps >= 1:
                            current_points[id(param)] = originals[id(param)] - lr * grad

        with torch.no_grad():
            for group in self.param_groups:
                for param in group["params"]:
                    param.data.copy_(current_points[id(param)])

        return last_loss


class ProposedTwoDirectionQP(Optimizer):
    """
    Closure-based adaptive two-direction update:
        delta = - beta * F + gamma * G

    with
        F = grad(total PPO loss)
        G = (F(z) - F(z - alpha * F(z))) / alpha

    The implementation is intentionally stateless: it never uses Adam moments or any
    optimizer.state entries.
    """

    requires_eval_closure = True

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        optimizer_scope: str = "full_policy",
        qp_normalization: str = "none",
        qp_g_alpha: float = 1e-3,
        beta_max: float = float("inf"),
        gamma_max: float = float("inf"),
        gamma_scale: float = 1.0,
        max_update_norm: float = float("inf"),
        qp_eps: float = 1e-8,
        objective: str = "loss_decrease",
        diagnostics_csv_path: Optional[str] = None,
        role: str = "policy",
        disable_g: bool = False,
    ):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        if optimizer_scope not in {"full_policy", "actor_game", "actor_logstd_only"}:
            raise ValueError(f"Unsupported optimizer_scope={optimizer_scope!r}")
        if qp_normalization not in {"none", "global", "block"}:
            raise ValueError(f"Unsupported qp_normalization={qp_normalization!r}")
        self.optimizer_scope = optimizer_scope
        self.qp_normalization = qp_normalization
        self.qp_g_alpha = max(float(qp_g_alpha), 1e-12)
        self.beta_max = float(beta_max)
        self.gamma_max = float(gamma_max)
        self.gamma_scale = float(gamma_scale)
        self.max_update_norm = float(max_update_norm)
        self.qp_eps = max(float(qp_eps), 1e-12)
        if objective not in {"loss_decrease", "normalized_loss_decrease"}:
            raise ValueError(f"Unsupported objective={objective!r}")
        self.objective = objective
        self.diagnostics_csv_path = diagnostics_csv_path
        self.role = role
        self.disable_g = bool(disable_g)
        self.g_convention = "G=(F(z)-F(z-alpha*F(z)))/alpha"
        self.last_step_metrics: Dict[str, float | int | str] = {}
        self._step_index = 0
        self._fieldnames = [
            "step_index",
            "role",
            "optimizer_scope",
            "qp_normalization",
            "G_convention",
            "F_norm_raw",
            "G_norm_raw",
            "F_norm_used",
            "G_norm_used",
            "cosine_F_G",
            "beta",
            "gamma",
            "beta_active",
            "gamma_active",
            "gamma_active_frac",
            "G_contribution_norm",
            "G_over_F_norm",
            "G_over_update_norm",
            "update_norm_pre_cap",
            "update_norm_post_cap",
            "boundary_solution_flag",
            "interior_solution_flag",
            "zero_update_flag",
            "cap_active_flag",
            "solver_status",
            "predicted_decrease",
            "same_minibatch_total_loss_change",
            "policy_loss_change",
            "value_loss_change",
            "entropy_change",
            "beta_boundary",
            "gamma_boundary",
            "objective_mode",
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

    def _normalize_direction(self, direction: NamedTensorMap, selected_names: Sequence[str]) -> NamedTensorMap:
        if self.qp_normalization == "none":
            return {name: direction[name].clone() for name in selected_names}

        if self.qp_normalization == "global":
            norm = max(named_tensor_norm(direction, selected_names), self.qp_eps)
            return {name: direction[name] / norm for name in selected_names}

        normalized: NamedTensorMap = {}
        block_norms = {
            block: max(block_norm(direction, selected_names, block), self.qp_eps)
            for block in ("actor", "logstd", "critic")
        }
        for name in selected_names:
            normalized[name] = direction[name] / block_norms[classify_parameter_block(name)]
        return normalized

    def _surrogate_objective(
        self,
        beta: float,
        gamma: float,
        dot_ff: float,
        dot_fg: float,
        dot_gg: float,
        grad_fu: float,
        grad_gu: float,
        objective_scale: float,
    ) -> float:
        return (
            objective_scale
            * (
                -beta * grad_fu
                + gamma * grad_gu
                + 0.5 * beta * beta * dot_ff
                - beta * gamma * dot_fg
                + 0.5 * gamma * gamma * dot_gg
            )
            + 0.5 * self.qp_eps * (beta * beta + gamma * gamma)
        )

    def _solve_coefficients(self, f_raw: torch.Tensor, f_used: torch.Tensor, g_used: torch.Tensor) -> Tuple[float, float, int, int, int, int, str]:
        grad_fu = float(torch.dot(f_raw, f_used).item()) if f_used.numel() > 0 else 0.0
        grad_gu = float(torch.dot(f_raw, g_used).item()) if g_used.numel() > 0 else 0.0
        dot_ff = float(torch.dot(f_used, f_used).item()) if f_used.numel() > 0 else 0.0
        dot_gg = float(torch.dot(g_used, g_used).item()) if g_used.numel() > 0 else 0.0
        dot_fg = float(torch.dot(f_used, g_used).item()) if f_used.numel() > 0 and g_used.numel() > 0 else 0.0
        objective_scale = 1.0
        if self.objective == "normalized_loss_decrease":
            objective_scale = 1.0 / max(float(torch.norm(f_raw).item()) if f_raw.numel() > 0 else 0.0, self.qp_eps)

        candidates: List[Tuple[str, float, float]] = [("zero", 0.0, 0.0)]
        beta_only = max(grad_fu / max(dot_ff + self.qp_eps, self.qp_eps), 0.0)
        if math.isfinite(self.beta_max):
            beta_only = min(beta_only, self.beta_max)
        candidates.append(("beta_only", beta_only, 0.0))

        if not self.disable_g and g_used.numel() > 0:
            gamma_only = max(-grad_gu / max(dot_gg + self.qp_eps, self.qp_eps), 0.0) * self.gamma_scale
            if math.isfinite(self.gamma_max):
                gamma_only = min(gamma_only, self.gamma_max)
            candidates.append(("gamma_only", 0.0, gamma_only))
            matrix = torch.tensor(
                [
                    [dot_ff + self.qp_eps, -dot_fg],
                    [-dot_fg, dot_gg + self.qp_eps],
                ],
                dtype=f_used.dtype if f_used.numel() > 0 else torch.float32,
            )
            rhs = torch.tensor([grad_fu, -grad_gu], dtype=matrix.dtype)
            determinant = float(torch.det(matrix).item())
            if math.isfinite(determinant) and abs(determinant) > self.qp_eps:
                solution = torch.linalg.solve(matrix, rhs)
                beta_int = float(solution[0].item())
                gamma_int = float(solution[1].item()) * self.gamma_scale
                if math.isfinite(self.beta_max):
                    beta_int = min(beta_int, self.beta_max)
                if math.isfinite(self.gamma_max):
                    gamma_int = min(gamma_int, self.gamma_max)
                if beta_int >= 0.0 and gamma_int >= 0.0 and math.isfinite(beta_int) and math.isfinite(gamma_int):
                    candidates.append(("interior", beta_int, gamma_int))

        best_status = "zero"
        best_beta = 0.0
        best_gamma = 0.0
        best_objective = float("inf")
        for status, beta, gamma in candidates:
            objective = self._surrogate_objective(beta, gamma, dot_ff, dot_fg, dot_gg, grad_fu, grad_gu, objective_scale)
            if objective < best_objective:
                best_objective = objective
                best_status = status
                best_beta = beta
                best_gamma = gamma

        interior_flag = int(best_status == "interior")
        boundary_flag = int(best_status not in {"interior", "zero"})
        beta_boundary = int(math.isfinite(self.beta_max) and abs(best_beta - self.beta_max) <= max(self.qp_eps, 1e-12))
        gamma_boundary = int(math.isfinite(self.gamma_max) and abs(best_gamma - self.gamma_max) <= max(self.qp_eps, 1e-12))
        return best_beta, best_gamma, boundary_flag, interior_flag, beta_boundary, gamma_boundary, best_status

    def step(  # type: ignore[override]
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
        *,
        eval_closure: Optional[Callable[..., Dict[str, object]]] = None,
        named_params: Optional[NamedParams] = None,
    ):
        del closure
        if eval_closure is None or named_params is None:
            raise ValueError("ProposedTwoDirectionQP requires eval_closure and named_params.")

        selected_names = self._selected_names(named_params)
        theta_old = clone_named_state(named_params)
        zero_state = zero_like_state(named_params)

        old_eval = eval_closure(theta_override=theta_old, backward=True)
        grads_all = old_eval["grads"]
        f_raw_map = {name: grads_all[name].detach().clone() for name in selected_names}
        f_raw_vec = flatten_named_tensors(f_raw_map, selected_names)

        probe_state = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            probe_state[name] = theta_old[name] - self.qp_g_alpha * f_raw_map[name]
        probe_eval = eval_closure(theta_override=probe_state, backward=True)
        if self.disable_g:
            g_raw_map = {name: zero_state[name].clone() for name in selected_names}
        else:
            g_raw_map = {
                name: (f_raw_map[name] - probe_eval["grads"][name].detach().clone()) / self.qp_g_alpha
                for name in selected_names
            }

        f_used_map = self._normalize_direction(f_raw_map, selected_names)
        g_used_map = self._normalize_direction(g_raw_map, selected_names)
        f_used_vec = flatten_named_tensors(f_used_map, selected_names)
        g_used_vec = flatten_named_tensors(g_used_map, selected_names)

        beta, gamma, boundary_flag, interior_flag, beta_boundary, gamma_boundary, solver_status = self._solve_coefficients(f_raw_vec, f_used_vec, g_used_vec)
        if self.disable_g:
            gamma = 0.0
            boundary_flag = int(beta > 0.0)
            interior_flag = 0
            gamma_boundary = 0
            solver_status = "beta_only_noG"

        step_scale = float(self.param_groups[0]["lr"])
        update_map: NamedTensorMap = {}
        for name in selected_names:
            update_map[name] = step_scale * (-beta * f_used_map[name] + gamma * g_used_map[name])

        update_vec_pre = flatten_named_tensors(update_map, selected_names)
        update_norm_pre = tensor_norm(update_vec_pre)
        update_norm_post = update_norm_pre
        cap_active_flag = 0
        if math.isfinite(self.max_update_norm) and update_norm_pre > self.max_update_norm > 0.0:
            scale = self.max_update_norm / max(update_norm_pre, self.qp_eps)
            for name in selected_names:
                update_map[name] = update_map[name] * scale
            update_vec_post = flatten_named_tensors(update_map, selected_names)
            update_norm_post = tensor_norm(update_vec_post)
            cap_active_flag = 1
        else:
            update_vec_post = update_vec_pre

        theta_new = apply_state_delta(theta_old, update_map)
        new_eval = eval_closure(theta_override=theta_new, backward=False)
        restore_named_state(named_params, theta_new)

        predicted_decrease = -(
            float(torch.dot(f_raw_vec, update_vec_post).item()) + 0.5 * float(torch.dot(update_vec_post, update_vec_post).item())
        ) if update_vec_post.numel() > 0 else 0.0
        g_contribution_norm = tensor_norm(gamma * g_used_vec) if g_used_vec.numel() > 0 else 0.0

        metrics: Dict[str, object] = {
            "step_index": self._step_index,
            "role": self.role,
            "optimizer_scope": self.optimizer_scope,
            "qp_normalization": self.qp_normalization,
            "G_convention": self.g_convention,
            "F_norm_raw": tensor_norm(f_raw_vec),
            "G_norm_raw": tensor_norm(flatten_named_tensors(g_raw_map, selected_names)),
            "F_norm_used": tensor_norm(f_used_vec),
            "G_norm_used": tensor_norm(g_used_vec),
            "cosine_F_G": cosine_similarity(f_raw_vec, flatten_named_tensors(g_raw_map, selected_names)),
            "beta": beta,
            "gamma": gamma,
            "beta_active": int(beta > self.qp_eps),
            "gamma_active": int(gamma > self.qp_eps),
            "gamma_active_frac": float(gamma > self.qp_eps),
            "G_contribution_norm": g_contribution_norm,
            "G_over_F_norm": g_contribution_norm / max(tensor_norm(f_raw_vec), self.qp_eps),
            "G_over_update_norm": g_contribution_norm / max(update_norm_post, self.qp_eps),
            "update_norm_pre_cap": update_norm_pre,
            "update_norm_post_cap": update_norm_post,
            "boundary_solution_flag": boundary_flag,
            "interior_solution_flag": interior_flag,
            "zero_update_flag": int(update_norm_post <= self.qp_eps),
            "cap_active_flag": cap_active_flag,
            "solver_status": solver_status,
            "predicted_decrease": predicted_decrease,
            "same_minibatch_total_loss_change": float(new_eval["total_loss"]) - float(old_eval["total_loss"]),
            "policy_loss_change": float(new_eval["policy_loss"]) - float(old_eval["policy_loss"]),
            "value_loss_change": float(new_eval["value_loss"]) - float(old_eval["value_loss"]),
            "entropy_change": float(new_eval["entropy_loss"]) - float(old_eval["entropy_loss"]),
            "beta_boundary": beta_boundary,
            "gamma_boundary": gamma_boundary,
            "objective_mode": self.objective,
        }
        self.last_step_metrics = metrics
        self._write_diagnostics_row(metrics)
        self._step_index += 1
        return new_eval["loss_tensor"]


class ProposedNoGOptimizer(ProposedTwoDirectionQP):
    def __init__(self, params: Iterable[torch.nn.Parameter], **kwargs):
        kwargs = dict(kwargs)
        kwargs["disable_g"] = True
        super().__init__(params, **kwargs)


OPTIMIZER_REGISTRY: Dict[str, Type[Optimizer]] = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "egm": EGM,
    "ppm": PPM,
    "proposed_qp": ProposedTwoDirectionQP,
    "proposed_nog": ProposedNoGOptimizer,
    "proposed_qp_new": ProposedQPNewOptimizer,
    "proposed_nog_new": ProposedNoGNewOptimizer,
    "proposed_qp_new_v2": ProposedQPNewV2Optimizer,
    "proposed_nog_new_v2": ProposedNoGNewV2Optimizer,
    "proposed_qp_rawfg": ProposedQPRawFGOptimizer,
    "proposed_nog_rawfg": ProposedNoGRawFGOptimizer,
    "proposed_qp_perflyap": ProposedQPPerfLyapOptimizer,
    "proposed_nog_perflyap": ProposedNoGPerfLyapOptimizer,
}


def get_optimizer_class(name: str) -> Type[Optimizer]:
    key = name.lower()
    if key == "proposed_noG".lower():
        key = "proposed_nog"
    if key == "proposed_noG_new".lower():
        key = "proposed_nog_new"
    if key == "proposed_noG_new_v2".lower():
        key = "proposed_nog_new_v2"
    if key == "proposed_noG_rawFG".lower():
        key = "proposed_nog_rawfg"
    if key == "proposed_noG_perfLyap".lower():
        key = "proposed_nog_perflyap"
    try:
        return OPTIMIZER_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"Unknown optimizer {name!r}, expected one of {sorted(OPTIMIZER_REGISTRY.keys())}") from exc
