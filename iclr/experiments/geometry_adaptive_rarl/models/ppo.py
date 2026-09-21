from __future__ import annotations

import csv
import inspect
import os
import pathlib
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.ppo.ppo import PPO as SB3PPO

from models.optimizers import clone_named_state, named_difference, block_norm, restore_named_state


def _windows_safe_path(raw_path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(raw_path)
    if os.name != "nt":
        return path
    path_str = str(path)
    if path_str.startswith("\\\\?\\"):
        return path
    if len(path_str) < 240:
        return path
    resolved = str(path.resolve())
    if resolved.startswith("\\\\"):
        return pathlib.Path("\\\\?\\UNC\\" + resolved.lstrip("\\"))
    return pathlib.Path("\\\\?\\" + resolved)


class PPO(SB3PPO):
    """
    Local PPO subclass that keeps SB3 behavior for Adam/SGD and adds
    closure-based/manual optimizer support for EGM/PPM/proposed-QP variants.
    """

    VALID_OPTIMIZER_SCOPES = {"full_policy", "actor_game", "actor_logstd_only"}

    def __init__(self, *args, optimizer_scope: str = "full_policy", **kwargs):
        self.training_metrics_csv_path = kwargs.pop("training_metrics_csv_path", None)
        self.optimizer_role = kwargs.pop("optimizer_role", "policy")
        if optimizer_scope not in self.VALID_OPTIMIZER_SCOPES:
            raise ValueError(f"Unsupported optimizer_scope={optimizer_scope!r}")
        self.optimizer_scope = optimizer_scope
        self._actor_game_critic_optimizer = None
        super().__init__(*args, **kwargs)
        if self.optimizer_scope != "full_policy":
            self._configure_scoped_optimizers()
        self._ensure_training_metrics_header()

    def _ensure_training_metrics_header(self) -> None:
        if not self.training_metrics_csv_path:
            return
        metrics_path = _windows_safe_path(self.training_metrics_csv_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if metrics_path.exists():
            return
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "optimizer_role",
                    "optimizer_scope",
                    "num_timesteps",
                    "n_updates",
                    "actor_update_norm",
                    "logstd_update_norm",
                    "critic_update_norm",
                    "approx_kl",
                    "clip_fraction",
                    "explained_variance",
                    "value_loss",
                    "policy_gradient_loss",
                    "entropy_loss",
                    "loss",
                    "beta",
                    "gamma",
                    "gamma_active_frac",
                    "G_contribution_norm",
                    "zero_update_flag",
                ],
            )
            writer.writeheader()

    def _append_training_metrics_row(self, row: Dict[str, object]) -> None:
        if not self.training_metrics_csv_path:
            return
        metrics_path = _windows_safe_path(self.training_metrics_csv_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "optimizer_role",
                    "optimizer_scope",
                    "num_timesteps",
                    "n_updates",
                    "actor_update_norm",
                    "logstd_update_norm",
                    "critic_update_norm",
                    "approx_kl",
                    "clip_fraction",
                    "explained_variance",
                    "value_loss",
                    "policy_gradient_loss",
                    "entropy_loss",
                    "loss",
                    "beta",
                    "gamma",
                    "gamma_active_frac",
                    "G_contribution_norm",
                    "zero_update_flag",
                ],
            )
            writer.writerow(row)

    def _actor_named_parameters(self) -> List[Tuple[str, th.nn.Parameter]]:
        return [(name, param) for name, param in self._named_policy_parameters() if "value" not in name]

    def _optimizer_requires_closure(self) -> bool:
        return bool(getattr(self.policy.optimizer, "requires_closure", False))

    def _optimizer_requires_eval_closure(self) -> bool:
        return bool(getattr(self.policy.optimizer, "requires_eval_closure", False))

    def _named_policy_parameters(self) -> List[Tuple[str, th.nn.Parameter]]:
        return [(name, param) for name, param in self.policy.named_parameters() if param.requires_grad]

    def _critic_named_parameters(self) -> List[Tuple[str, th.nn.Parameter]]:
        return [(name, param) for name, param in self._named_policy_parameters() if "value" in name]

    def _actor_logstd_named_parameters(self) -> List[Tuple[str, th.nn.Parameter]]:
        return [(name, param) for name, param in self._named_policy_parameters() if "value" not in name]

    def _extract_optimizer_kwargs(self, optimizer) -> Dict[str, object]:
        signature = inspect.signature(type(optimizer).__init__)
        valid_keys = {
            name
            for name in signature.parameters.keys()
            if name not in {"self", "params"}
        }
        kwargs = {}
        for key, value in optimizer.defaults.items():
            if key in valid_keys:
                kwargs[key] = value
        return kwargs

    def _configure_scoped_optimizers(self) -> None:
        actor_named_params = self._actor_logstd_named_parameters()
        critic_named_params = self._critic_named_parameters()
        actor_params = [param for _, param in actor_named_params]
        critic_params = [param for _, param in critic_named_params]
        optimizer_class = type(self.policy.optimizer)
        optimizer_kwargs = self._extract_optimizer_kwargs(self.policy.optimizer)
        self.policy.optimizer = optimizer_class(actor_params, **optimizer_kwargs)
        if self.optimizer_scope == "actor_game":
            critic_lr = float(optimizer_kwargs.get("lr", self.lr_schedule(1.0)))
            self._actor_game_critic_optimizer = th.optim.Adam(critic_params, lr=critic_lr)
        else:
            self._actor_game_critic_optimizer = None

    def _build_shared_policy_loss(
        self,
        rollout_data,
        actions,
        clip_range: float,
        clip_range_vf: Optional[float],
    ):
        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()
        advantages = rollout_data.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = th.exp(log_prob - rollout_data.old_log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss_unclipped = -policy_loss_1.mean()
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()

        if clip_range_vf is None:
            values_pred = values
        else:
            values_pred = rollout_data.old_values + th.clamp(
                values - rollout_data.old_values, -clip_range_vf, clip_range_vf
            )
        value_loss = F.mse_loss(rollout_data.returns, values_pred)

        if entropy is None:
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        with th.no_grad():
            log_ratio = log_prob - rollout_data.old_log_prob
            approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

        return loss, policy_loss, policy_loss_unclipped, value_loss, entropy_loss, clip_fraction, approx_kl_div

    def _build_eval_closure(self, rollout_data, actions, clip_range: float, clip_range_vf: Optional[float]):
        named_params = self._named_policy_parameters()

        def eval_closure(
            *,
            theta_override: Optional[Dict[str, th.Tensor]] = None,
            backward: bool = True,
            grad_scope_names: Optional[List[str]] = None,
            objective_mode: str = "total_loss",
        ) -> Dict[str, object]:
            if theta_override is not None:
                restore_named_state(named_params, theta_override)
            self.policy.optimizer.zero_grad()
            critic_optimizer = getattr(self, "_actor_game_critic_optimizer", None)
            if critic_optimizer is not None:
                critic_optimizer.zero_grad()
            total_loss, policy_loss, policy_loss_unclipped, value_loss, entropy_loss, clip_fraction, approx_kl = self._build_shared_policy_loss(
                rollout_data,
                actions,
                clip_range,
                clip_range_vf,
            )
            optimizer_cost_mode = str(getattr(self.policy.optimizer, "cost_mode", ""))
            if "unclipped" in optimizer_cost_mode:
                performance_loss = policy_loss_unclipped + self.vf_coef * value_loss + self.ent_coef * entropy_loss
            else:
                performance_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
            if backward:
                backward_tensor = performance_loss if objective_mode == "performance_loss" else total_loss
                backward_tensor.backward()
                if np.isfinite(self.max_grad_norm):
                    if grad_scope_names is None:
                        grad_params = list(self.policy.parameters())
                    else:
                        scope_name_set = set(grad_scope_names)
                        grad_params = [param for name, param in named_params if name in scope_name_set]
                    th.nn.utils.clip_grad_norm_(grad_params, self.max_grad_norm)
            grads = {
                name: (param.grad.detach().clone() if param.grad is not None else th.zeros_like(param.data))
                for name, param in named_params
            }
            return {
                "loss_tensor": total_loss.detach(),
                "total_loss": float(total_loss.item()),
                "policy_loss": float(policy_loss.item()),
                "policy_loss_unclipped": float(policy_loss_unclipped.item()),
                "value_loss": float(value_loss.item()),
                "entropy_loss": float(entropy_loss.item()),
                "performance_loss": float(performance_loss.item()),
                "clip_fraction": float(clip_fraction),
                "approx_kl": float(approx_kl),
                "grads": grads,
            }

        return eval_closure

    def _infer_env_id(self) -> Optional[str]:
        env = getattr(self, "env", None)
        if env is None:
            return None
        current = env
        for attr in ("envs",):
            if hasattr(current, attr):
                envs = getattr(current, attr)
                if envs:
                    current = envs[0]
                    break
        while hasattr(current, "env"):
            current = current.env
        spec = getattr(current, "spec", None)
        return getattr(spec, "id", None)

    def _short_clean_return_cost(
        self,
        *,
        theta_state: Dict[str, th.Tensor],
        named_params,
        base_seed: int,
        horizon: int,
        episodes: int,
    ) -> float:
        env_id = self._infer_env_id()
        if not env_id:
            return float("nan")
        total_returns: List[float] = []
        env = gym.make(env_id)
        try:
            restore_named_state(named_params, theta_state)
            for ep in range(max(int(episodes), 1)):
                obs, _ = env.reset(seed=int(base_seed + ep))
                ep_return = 0.0
                for _ in range(max(int(horizon), 1)):
                    action, _ = self.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_return += float(reward)
                    if bool(terminated or truncated):
                        break
                total_returns.append(ep_return)
        finally:
            env.close()
        return float(-np.mean(total_returns)) if total_returns else float("nan")

    def _short_rarl_return_cost(
        self,
        *,
        theta_state: Dict[str, th.Tensor],
        named_params,
        base_seed: int,
        horizon: int,
        episodes: int,
    ) -> float:
        env_id = self._infer_env_id()
        env = getattr(self, "env", None)
        if not env_id or env is None or not hasattr(env, "get_attr"):
            return float("nan")
        try:
            adv_policies = env.get_attr("_adv_policy")
        except Exception:
            adv_policies = None
        adv_policy = adv_policies[0] if adv_policies else None
        if adv_policy is None:
            return float("nan")

        from utils.wrappers import AdversarialClassicControlWrapper

        total_returns: List[float] = []
        wrapped_env = AdversarialClassicControlWrapper(
            gym.make(env_id),
            adv_fraction=2.5,
            device=str(self.device),
        )
        wrapped_env.operating_mode = "protagonist"
        wrapped_env._adv_policy = adv_policy
        wrapped_env.adv_strength = 1.0
        try:
            restore_named_state(named_params, theta_state)
            for ep in range(max(int(episodes), 1)):
                obs, _ = wrapped_env.reset(seed=int(base_seed + ep))
                ep_return = 0.0
                for _ in range(max(int(horizon), 1)):
                    action, _ = self.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = wrapped_env.step(action)
                    ep_return += float(reward)
                    if bool(terminated or truncated):
                        break
                total_returns.append(ep_return)
        finally:
            wrapped_env.close()
        return float(-np.mean(total_returns)) if total_returns else float("nan")

    def _build_candidate_merit_evaluator(
        self,
        *,
        eval_closure,
        named_params,
        rollout_data,
    ) -> Optional[Callable[[Dict[str, th.Tensor], Sequence[str]], Dict[str, float]]]:
        optimizer = self.policy.optimizer
        cost_mode = str(getattr(optimizer, "cost_mode", ""))
        if cost_mode not in {
            "mixed_clean_unclipped_actor_surrogate_cost",
            "mixed_clean_actor_surrogate_cost",
            "mixed_rarl_unclipped_actor_surrogate_cost",
        }:
            return None
        if getattr(self, "optimizer_role", "policy") != "protagonist":
            return None
        horizon = int(getattr(optimizer, "short_return_horizon", 16))
        episodes = int(getattr(optimizer, "short_return_episodes", 1))
        seed_offset = int(getattr(optimizer, "short_return_seed_offset", 0))
        step_seed = int(self.num_timesteps + self._n_updates + seed_offset)

        def merit_eval(theta_state: Dict[str, th.Tensor], selected_names: Sequence[str]) -> Dict[str, float]:
            info = eval_closure(theta_override=theta_state, backward=False, grad_scope_names=list(selected_names))
            if "unclipped" in cost_mode:
                actor_cost = float(info["policy_loss_unclipped"])
            else:
                actor_cost = float(info["policy_loss"])
            value_loss = float(info["value_loss"])
            entropy_loss = float(info["entropy_loss"])
            short_clean_return_cost = self._short_clean_return_cost(
                theta_state=theta_state,
                named_params=named_params,
                base_seed=step_seed,
                horizon=horizon,
                episodes=episodes,
            )
            if cost_mode == "mixed_rarl_unclipped_actor_surrogate_cost":
                short_rarl_return_cost = self._short_rarl_return_cost(
                    theta_state=theta_state,
                    named_params=named_params,
                    base_seed=step_seed,
                    horizon=horizon,
                    episodes=episodes,
                )
                mixed_return_cost = short_rarl_return_cost
            else:
                short_rarl_return_cost = float("nan")
                mixed_return_cost = short_clean_return_cost
            mixed_merit = mixed_return_cost + actor_cost + float(self.vf_coef) * value_loss + float(self.ent_coef) * entropy_loss
            return {
                "mixed_merit": float(mixed_merit),
                "actor_cost": float(actor_cost),
                "short_clean_return_cost": float(short_clean_return_cost),
                "short_rarl_return_cost": float(short_rarl_return_cost),
                "value_loss": float(value_loss),
                "entropy_loss": float(entropy_loss),
            }

        return merit_eval

    def _apply_actor_game_critic_step(self, eval_closure) -> None:
        critic_named_params = self._critic_named_parameters()
        if not critic_named_params:
            return 0.0
        critic_params = [param for _, param in critic_named_params]
        critic_optimizer = getattr(self, "_actor_game_critic_optimizer", None)
        if critic_optimizer is None:
            raise RuntimeError("actor_game critic optimizer is not initialized")
        critic_before = clone_named_state(critic_named_params)

        critic_optimizer.zero_grad()
        eval_closure(backward=True)
        if np.isfinite(self.max_grad_norm):
            th.nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm)
        critic_optimizer.step()
        critic_after = clone_named_state(critic_named_params)
        critic_diff = named_difference(critic_after, critic_before)
        return block_norm(critic_diff, list(critic_diff.keys()), "critic")

    def set_parameters(self, load_path_or_dict, exact_match: bool = True, device: str | th.device = "auto") -> None:
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device, load_data=False)

        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            try:
                attr = recursive_getattr(self, name)
            except Exception as exc:
                raise ValueError(f"Key {name} is an invalid object name.") from exc

            if isinstance(attr, th.optim.Optimizer):
                try:
                    attr.load_state_dict(params[name])
                except ValueError as exc:
                    if "parameter group" not in str(exc):
                        raise
                    warnings.warn(
                        f"Skipping optimizer state for {name!r} due to parameter-group mismatch during load: {exc}",
                        RuntimeWarning,
                    )
                updated_objects.add(name)
                continue

            attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        optimizer_metric_history: Dict[str, List[float]] = {}
        actor_update_norms: List[float] = []
        logstd_update_norms: List[float] = []
        critic_update_norms: List[float] = []

        continue_training = True
        last_loss = None

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                loss, policy_loss, policy_loss_unclipped, value_loss, entropy_loss, clip_fraction, approx_kl_div = self._build_shared_policy_loss(
                    rollout_data,
                    actions,
                    clip_range,
                    clip_range_vf,
                )

                pg_losses.append(policy_loss.item())
                clip_fractions.append(clip_fraction)
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                named_params = self._named_policy_parameters()
                theta_before = clone_named_state(named_params)
                eval_closure = self._build_eval_closure(rollout_data, actions, clip_range, clip_range_vf)
                candidate_merit_evaluator = self._build_candidate_merit_evaluator(
                    eval_closure=eval_closure,
                    named_params=named_params,
                    rollout_data=rollout_data,
                )

                if self._optimizer_requires_eval_closure():
                    step_kwargs = {
                        "eval_closure": eval_closure,
                        "named_params": named_params,
                    }
                    step_parameters = inspect.signature(self.policy.optimizer.step).parameters
                    if "candidate_merit_evaluator" in step_parameters:
                        step_kwargs["candidate_merit_evaluator"] = candidate_merit_evaluator
                    step_loss = self.policy.optimizer.step(**step_kwargs)
                    critic_update_norm = 0.0
                    if getattr(self.policy.optimizer, "optimizer_scope", "full_policy") == "actor_game":
                        critic_update_norm = self._apply_actor_game_critic_step(eval_closure)
                    last_loss = step_loss if step_loss is not None else loss.detach()
                    metrics = getattr(self.policy.optimizer, "last_step_metrics", {})
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
                            optimizer_metric_history.setdefault(key, []).append(float(value))
                    theta_after = clone_named_state(named_params)
                    diff = named_difference(theta_after, theta_before)
                    selected_names = list(diff.keys())
                    actor_update_norms.append(block_norm(diff, selected_names, "actor"))
                    logstd_update_norms.append(block_norm(diff, selected_names, "logstd"))
                    if getattr(self.policy.optimizer, "optimizer_scope", "full_policy") == "actor_game":
                        critic_update_norms.append(float(critic_update_norm))
                    else:
                        critic_update_norms.append(block_norm(diff, selected_names, "critic"))
                elif self._optimizer_requires_closure():
                    def closure():
                        return eval_closure(backward=True)["loss_tensor"]

                    self.policy.optimizer.step(closure)
                    critic_update_norm = 0.0
                    if self.optimizer_scope == "actor_game":
                        critic_update_norm = self._apply_actor_game_critic_step(eval_closure)
                    last_loss = loss.detach()
                    theta_after = clone_named_state(named_params)
                    diff = named_difference(theta_after, theta_before)
                    selected_names = list(diff.keys())
                    actor_update_norms.append(block_norm(diff, selected_names, "actor"))
                    logstd_update_norms.append(block_norm(diff, selected_names, "logstd"))
                    if self.optimizer_scope == "actor_game":
                        critic_update_norms.append(float(critic_update_norm))
                    else:
                        critic_update_norms.append(block_norm(diff, selected_names, "critic"))
                else:
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    if np.isfinite(self.max_grad_norm):
                        th.nn.utils.clip_grad_norm_(self.policy.optimizer.param_groups[0]["params"], self.max_grad_norm)
                    self.policy.optimizer.step()
                    critic_update_norm = 0.0
                    if self.optimizer_scope == "actor_game":
                        critic_update_norm = self._apply_actor_game_critic_step(eval_closure)
                    last_loss = loss.detach()
                    theta_after = clone_named_state(named_params)
                    diff = named_difference(theta_after, theta_before)
                    selected_names = list(diff.keys())
                    actor_update_norms.append(block_norm(diff, selected_names, "actor"))
                    logstd_update_norms.append(block_norm(diff, selected_names, "logstd"))
                    if self.optimizer_scope == "actor_game":
                        critic_update_norms.append(float(critic_update_norm))
                    else:
                        critic_update_norms.append(block_norm(diff, selected_names, "critic"))

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        if last_loss is not None:
            self.logger.record("train/loss", float(last_loss.item()))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        for key, values in optimizer_metric_history.items():
            if values:
                self.logger.record(f"train/{key}", float(np.mean(values)))
        if actor_update_norms:
            self.logger.record("train/actor_update_norm", float(np.mean(actor_update_norms)))
        if logstd_update_norms:
            self.logger.record("train/logstd_update_norm", float(np.mean(logstd_update_norms)))
        if critic_update_norms:
            self.logger.record("train/critic_update_norm", float(np.mean(critic_update_norms)))

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self._append_training_metrics_row(
            {
                "optimizer_role": self.optimizer_role,
                "optimizer_scope": self.optimizer_scope,
                "num_timesteps": self.num_timesteps,
                "n_updates": self._n_updates,
                "actor_update_norm": float(np.mean(actor_update_norms)) if actor_update_norms else float("nan"),
                "logstd_update_norm": float(np.mean(logstd_update_norms)) if logstd_update_norms else float("nan"),
                "critic_update_norm": float(np.mean(critic_update_norms)) if critic_update_norms else float("nan"),
                "approx_kl": float(np.mean(approx_kl_divs)) if approx_kl_divs else float("nan"),
                "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else float("nan"),
                "explained_variance": float(explained_var),
                "value_loss": float(np.mean(value_losses)) if value_losses else float("nan"),
                "policy_gradient_loss": float(np.mean(pg_losses)) if pg_losses else float("nan"),
                "entropy_loss": float(np.mean(entropy_losses)) if entropy_losses else float("nan"),
                "loss": float(last_loss.item()) if last_loss is not None else float("nan"),
                "beta": float(np.mean(optimizer_metric_history["beta"])) if "beta" in optimizer_metric_history else float("nan"),
                "gamma": float(np.mean(optimizer_metric_history["gamma"])) if "gamma" in optimizer_metric_history else float("nan"),
                "gamma_active_frac": float(np.mean(optimizer_metric_history["gamma_active_frac"])) if "gamma_active_frac" in optimizer_metric_history else float("nan"),
                "G_contribution_norm": float(np.mean(optimizer_metric_history["G_contribution_norm"])) if "G_contribution_norm" in optimizer_metric_history else float("nan"),
                "zero_update_flag": float(np.mean(optimizer_metric_history["zero_update_flag"])) if "zero_update_flag" in optimizer_metric_history else float("nan"),
            }
        )
