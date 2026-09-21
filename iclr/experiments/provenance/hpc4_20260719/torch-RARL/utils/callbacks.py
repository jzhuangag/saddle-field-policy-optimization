import csv
import os
import pathlib
import optuna
from typing import Optional
import numpy as np
import torch

from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


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


def _find_vec_normalize(env: VecEnv) -> Optional[VecNormalize]:
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, VecNormalize):
            return current
        visited.add(id(current))
        current = getattr(current, "venv", None)
    return None


class NormalizedOpponentPolicy:
    """Policy view for opponents invoked inside a raw-observation Gym wrapper."""

    def __init__(self, policy, env: VecEnv):
        self.policy = policy
        self.vec_normalize = _find_vec_normalize(env)

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        if self.vec_normalize is None or not self.vec_normalize.norm_obs:
            normalized = observation
        else:
            obs_np = observation.detach().cpu().numpy()
            obs_np = self.vec_normalize.normalize_obs(obs_np)
            normalized = torch.as_tensor(
                np.asarray(obs_np), dtype=observation.dtype, device=observation.device
            )
        return self.policy._predict(normalized, deterministic=deterministic)


def normalized_opponent_policy(policy, env: VecEnv):
    """Return a policy adapter using the observation statistics of ``env``."""
    return NormalizedOpponentPolicy(policy, env)


class ProtagonistEpisodeReturnCallback(BaseCallback):
    """Log only episodes wholly contained in protagonist collection phases."""

    def __init__(self, csv_path: str | pathlib.Path, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = _windows_safe_path(csv_path)

    def _init_callback(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(
                    handle,
                    fieldnames=["protagonist_timesteps", "episode_return", "episode_length"],
                ).writeheader()

    def _on_step(self) -> bool:
        rows = []
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue
            if info.get("episode_start_mode") != "protagonist" or info.get("operating_mode") != "protagonist":
                continue
            rows.append(
                {
                    "protagonist_timesteps": int(self.num_timesteps),
                    "episode_return": float(episode["r"]),
                    "episode_length": int(episode["l"]),
                }
            )
        if rows:
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writerows(rows)
        return True


class JointTransitionRecorderCallback(BaseCallback):
    """Keep protagonist-phase joint-action samples for an outer game critic."""

    def __init__(self, capacity: int = 200_000, verbose: int = 0):
        super().__init__(verbose)
        self.capacity = int(capacity)
        self.rows: list[dict[str, object]] = []
        self._rollout_start = 0
        self.rollout_index = 0

    def _on_rollout_start(self) -> None:
        self._rollout_start = len(self.rows)

    def _on_step(self) -> bool:
        obs_tensor = self.locals.get("obs_tensor")
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [])
        new_obs = self.locals.get("new_obs")
        if any(value is None for value in (obs_tensor, actions, rewards, dones, new_obs)):
            return True
        obs = obs_tensor.detach().cpu().numpy()
        acts = np.asarray(actions)
        next_obs = np.asarray(new_obs)
        for index, info in enumerate(infos):
            if info.get("operating_mode") != "protagonist" or "adversary_action" not in info:
                continue
            self.rows.append(
                {
                    "obs": np.asarray(obs[index], dtype=np.float32).copy(),
                    "u": np.asarray(info.get("protagonist_action", acts[index]), dtype=np.float32).copy(),
                    "w": np.asarray(info["adversary_action"], dtype=np.float32).copy(),
                    "reward": float(np.asarray(rewards)[index]),
                    "next_obs": np.asarray(next_obs[index], dtype=np.float32).copy(),
                    "done": float(np.asarray(dones)[index]),
                    "rollout": self.rollout_index,
                    "mc_return": float("nan"),
                }
            )
        return True

    def _on_rollout_end(self) -> None:
        returns = np.asarray(self.model.rollout_buffer.returns).reshape(-1)
        current = self.rows[self._rollout_start :]
        if len(current) == len(returns):
            for row, value in zip(current, returns):
                row["mc_return"] = float(value)
        self.rollout_index += 1
        if len(self.rows) > self.capacity:
            del self.rows[: len(self.rows) - self.capacity]

    def arrays(self, exclude_latest: bool = False, latest_only: bool = False) -> dict[str, np.ndarray]:
        if not self.rows:
            return {}
        latest = max(int(row["rollout"]) for row in self.rows)
        selected = [
            row for row in self.rows
            if np.isfinite(float(row["mc_return"]))
            and (not exclude_latest or int(row["rollout"]) < latest)
            and (not latest_only or int(row["rollout"]) == latest)
        ]
        if not selected:
            return {}
        return {
            key: np.stack([np.asarray(row[key]) for row in selected])
            for key in ("obs", "u", "w", "reward", "next_obs", "done", "mc_return")
        }


# ================================================
#   Define and customize callback functions
# ================================================


class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.

    Taken from RL Baselines3 Zoo 
    <https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/callbacks.py>

    MIT License

    Copyright (c) 2019 Antonin RAFFIN

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
    ):

        super(TrialEvalCallback, self).__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super(TrialEvalCallback, self)._on_step()
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elasped time ?
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default) only one file will be kept.

    Taken from RL Baselines3 Zoo 
    <https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/callbacks.py>

    MIT License

    Copyright (c) 2019 Antonin RAFFIN

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: Optional[str] = None, verbose: int = 0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True


class CustomRewardCallback(BaseCallback):
    """
    Manipulates reward after each step

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(CustomRewardCallback, self).__init__(verbose)
        

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class SetupProTrainingCallback(BaseCallback):
    """
    Sets up training mode for protagonist

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, policy, verbose=0):
        super(SetupProTrainingCallback, self).__init__(verbose)
        self.policy = policy


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.training_env.set_attr("operating_mode", "protagonist")
        self.training_env.set_attr(
            "_adv_policy", normalized_opponent_policy(self.policy, self.training_env)
        )


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True


class SetupAdvTrainingCallback(BaseCallback):
    """
    Sets up training mode for protagonist

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, policy, verbose=0):
        super(SetupAdvTrainingCallback, self).__init__(verbose)
        self.policy = policy


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.training_env.set_attr("operating_mode", "adversary")
        self.training_env.set_attr(
            "_pro_policy", normalized_opponent_policy(self.policy, self.training_env)
        )


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True


class AdversarialEvalCallback(EvalCallback):
    """
    Evaluate the protagonist under adversarial impact using the current
    opponent policy stored on the callback.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
        policy=None,
    ):
        super(AdversarialEvalCallback, self).__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self.policy = policy

    def _prepare_eval_env(self) -> None:
        if self.policy is None:
            return
        self.eval_env.set_attr("operating_mode", "protagonist")
        self.eval_env.set_attr(
            "_adv_policy", normalized_opponent_policy(self.policy, self.eval_env)
        )

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._prepare_eval_env()
        return super(AdversarialEvalCallback, self)._on_step()


class ParameterNormLoggingCallback(BaseCallback):
    """
    Log grouped parameter norms after each rollout update.
    """

    def __init__(self, csv_path: str, agent_name: str, verbose: int = 0):
        super(ParameterNormLoggingCallback, self).__init__(verbose)
        self.csv_path = csv_path
        self.agent_name = agent_name
        self.rollout_index = 0

    def _init_callback(self) -> None:
        csv_path = _windows_safe_path(self.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not csv_path.exists():
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "agent_name",
                        "rollout_index",
                        "num_timesteps",
                        "actor_param_norm",
                        "critic_param_norm",
                        "log_std_norm",
                        "total_param_norm",
                    ],
                )
                writer.writeheader()

    @staticmethod
    def _group_norm(named_parameters, include_predicate) -> float:
        total = 0.0
        found = False
        for name, parameter in named_parameters:
            if not include_predicate(name):
                continue
            found = True
            total += float(torch.sum(parameter.data.detach() ** 2).item())
        if not found:
            return float("nan")
        return total ** 0.5

    def _write_row(self) -> None:
        named_parameters = list(self.model.policy.named_parameters())
        actor_norm = self._group_norm(
            named_parameters,
            lambda name: name.startswith("mlp_extractor.policy_net") or name.startswith("action_net"),
        )
        critic_norm = self._group_norm(
            named_parameters,
            lambda name: name.startswith("mlp_extractor.value_net") or name.startswith("value_net"),
        )
        log_std_norm = self._group_norm(named_parameters, lambda name: name == "log_std")
        total_norm = self._group_norm(named_parameters, lambda _: True)

        csv_path = _windows_safe_path(self.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "agent_name",
                    "rollout_index",
                    "num_timesteps",
                    "actor_param_norm",
                    "critic_param_norm",
                    "log_std_norm",
                    "total_param_norm",
                ],
            )
            writer.writerow(
                {
                    "agent_name": self.agent_name,
                    "rollout_index": self.rollout_index,
                    "num_timesteps": self.model.num_timesteps,
                    "actor_param_norm": actor_norm,
                    "critic_param_norm": critic_norm,
                    "log_std_norm": log_std_norm,
                    "total_param_norm": total_norm,
                }
            )

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.rollout_index += 1
        self._write_row()
