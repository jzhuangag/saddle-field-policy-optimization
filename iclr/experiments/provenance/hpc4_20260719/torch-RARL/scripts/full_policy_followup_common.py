from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from models.RARL import RARL
from utils.callbacks import normalized_opponent_policy
from utils.wrappers import AdversarialClassicControlWrapper, AdversarialMujocoWrapper


@dataclass(frozen=True)
class SavedRun:
    method: str
    tag: str
    run_root: pathlib.Path
    model_dir: pathlib.Path
    args_data: Dict
    config_data: Dict


def load_yaml(path: pathlib.Path) -> Dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.UnsafeLoader)


def load_stage5_best_runs(stage5_root: pathlib.Path) -> Dict[str, SavedRun]:
    config_map = json.loads((stage5_root / "final_full_policy_optimizer_configs.json").read_text(encoding="utf-8"))
    results: Dict[str, SavedRun] = {}
    for method, spec in config_map.items():
        tag = spec["tag"]
        phase_root = stage5_root / "final" / method / tag / "saved_models" / "rarl-ppo"
        env_root = next(path for path in phase_root.iterdir() if path.is_dir())
        run_root = next(path for path in env_root.iterdir() if path.is_dir())
        model_dir = run_root / env_root.name
        results[method] = SavedRun(
            method=method,
            tag=tag,
            run_root=run_root,
            model_dir=model_dir,
            args_data=load_yaml(model_dir / "args.yml"),
            config_data=load_yaml(model_dir / "config.yml"),
        )
    return results


def make_eval_vec_env(
    *,
    saved_run: SavedRun,
    adv_impact: str,
    adv_strength: float,
    device: str,
) -> VecNormalize:
    env_id = saved_run.args_data["env"]
    adv_fraction = float(saved_run.config_data.get("adv_fraction", saved_run.args_data.get("adv_fraction", 1.0)))
    adv_index_list = list(saved_run.args_data.get("adv_index_list", ["torso"]))
    adv_force_dim = int(saved_run.args_data.get("adv_force_dim", 2))

    def make_env():
        env = gym.make(env_id)
        if adv_impact == "force":
            wrapped = AdversarialMujocoWrapper(
                env,
                adv_fraction=adv_fraction,
                index_list=adv_index_list,
                force_dim=adv_force_dim,
                device=device,
            )
        elif adv_impact == "control":
            wrapper_kwargs = {
                "adv_fraction": adv_fraction,
                "device": device,
            }
            # Only force-trained checkpoints need the reduced adversary dimension
            # to stay compatible with their force-action head.
            if str(saved_run.args_data.get("adv_impact", "")).lower() == "force":
                wrapper_kwargs["adv_action_dim"] = adv_force_dim
            wrapped = AdversarialClassicControlWrapper(env, **wrapper_kwargs)
        else:
            raise ValueError(f"Unsupported adv_impact {adv_impact!r}")
        wrapped.adv_strength = adv_strength
        return wrapped

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(str(saved_run.model_dir / "vecnormalize.pkl"), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    vec_env.set_attr("adv_strength", float(adv_strength))
    return vec_env


def load_rarl_for_eval(saved_run: SavedRun, *, adv_impact: str, adv_strength: float, device: str) -> tuple[RARL, VecNormalize]:
    vec_env = make_eval_vec_env(saved_run=saved_run, adv_impact=adv_impact, adv_strength=adv_strength, device=device)
    model = RARL.load(str(saved_run.model_dir), env=vec_env, device=device)
    return model, vec_env


def set_rarl_eval_mode(model: RARL, vec_env: VecNormalize, *, operating_mode: Optional[str], adv_strength: float) -> None:
    vec_env.set_attr("adv_strength", float(adv_strength))
    vec_env.set_attr("operating_mode", operating_mode)
    if operating_mode == "protagonist":
        vec_env.set_attr("_adv_policy", normalized_opponent_policy(model.adversary.policy, vec_env))
    elif operating_mode == "adversary":
        vec_env.set_attr("_pro_policy", normalized_opponent_policy(model.protagonist.policy, vec_env))


def rolling_mean(values: Iterable[float], window: int = 3) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return array
    out = np.zeros_like(array)
    for idx in range(array.size):
        left = max(0, idx - window + 1)
        out[idx] = array[left : idx + 1].mean()
    return out
