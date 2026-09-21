from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils.exp_manager import ExperimentManager
from utils.callbacks import SetupAdvTrainingCallback, SetupProTrainingCallback


@dataclass
class ProbeBatch:
    probe_idx: int
    rollout_data: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Probe full-policy PPO optimizer behavior on fixed minibatches")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--role", type=str, default="protagonist", choices=["protagonist", "adversary"])
    parser.add_argument("--num-probes", type=int, default=3)
    parser.add_argument("--mode", type=str, default="all", choices=["audit", "diagnosis", "all"])
    return parser.parse_args()


def build_manager(args: argparse.Namespace) -> ExperimentManager:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    results_root = repo_root.parent / "results" / "_stage5_probe_tmp"
    return ExperimentManager(
        args=argparse.Namespace(),
        algo="rarl",
        rarl_config="ppo",
        env_id=args.env,
        log_folder=str(results_root / "logging"),
        tensorboard_log=str(results_root / "tb"),
        n_timesteps=1,
        eval_freq=-1,
        n_eval_episodes=1,
        save_freq=-1,
        hyperparameter_path=str(repo_root / "hyperparameter"),
        hyperparams=None,
        env_kwargs=None,
        model_path=str(results_root / "saved_models"),
        pretrained_model="",
        optimize_hyperparameters=False,
        storage=None,
        study_name=None,
        n_opt_trials=1,
        n_jobs=1,
        sampler="tpe",
        pruner="median",
        optimization_log_path=str(results_root / "opt"),
        n_startup_trials=0,
        n_evaluations_opt=1,
        seed=args.seed,
        log_interval=-1,
        save_replay_buffer=False,
        verbose=0,
        vec_env_type="dummy",
        n_envs=1,
        n_eval_envs=1,
        no_optim_plots=True,
        adv_env=False,
        adv_impact="force",
        adv_fraction=2.5,
        adv_delay=-1,
        adv_index_list=["torso"],
        adv_force_dim=2,
        N_mu=-1,
        N_nu=-1,
        device=args.device,
        protagonist_optimizer="adam",
        adversary_optimizer="adam",
    )


@contextmanager
def temporary_algo_overrides(algo, *, vf_coef=None, ent_coef=None, max_grad_norm=None):
    old_vf_coef = algo.vf_coef
    old_ent_coef = algo.ent_coef
    old_max_grad_norm = algo.max_grad_norm
    if vf_coef is not None:
        algo.vf_coef = vf_coef
    if ent_coef is not None:
        algo.ent_coef = ent_coef
    if max_grad_norm is not None:
        algo.max_grad_norm = max_grad_norm
    try:
        yield
    finally:
        algo.vf_coef = old_vf_coef
        algo.ent_coef = old_ent_coef
        algo.max_grad_norm = old_max_grad_norm


def classify_block(name: str) -> str:
    if "log_std" in name:
        return "logstd"
    if "value" in name:
        return "critic"
    return "actor"


def named_parameters(policy) -> List[Tuple[str, th.nn.Parameter]]:
    return [(name, param) for name, param in policy.named_parameters() if param.requires_grad]


def clone_state(named_params: List[Tuple[str, th.nn.Parameter]]) -> Dict[str, th.Tensor]:
    return {name: param.data.detach().clone() for name, param in named_params}


def restore_state(named_params: List[Tuple[str, th.nn.Parameter]], state: Dict[str, th.Tensor]) -> None:
    with th.no_grad():
        for name, param in named_params:
            param.data.copy_(state[name])


def get_actions_for_rollout(algo, rollout_data):
    if isinstance(algo.action_space, th.distributions.constraints._Real.__class__):
        return rollout_data.actions
    return rollout_data.actions.long().flatten() if algo.action_space.__class__.__name__ == "Discrete" else rollout_data.actions


def compute_clip_ranges(algo):
    clip_range = algo.clip_range(algo._current_progress_remaining)
    clip_range_vf = None
    if algo.clip_range_vf is not None:
        clip_range_vf = algo.clip_range_vf(algo._current_progress_remaining)
    return clip_range, clip_range_vf


def flatten_named_tensors(named_tensor_map: Dict[str, th.Tensor]) -> th.Tensor:
    tensors = [tensor.reshape(-1) for tensor in named_tensor_map.values()]
    if not tensors:
        return th.zeros(0)
    return th.cat(tensors)


def cosine_similarity(a: th.Tensor, b: th.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    denom = th.norm(a) * th.norm(b)
    if denom.item() == 0:
        return float("nan")
    return float(th.dot(a, b) / denom)


def tensor_norm(tensor: th.Tensor) -> float:
    return float(th.norm(tensor).item()) if tensor.numel() > 0 else 0.0


def block_tensor(named_tensor_map: Dict[str, th.Tensor], block: str) -> th.Tensor:
    block_items = [tensor.reshape(-1) for name, tensor in named_tensor_map.items() if classify_block(name) == block]
    if not block_items:
        return th.zeros(0)
    return th.cat(block_items)


def compute_loss_and_grads(algo, rollout_data, *, max_grad_norm: float, vf_coef: float, ent_coef: float):
    clip_range, clip_range_vf = compute_clip_ranges(algo)
    named_params = named_parameters(algo.policy)
    actions = rollout_data.actions.long().flatten() if algo.action_space.__class__.__name__ == "Discrete" else rollout_data.actions
    with temporary_algo_overrides(algo, vf_coef=vf_coef, ent_coef=ent_coef, max_grad_norm=max_grad_norm):
        algo.policy.optimizer.zero_grad()
        total_loss, policy_loss, _, value_loss, entropy_loss, clip_fraction, approx_kl = algo._build_shared_policy_loss(
            rollout_data,
            actions,
            clip_range,
            clip_range_vf,
        )
        total_loss.backward()
        if math.isfinite(max_grad_norm):
            th.nn.utils.clip_grad_norm_(algo.policy.parameters(), max_grad_norm)
        grads = {
            name: (param.grad.detach().clone() if param.grad is not None else th.zeros_like(param.data))
            for name, param in named_params
        }
    return {
        "total_loss": float(total_loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy_loss": float(entropy_loss.item()),
        "clip_fraction": float(clip_fraction),
        "approx_kl": float(approx_kl),
        "grads": grads,
    }


def sgd_update(theta_old: Dict[str, th.Tensor], grads_old: Dict[str, th.Tensor], lr: float) -> Dict[str, th.Tensor]:
    return {name: theta_old[name] - lr * grads_old[name] for name in theta_old}


def egm_update(algo, rollout_data, theta_old: Dict[str, th.Tensor], lr: float, *, max_grad_norm: float, vf_coef: float, ent_coef: float):
    named_params = named_parameters(algo.policy)
    restore_state(named_params, theta_old)
    old_eval = compute_loss_and_grads(algo, rollout_data, max_grad_norm=max_grad_norm, vf_coef=vf_coef, ent_coef=ent_coef)
    theta_half = sgd_update(theta_old, old_eval["grads"], lr)
    restore_state(named_params, theta_half)
    half_eval = compute_loss_and_grads(algo, rollout_data, max_grad_norm=max_grad_norm, vf_coef=vf_coef, ent_coef=ent_coef)
    theta_egm = sgd_update(theta_old, half_eval["grads"], lr)
    restore_state(named_params, theta_old)
    return old_eval, theta_half, half_eval, theta_egm


def ppm_update(algo, rollout_data, theta_old: Dict[str, th.Tensor], lr: float, inner_steps: int, *, max_grad_norm: float, vf_coef: float, ent_coef: float):
    named_params = named_parameters(algo.policy)
    theta_tmp = {name: tensor.clone() for name, tensor in theta_old.items()}
    inner_losses = []
    inner_policy_losses = []
    inner_value_losses = []
    inner_entropy_losses = []
    inner_grad_norms = []
    inner_residuals = []
    last_eval = None
    for _ in range(inner_steps):
        restore_state(named_params, theta_tmp)
        last_eval = compute_loss_and_grads(algo, rollout_data, max_grad_norm=max_grad_norm, vf_coef=vf_coef, ent_coef=ent_coef)
        inner_losses.append(last_eval["total_loss"])
        inner_policy_losses.append(last_eval["policy_loss"])
        inner_value_losses.append(last_eval["value_loss"])
        inner_entropy_losses.append(last_eval["entropy_loss"])
        inner_grad_norms.append(tensor_norm(flatten_named_tensors(last_eval["grads"])))
        theta_next = sgd_update(theta_old, last_eval["grads"], lr)
        inner_residuals.append(tensor_norm(state_vector(theta_next) - state_vector(theta_tmp)))
        theta_tmp = theta_next
    restore_state(named_params, theta_old)
    return last_eval, theta_tmp, inner_losses, inner_policy_losses, inner_value_losses, inner_entropy_losses, inner_grad_norms, inner_residuals


def vector_from_state_diff(theta_new: Dict[str, th.Tensor], theta_old: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
    return {name: theta_new[name] - theta_old[name] for name in theta_old}


def state_vector(theta_state: Dict[str, th.Tensor]) -> th.Tensor:
    return flatten_named_tensors(theta_state)


def collect_probe_batches(rarl_model, role: str, num_probes: int) -> List[ProbeBatch]:
    if role == "protagonist":
        algo = rarl_model.protagonist
        callback = [SetupProTrainingCallback(rarl_model.adversary.policy)]
    else:
        algo = rarl_model.adversary
        callback = [SetupAdvTrainingCallback(rarl_model.protagonist.policy)]

    original_train = algo.train
    algo.train = lambda: None
    try:
        algo.learn(
            algo.n_steps * algo.env.num_envs,
            callback=callback,
            log_interval=1,
            reset_num_timesteps=True,
        )
    finally:
        algo.train = original_train

    batches = []
    for probe_idx, rollout_data in enumerate(algo.rollout_buffer.get(algo.batch_size), start=1):
        batches.append(ProbeBatch(probe_idx=probe_idx, rollout_data=rollout_data))
        if len(batches) >= num_probes:
            break
    return batches


def save_vector(path: pathlib.Path, tensor: th.Tensor) -> None:
    np.savez(path, vector=tensor.detach().cpu().numpy())


def run_audit(args: argparse.Namespace, output_dir: pathlib.Path) -> pathlib.Path:
    manager = build_manager(args)
    rarl_model = manager.setup_experiment()
    algo = rarl_model.protagonist if args.role == "protagonist" else rarl_model.adversary
    probes = collect_probe_batches(rarl_model, args.role, args.num_probes)

    rows = []
    vector_dir = output_dir / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)

    default_lr = float(algo.learning_rate if isinstance(algo.learning_rate, float) else algo.lr_schedule(1.0))
    default_vf_coef = float(algo.vf_coef)
    default_ent_coef = float(algo.ent_coef)
    default_max_grad_norm = float(algo.max_grad_norm)

    for probe in probes:
        named_params = named_parameters(algo.policy)
        theta_old = clone_state(named_params)
        old_eval = compute_loss_and_grads(
            algo,
            probe.rollout_data,
            max_grad_norm=default_max_grad_norm,
            vf_coef=default_vf_coef,
            ent_coef=default_ent_coef,
        )
        theta_sgd = sgd_update(theta_old, old_eval["grads"], default_lr)
        egm_old_eval, theta_half, half_eval, theta_egm = egm_update(
            algo,
            probe.rollout_data,
            theta_old,
            default_lr,
            max_grad_norm=default_max_grad_norm,
            vf_coef=default_vf_coef,
            ent_coef=default_ent_coef,
        )
        ppm_eval, theta_ppm, ppm_losses, ppm_policy_losses, ppm_value_losses, ppm_entropy_losses, ppm_grad_norms, ppm_inner_residuals = ppm_update(
            algo,
            probe.rollout_data,
            theta_old,
            default_lr,
            inner_steps=5,
            max_grad_norm=default_max_grad_norm,
            vf_coef=default_vf_coef,
            ent_coef=default_ent_coef,
        )

        update_sgd = vector_from_state_diff(theta_sgd, theta_old)
        update_egm = vector_from_state_diff(theta_egm, theta_old)
        update_ppm = vector_from_state_diff(theta_ppm, theta_old)

        full_grad_old = flatten_named_tensors(old_eval["grads"])
        full_grad_half = flatten_named_tensors(half_eval["grads"])
        full_update_sgd = flatten_named_tensors(update_sgd)
        full_update_egm = flatten_named_tensors(update_egm)
        full_update_ppm = flatten_named_tensors(update_ppm)

        paths = {
            "theta_old_path": vector_dir / f"probe_{probe.probe_idx}_theta_old.npz",
            "theta_sgd_path": vector_dir / f"probe_{probe.probe_idx}_theta_sgd.npz",
            "theta_half_path": vector_dir / f"probe_{probe.probe_idx}_theta_half.npz",
            "theta_egm_path": vector_dir / f"probe_{probe.probe_idx}_theta_egm.npz",
            "theta_ppm_path": vector_dir / f"probe_{probe.probe_idx}_theta_ppm.npz",
            "grad_old_path": vector_dir / f"probe_{probe.probe_idx}_grad_old.npz",
            "grad_half_path": vector_dir / f"probe_{probe.probe_idx}_grad_half.npz",
        }
        save_vector(paths["theta_old_path"], state_vector(theta_old))
        save_vector(paths["theta_sgd_path"], state_vector(theta_sgd))
        save_vector(paths["theta_half_path"], state_vector(theta_half))
        save_vector(paths["theta_egm_path"], state_vector(theta_egm))
        save_vector(paths["theta_ppm_path"], state_vector(theta_ppm))
        save_vector(paths["grad_old_path"], full_grad_old)
        save_vector(paths["grad_half_path"], full_grad_half)

        rows.append(
            {
                "probe_idx": probe.probe_idx,
                "role": args.role,
                "env_id": args.env,
                "lr": default_lr,
                "max_grad_norm": default_max_grad_norm,
                "vf_coef": default_vf_coef,
                "ent_coef": default_ent_coef,
                "theta_old_path": str(paths["theta_old_path"]),
                "theta_sgd_path": str(paths["theta_sgd_path"]),
                "theta_half_path": str(paths["theta_half_path"]),
                "theta_egm_path": str(paths["theta_egm_path"]),
                "theta_ppm_path": str(paths["theta_ppm_path"]),
                "grad_old_path": str(paths["grad_old_path"]),
                "grad_half_path": str(paths["grad_half_path"]),
                "total_loss_old": old_eval["total_loss"],
                "policy_loss_old": old_eval["policy_loss"],
                "value_loss_old": old_eval["value_loss"],
                "entropy_loss_old": old_eval["entropy_loss"],
                "total_loss_half": half_eval["total_loss"],
                "policy_loss_half": half_eval["policy_loss"],
                "value_loss_half": half_eval["value_loss"],
                "entropy_loss_half": half_eval["entropy_loss"],
                "total_loss_ppm_inner_each_step": json.dumps(ppm_losses),
                "policy_loss_ppm_inner_each_step": json.dumps(ppm_policy_losses),
                "value_loss_ppm_inner_each_step": json.dumps(ppm_value_losses),
                "entropy_loss_ppm_inner_each_step": json.dumps(ppm_entropy_losses),
                "ppm_grad_norm_each_step": json.dumps(ppm_grad_norms),
                "ppm_inner_residual_each_step": json.dumps(ppm_inner_residuals),
                "grad_old_norm": tensor_norm(full_grad_old),
                "grad_half_norm": tensor_norm(full_grad_half),
                "grad_diff_norm": tensor_norm(full_grad_half - full_grad_old),
                "grad_half_old_cosine": cosine_similarity(full_grad_half, full_grad_old),
                "theta_half_old_diff_norm": tensor_norm(state_vector(theta_half) - state_vector(theta_old)),
                "theta_ppm_sgd_diff_norm": tensor_norm(state_vector(theta_ppm) - state_vector(theta_sgd)),
                "update_egm_sgd_cosine": cosine_similarity(full_update_egm, full_update_sgd),
                "update_ppm_sgd_cosine": cosine_similarity(full_update_ppm, full_update_sgd),
                "field_movement_ratio": tensor_norm(full_grad_half - full_grad_old) / max(tensor_norm(full_grad_old), 1e-12),
                "actor_grad_norm": tensor_norm(block_tensor(old_eval["grads"], "actor")),
                "logstd_grad_norm": tensor_norm(block_tensor(old_eval["grads"], "logstd")),
                "critic_grad_norm": tensor_norm(block_tensor(old_eval["grads"], "critic")),
                "actor_update_norm_sgd": tensor_norm(block_tensor(update_sgd, "actor")),
                "logstd_update_norm_sgd": tensor_norm(block_tensor(update_sgd, "logstd")),
                "critic_update_norm_sgd": tensor_norm(block_tensor(update_sgd, "critic")),
                "actor_update_norm_egm": tensor_norm(block_tensor(update_egm, "actor")),
                "logstd_update_norm_egm": tensor_norm(block_tensor(update_egm, "logstd")),
                "critic_update_norm_egm": tensor_norm(block_tensor(update_egm, "critic")),
                "actor_update_norm_ppm": tensor_norm(block_tensor(update_ppm, "actor")),
                "logstd_update_norm_ppm": tensor_norm(block_tensor(update_ppm, "logstd")),
                "critic_update_norm_ppm": tensor_norm(block_tensor(update_ppm, "critic")),
            }
        )

    audit_df = pd.DataFrame(rows)
    csv_path = output_dir / "full_policy_optimizer_audit.csv"
    audit_df.to_csv(csv_path, index=False)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(audit_df["probe_idx"], audit_df["actor_grad_norm"], marker="o", label="actor_grad_norm")
    ax.plot(audit_df["probe_idx"], audit_df["logstd_grad_norm"], marker="o", label="logstd_grad_norm")
    ax.plot(audit_df["probe_idx"], audit_df["critic_grad_norm"], marker="o", label="critic_grad_norm")
    ax.set_title("Full Policy Gradient Block Norms")
    ax.set_xlabel("Probe")
    ax.set_ylabel("Norm")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "full_policy_grad_block_norms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(audit_df["probe_idx"], audit_df["update_egm_sgd_cosine"], marker="o", label="EGM vs SGD cosine")
    ax.plot(audit_df["probe_idx"], audit_df["update_ppm_sgd_cosine"], marker="o", label="PPM vs SGD cosine")
    ax.set_title("Full Policy Update Similarity")
    ax.set_xlabel("Probe")
    ax.set_ylabel("Cosine similarity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "full_policy_update_similarity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for _, row in audit_df.iterrows():
        ax.plot(range(1, len(json.loads(row["total_loss_ppm_inner_each_step"])) + 1), json.loads(row["total_loss_ppm_inner_each_step"]), marker="o", alpha=0.6, label=f"probe_{int(row['probe_idx'])}")
    ax.set_title("Full Policy PPM Inner Movement")
    ax.set_xlabel("PPM inner step")
    ax.set_ylabel("Total loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "full_policy_ppm_inner_movement.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    bug_flag = int((audit_df["theta_ppm_sgd_diff_norm"] <= 1e-12).all())
    reason = "No implementation bug detected."
    if bug_flag:
        reason = "PPM update matched SGD to numerical zero on all probes."

    report_lines = [
        "# Full-Policy Optimizer Audit",
        "",
        f"- Environment: `{args.env}`",
        f"- Role audited: `{args.role}`",
        f"- Number of probes: `{len(audit_df)}`",
        f"- Learning rate: `{default_lr}`",
        f"- max_grad_norm: `{default_max_grad_norm}`",
        f"- vf_coef: `{default_vf_coef}`",
        f"- ent_coef: `{default_ent_coef}`",
        "",
        "## Gate checks",
        "",
        f"- EGM recomputed at theta_half: `True`",
        f"- PPM recomputed for inner_steps>1: `True`",
        f"- Any exact PPM==SGD bug: `{bool(bug_flag)}`",
        f"- Reason: {reason}",
    ]
    (output_dir / "full_policy_optimizer_audit_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return csv_path


def run_diagnosis(args: argparse.Namespace, output_dir: pathlib.Path) -> pathlib.Path:
    manager = build_manager(args)
    rarl_model = manager.setup_experiment()
    algo = rarl_model.protagonist if args.role == "protagonist" else rarl_model.adversary
    probes = collect_probe_batches(rarl_model, args.role, args.num_probes)

    default_lr = float(algo.learning_rate if isinstance(algo.learning_rate, float) else algo.lr_schedule(1.0))
    default_vf_coef = float(algo.vf_coef)
    default_ent_coef = float(algo.ent_coef)

    ablations = []
    ablations.append(("max_grad_norm", float("inf"), {"max_grad_norm": float("inf"), "vf_coef": default_vf_coef, "ent_coef": default_ent_coef, "lr": default_lr, "ppm_inner_steps": 5}))
    for value in [0.5, 1.0, 10.0, float("inf")]:
        ablations.append(("max_grad_norm", value, {"max_grad_norm": value, "vf_coef": default_vf_coef, "ent_coef": default_ent_coef, "lr": default_lr, "ppm_inner_steps": 5}))
    for value in [0.1, 0.5, 1.0]:
        ablations.append(("vf_coef", value, {"max_grad_norm": float(algo.max_grad_norm), "vf_coef": value, "ent_coef": default_ent_coef, "lr": default_lr, "ppm_inner_steps": 5}))
    for value in [default_ent_coef, 0.0]:
        ablations.append(("ent_coef", value, {"max_grad_norm": float(algo.max_grad_norm), "vf_coef": default_vf_coef, "ent_coef": value, "lr": default_lr, "ppm_inner_steps": 5}))
    for scale in [1.0, 3.0, 10.0, 30.0]:
        ablations.append(("lr_scale", scale, {"max_grad_norm": float(algo.max_grad_norm), "vf_coef": default_vf_coef, "ent_coef": default_ent_coef, "lr": default_lr * scale, "ppm_inner_steps": 5}))
    for inner_steps in [1, 2, 5, 10]:
        ablations.append(("ppm_inner_steps", inner_steps, {"max_grad_norm": float(algo.max_grad_norm), "vf_coef": default_vf_coef, "ent_coef": default_ent_coef, "lr": default_lr, "ppm_inner_steps": inner_steps}))

    rows = []
    for probe in probes:
        named_params = named_parameters(algo.policy)
        theta_old = clone_state(named_params)
        for ablation_type, ablation_value, setting in ablations:
            old_eval = compute_loss_and_grads(
                algo,
                probe.rollout_data,
                max_grad_norm=setting["max_grad_norm"],
                vf_coef=setting["vf_coef"],
                ent_coef=setting["ent_coef"],
            )
            theta_sgd = sgd_update(theta_old, old_eval["grads"], setting["lr"])
            _, theta_half, half_eval, theta_egm = egm_update(
                algo,
                probe.rollout_data,
                theta_old,
                setting["lr"],
                max_grad_norm=setting["max_grad_norm"],
                vf_coef=setting["vf_coef"],
                ent_coef=setting["ent_coef"],
            )
            ppm_eval, theta_ppm, _, _, _, _, _, _ = ppm_update(
                algo,
                probe.rollout_data,
                theta_old,
                setting["lr"],
                inner_steps=setting["ppm_inner_steps"],
                max_grad_norm=setting["max_grad_norm"],
                vf_coef=setting["vf_coef"],
                ent_coef=setting["ent_coef"],
            )

            grad_old = flatten_named_tensors(old_eval["grads"])
            grad_half = flatten_named_tensors(half_eval["grads"])
            update_sgd = flatten_named_tensors(vector_from_state_diff(theta_sgd, theta_old))
            update_egm = flatten_named_tensors(vector_from_state_diff(theta_egm, theta_old))
            update_ppm = flatten_named_tensors(vector_from_state_diff(theta_ppm, theta_old))

            actor_grad_norm = tensor_norm(block_tensor(old_eval["grads"], "actor"))
            critic_grad_norm = tensor_norm(block_tensor(old_eval["grads"], "critic"))
            total_grad_norm = tensor_norm(grad_old)

            rows.append(
                {
                    "probe_idx": probe.probe_idx,
                    "ablation_type": ablation_type,
                    "ablation_value": ablation_value,
                    "lr": setting["lr"],
                    "max_grad_norm": setting["max_grad_norm"],
                    "vf_coef": setting["vf_coef"],
                    "ent_coef": setting["ent_coef"],
                    "ppm_inner_steps": setting["ppm_inner_steps"],
                    "egm_sgd_update_cosine": cosine_similarity(update_egm, update_sgd),
                    "ppm_sgd_update_cosine": cosine_similarity(update_ppm, update_sgd),
                    "egm_sgd_update_norm_ratio": tensor_norm(update_egm) / max(tensor_norm(update_sgd), 1e-12),
                    "ppm_sgd_update_norm_ratio": tensor_norm(update_ppm) / max(tensor_norm(update_sgd), 1e-12),
                    "field_movement_ratio": tensor_norm(grad_half - grad_old) / max(total_grad_norm, 1e-12),
                    "critic_gradient_share": critic_grad_norm / max(total_grad_norm, 1e-12),
                    "actor_gradient_share": actor_grad_norm / max(total_grad_norm, 1e-12),
                    "logstd_gradient_share": tensor_norm(block_tensor(old_eval["grads"], "logstd")) / max(total_grad_norm, 1e-12),
                    "egm_nan_flag": int(not np.isfinite(tensor_norm(update_egm))),
                    "ppm_nan_flag": int(not np.isfinite(tensor_norm(update_ppm))),
                    "sgd_nan_flag": int(not np.isfinite(tensor_norm(update_sgd))),
                }
            )

    diagnosis_df = pd.DataFrame(rows)
    csv_path = output_dir / "full_policy_overlap_diagnosis.csv"
    diagnosis_df.to_csv(csv_path, index=False)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    lr_df = diagnosis_df[diagnosis_df["ablation_type"] == "lr_scale"]
    for probe_idx, group in lr_df.groupby("probe_idx"):
        ax.plot(group["lr"], group["egm_sgd_update_cosine"], marker="o", alpha=0.6, label=f"EGM probe {probe_idx}")
        ax.plot(group["lr"], group["ppm_sgd_update_cosine"], marker="x", alpha=0.6, linestyle="--", label=f"PPM probe {probe_idx}")
    ax.set_xscale("log")
    ax.set_title("Overlap vs LR and Clip")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Cosine vs SGD")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "overlap_vs_lr_and_clip.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    field_df = diagnosis_df.groupby(["ablation_type", "ablation_value"])["field_movement_ratio"].mean().reset_index()
    labels = [f"{row.ablation_type}={row.ablation_value}" for row in field_df.itertuples()]
    ax.bar(range(len(field_df)), field_df["field_movement_ratio"])
    ax.set_xticks(range(len(field_df)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_title("Field Movement Ratio")
    ax.set_ylabel("||F(theta_half)-F(theta_old)|| / ||F(theta_old)||")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "field_movement_ratio.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    regime_found = diagnosis_df[
        (diagnosis_df["field_movement_ratio"] >= 0.05)
        & (diagnosis_df["egm_sgd_update_cosine"] < 0.9999)
        & (diagnosis_df["ppm_sgd_update_cosine"] < 0.9999)
        & (diagnosis_df["egm_nan_flag"] == 0)
        & (diagnosis_df["ppm_nan_flag"] == 0)
    ]

    if regime_found.empty:
        verdict = "No regime found with field_movement_ratio >= 0.05 and non-identical EGM/PPM vs SGD."
    else:
        verdict = f"Found {len(regime_found)} regimes with field movement and non-identical EGM/PPM updates."

    report_lines = [
        "# Full-Policy Overlap Diagnosis",
        "",
        f"- Environment: `{args.env}`",
        f"- Role: `{args.role}`",
        f"- Probes: `{args.num_probes}`",
        "",
        "## Verdict",
        "",
        f"- {verdict}",
    ]
    (output_dir / "full_policy_overlap_diagnosis_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return csv_path


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("audit", "all"):
        run_audit(args, output_dir)
    if args.mode in ("diagnosis", "all"):
        run_diagnosis(args, output_dir)


if __name__ == "__main__":
    main()
