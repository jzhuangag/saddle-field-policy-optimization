from __future__ import annotations

import argparse
import json
import pathlib
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Audit optimizer scopes for standard alternating RARL without training")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument(
        "--output-root",
        type=str,
        default=r"C:\Users\jzhuangag\work\rarl\original\results\standard_rarl_baseline_positive_search",
    )
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--fallback-env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--shared-lr", type=float, default=1e-3)
    parser.add_argument("--shared-max-grad-norm", type=float, default=10.0)
    parser.add_argument("--shared-vf-coef", type=float, default=0.5)
    return parser.parse_args()


def ensure_hyperparams(repo_dir: pathlib.Path, output_root: pathlib.Path, requested_env: str, fallback_env: str) -> pathlib.Path:
    source_path = repo_dir / "hyperparameter" / "PPO-rarl.yml"
    temp_dir = output_root / "temp_hyperparams"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target_path = temp_dir / "PPO-rarl.yml"

    with source_path.open("r", encoding="utf-8") as handle:
        hyperparams = yaml.safe_load(handle)

    mapping_note = "native"
    if requested_env not in hyperparams:
        if fallback_env not in hyperparams:
            raise KeyError(f"Neither {requested_env} nor fallback {fallback_env} exist in {source_path}")
        hyperparams[requested_env] = hyperparams[fallback_env]
        mapping_note = f"copied_from_{fallback_env}"

    with target_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(hyperparams, handle, sort_keys=False)

    metadata = {
        "requested_env": requested_env,
        "fallback_env": fallback_env,
        "env_used": requested_env,
        "mapping_note": mapping_note,
        "source_yaml": str(source_path),
        "target_yaml": str(target_path),
    }
    (temp_dir / "hyperparam_mapping.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return temp_dir


def build_args_namespace(
    *,
    env_id: str,
    seed: int,
    device: str,
    optimizer_scope: str,
    hyperparam_dir: pathlib.Path,
    run_root: pathlib.Path,
    shared_lr: float,
    shared_max_grad_norm: float,
    shared_vf_coef: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        verbose=1,
        seed=seed,
        num_exps=1,
        num_threads=-1,
        env=env_id,
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
        hyperparameter_path=str(hyperparam_dir),
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
        n_timesteps=1,
        save_freq=10240,
        log_interval=-1,
        device=device,
        eval_freq=10240,
        n_eval_envs=1,
        n_eval_episodes=5,
        control_proxy_eval=False,
        tensorboard_log=str(run_root / "tb"),
        log_folder=str(run_root / "log"),
        protagonist_policy="MlpPolicy",
        adversary_policy="MlpPolicy",
        protagonist_optimizer="sgd",
        adversary_optimizer="sgd",
        protagonist_optimizer_kwargs={},
        adversary_optimizer_kwargs={},
        protagonist_lr=shared_lr,
        adversary_lr=shared_lr,
        protagonist_max_grad_norm=shared_max_grad_norm,
        adversary_max_grad_norm=shared_max_grad_norm,
        protagonist_vf_coef=shared_vf_coef,
        adversary_vf_coef=shared_vf_coef,
        optimizer_scope=optimizer_scope,
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
        qp_beta_probe=shared_lr,
        qp_gamma_probe=max(shared_lr * shared_lr, 1e-6),
        qp_ridge=1e-8,
        qp_actor_weight=1.0,
        qp_logstd_weight=1.0,
        qp_step_solver="lyapunov_quadratic_bound",
        N_mu=-1,
        N_nu=-1,
        adv_impact="control",
        adv_delay=-1,
        adv_fraction=1.0,
        adv_fraction_override=True,
        requested_alpha=1.0,
        adv_index_list=["torso"],
        adv_force_dim=2,
    )


def classify_parameter_name(name: str) -> str:
    if "log_std" in name:
        return "log_std"
    if "value" in name:
        return "value"
    if any(token in name for token in ("policy_net", "action_net")):
        return "actor"
    return "shared"


def unique_sample(values: List[str], limit: int = 8) -> List[str]:
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
        if len(seen) >= limit:
            break
    return seen


def audit_optimizer_params(agent_name: str, optimizer_scope: str, ppo_model) -> Dict[str, object]:
    named_params: List[Tuple[str, object]] = [(name, param) for name, param in ppo_model.policy.named_parameters() if param.requires_grad]
    optimizer_param_ids = {id(param) for group in ppo_model.policy.optimizer.param_groups for param in group["params"]}
    selected = [(name, param) for name, param in named_params if id(param) in optimizer_param_ids]
    selected_names = [name for name, _ in selected]
    classes = [classify_parameter_name(name) for name in selected_names]

    return {
        "optimizer_scope": optimizer_scope,
        "agent_name": agent_name,
        "number_of_trainable_optimizer_tensors": len(selected),
        "number_of_trainable_optimizer_params": int(sum(param.numel() for _, param in selected)),
        "number_of_total_trainable_policy_tensors": len(named_params),
        "number_of_total_trainable_policy_params": int(sum(param.numel() for _, param in named_params)),
        "sample_parameter_names": json.dumps(selected_names[:8]),
        "sample_actor_names": json.dumps(unique_sample([name for name in selected_names if classify_parameter_name(name) == "actor"])),
        "sample_log_std_names": json.dumps(unique_sample([name for name in selected_names if classify_parameter_name(name) == "log_std"])),
        "sample_value_names": json.dumps(unique_sample([name for name in selected_names if classify_parameter_name(name) == "value"])),
        "sample_shared_names": json.dumps(unique_sample([name for name in selected_names if classify_parameter_name(name) == "shared"])),
        "contains_actor_params": bool("actor" in classes),
        "contains_log_std": bool("log_std" in classes),
        "contains_value_params": bool("value" in classes),
        "contains_shared_params": bool("shared" in classes),
        "optimizer_class": type(ppo_model.policy.optimizer).__name__,
        "critic_optimizer_present": getattr(ppo_model, "_actor_game_critic_optimizer", None) is not None,
    }


def close_model(model) -> None:
    for attr_name in ("protagonist", "adversary"):
        agent = getattr(model, attr_name, None)
        env = getattr(agent, "env", None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from utils.exp_manager import ExperimentManager

    hyperparam_dir = ensure_hyperparams(repo_dir, output_root, args.env, args.fallback_env)

    rows: List[Dict[str, object]] = []
    for optimizer_scope in ("full_policy", "actor_logstd_only"):
        run_root = output_root / f"scope_{optimizer_scope}"
        run_root.mkdir(parents=True, exist_ok=True)
        ns = build_args_namespace(
            env_id=args.env,
            seed=args.seed,
            device=args.device,
            optimizer_scope=optimizer_scope,
            hyperparam_dir=hyperparam_dir,
            run_root=run_root,
            shared_lr=args.shared_lr,
            shared_max_grad_norm=args.shared_max_grad_norm,
            shared_vf_coef=args.shared_vf_coef,
        )
        manager = ExperimentManager(
            ns,
            algo="rarl",
            env_id=args.env,
            log_folder=ns.log_folder,
            tensorboard_log=ns.tensorboard_log,
            n_timesteps=ns.n_timesteps,
            eval_freq=ns.eval_freq,
            n_eval_episodes=ns.n_eval_episodes,
            save_freq=ns.save_freq,
            hyperparameter_path=ns.hyperparameter_path,
            hyperparams=ns.hyperparameter,
            env_kwargs=ns.env_kwargs,
            model_path=str(run_root / "saved_models" / "rarl-ppo" / args.env),
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
            raise RuntimeError("Unexpected hyperparameter-optimization mode during scope audit.")
        rows.append(audit_optimizer_params("protagonist", optimizer_scope, model.protagonist))
        rows.append(audit_optimizer_params("adversary", optimizer_scope, model.adversary))
        close_model(model)

    audit_df = pd.DataFrame(rows).sort_values(["optimizer_scope", "agent_name"]).reset_index(drop=True)
    csv_path = output_root / "optimizer_scope_audit.csv"
    md_path = output_root / "optimizer_scope_audit.md"
    audit_df.to_csv(csv_path, index=False)

    lines = [
        "# Optimizer Scope Audit",
        "",
        f"- Environment: `{args.env}`",
        f"- Seed: `{args.seed}`",
        "- No training was run. This report only inspects instantiated protagonist/adversary PPO optimizers inside alternating RARL.",
        "",
        "## Findings",
        "",
    ]
    for scope in ("full_policy", "actor_logstd_only"):
        scope_df = audit_df[audit_df["optimizer_scope"] == scope]
        lines.append(f"### `{scope}`")
        lines.append("")
        for _, row in scope_df.iterrows():
            lines.append(f"- `{row['agent_name']}` optimizer tensors: `{row['number_of_trainable_optimizer_tensors']}`")
            lines.append(f"- `{row['agent_name']}` optimizer parameter count: `{row['number_of_trainable_optimizer_params']}`")
            lines.append(
                f"- `{row['agent_name']}` contains actor/log_std/value/shared: "
                f"`{row['contains_actor_params']}` / `{row['contains_log_std']}` / "
                f"`{row['contains_value_params']}` / `{row['contains_shared_params']}`"
            )
            lines.append(f"- `{row['agent_name']}` sample parameter names: `{row['sample_parameter_names']}`")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- `full_policy` should include actor, log_std, and value-network parameters.",
            "- `actor_logstd_only` is implemented to exclude all value-network parameters from the main PPO optimizer while keeping actor/log_std parameters trainable.",
            "- `contains_shared_params=True` means the optimizer touches parameters outside explicit `policy_net`/`action_net`/`value`/`log_std` buckets, e.g. shared feature or extractor blocks.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
