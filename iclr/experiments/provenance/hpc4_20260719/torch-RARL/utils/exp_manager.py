import os, sys
import argparse
import yaml
import warnings
import optuna
import time
import pickle as pkl
import gymnasium as gym
import numpy as np 
from collections import OrderedDict
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch

from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.integration.skopt import SkoptSampler
from optuna.pruners import BasePruner, MedianPruner, SuccessiveHalvingPruner
from optuna.visualization import plot_optimization_history, plot_param_importances

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize, VecTransposeImage, is_vecenv_wrapped
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.hyperparams_opt import HYPERPARAMS_SAMPLER
from utils.utils import get_latest_run_id, linear_schedule, get_wrapper_class, get_callback_list
from utils.callbacks import (
    AdversarialEvalCallback,
    ParameterNormLoggingCallback,
    SaveVecNormalizeCallback,
    TrialEvalCallback,
    SetupProTrainingCallback,
    SetupAdvTrainingCallback,
    ProtagonistEpisodeReturnCallback,
)
from utils.wrappers import AdversarialClassicControlWrapper, AdversarialMujocoWrapper
from models.algorithms import ALGOS
from models.optimizers import get_optimizer_class


class ExperimentManager(object):
    """
    Reads and preprocesses hyperparameter. 
    Creates the environment and prepares the RL model for training.

    Taken and modified from RL Baselines3 Zoo 
    <https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/exp_manager.py>

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
    def __init__(self, 
        args: argparse.Namespace,
        algo: str,
        env_id: str,
        log_folder: str,
        tensorboard_log: str = "",
        n_timesteps: int = 0,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_freq: int = -1,
        hyperparameter_path: str = "",
        hyperparams: Optional[Dict[str, Any]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        model_path: str = "",
        pretrained_model: str = "",
        optimize_hyperparameters: bool = False,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        n_opt_trials: int = 1,
        n_jobs: int = 1,
        sampler: str = "tpe",
        pruner: str = "median",
        optimization_log_path: Optional[str] = None,
        n_startup_trials: int = 0,
        n_evaluations_opt: int = 1,
        seed: int = 0,
        log_interval: int = 0,
        save_replay_buffer: bool = False,
        verbose: int = 1,
        vec_env_type: str = "dummy",
        n_envs: int = 1,
        n_eval_envs: int = 1,
        no_optim_plots: bool = False,
        adv_env: bool = False,
        adv_impact: str = "",
        adv_fraction: float = 1.0,
        adv_delay: int = -1,
        rarl_update_mode: str = "alternating",
        clean_warmup: bool = False,
        composite_outer_mode: str = "none",
        composite_outer_warmup: int = 10,
        composite_critic_updates: int = 100,
        composite_batch_size: int = 256,
        composite_state_batch_size: int = 256,
        composite_actor_lr: float = 1e-6,
        adv_index_list: List = None,
        adv_force_dim: int = 2,
        N_mu: int = 10, 
        N_nu: int = 10,
        device: str = None,
        rarl_config: str = "trpo",
        protagonist_optimizer: Optional[str] = None,
        adversary_optimizer: Optional[str] = None,
        protagonist_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        adversary_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        protagonist_lr: Optional[float] = None,
        adversary_lr: Optional[float] = None,
        protagonist_max_grad_norm: Optional[float] = None,
        adversary_max_grad_norm: Optional[float] = None,
        protagonist_vf_coef: Optional[float] = None,
        adversary_vf_coef: Optional[float] = None,
        optimizer_scope: str = "full_policy",
        qp_normalization: str = "none",
        qp_g_alpha: float = 1e-3,
        max_update_norm: float = float("inf"),
        qp_eps: float = 1e-8,
        qp_alpha: float = 0.3,
        qp_beta_max: float = 1.0,
        qp_gamma_max: float = 1.0,
        qp_step_grid: str = "0,0.1,0.3,1.0,3.0",
        qp_objective: str = "loss",
        qp_accept_rule: str = "none",
        qp_min_g_contribution: float = 0.0,
        qp_critic_weight: float = 1.0,
        qp_g_sign: str = "plus",
        qp_fd_eps: float = 1e-3,
        qp_beta_probe: float = 1e-3,
        qp_gamma_probe: float = 1e-3,
        qp_ridge: float = 1e-8,
        qp_actor_weight: float = 1.0,
        qp_logstd_weight: float = 1.0,
        qp_step_solver: str = "lyapunov_quadratic_bound",
        control_proxy_eval: bool = False,
    ) -> None:
        super(ExperimentManager, self).__init__()
        self.args = args
        self.seed = seed

        # algorithm
        self.algo = algo
        self.rarl_config = rarl_config.lower()
        self.algo_tag = self.algo if self.algo != "rarl" else f"{self.algo}-{self.rarl_config}"
        self.protagonist_optimizer = protagonist_optimizer.lower() if protagonist_optimizer is not None else None
        self.adversary_optimizer = adversary_optimizer.lower() if adversary_optimizer is not None else None
        self.protagonist_optimizer_kwargs = protagonist_optimizer_kwargs
        self.adversary_optimizer_kwargs = adversary_optimizer_kwargs
        self.protagonist_lr = protagonist_lr
        self.adversary_lr = adversary_lr
        self.protagonist_max_grad_norm = protagonist_max_grad_norm
        self.adversary_max_grad_norm = adversary_max_grad_norm
        self.protagonist_vf_coef = protagonist_vf_coef
        self.adversary_vf_coef = adversary_vf_coef
        self.optimizer_scope = optimizer_scope
        self.qp_normalization = qp_normalization
        self.qp_g_alpha = qp_g_alpha
        self.max_update_norm = max_update_norm
        self.qp_eps = qp_eps
        self.qp_alpha = qp_alpha
        self.qp_beta_max = qp_beta_max
        self.qp_gamma_max = qp_gamma_max
        self.qp_step_grid = qp_step_grid
        self.qp_objective = qp_objective
        self.qp_accept_rule = qp_accept_rule
        self.qp_min_g_contribution = qp_min_g_contribution
        self.qp_critic_weight = qp_critic_weight
        self.qp_g_sign = qp_g_sign
        self.qp_fd_eps = qp_fd_eps
        self.qp_beta_probe = qp_beta_probe
        self.qp_gamma_probe = qp_gamma_probe
        self.qp_ridge = qp_ridge
        self.qp_actor_weight = qp_actor_weight
        self.qp_logstd_weight = qp_logstd_weight
        self.qp_step_solver = qp_step_solver
        self.control_proxy_eval = control_proxy_eval

        # environment
        self.n_envs = n_envs  # will be updated when reading hyperparams
        self.n_actions = None  # For DDPG/TD3 action noise objects
        self.env_id = env_id
        self.env_kwargs = {} if env_kwargs is None else env_kwargs
        self.normalize = False
        self.normalize_kwargs = {}
        self.env_wrapper = None
        self.frame_stack = None
        self.adv_env = adv_env

        self.vec_env_class = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[vec_env_type]
        self.vec_env_kwargs = {}

        self._is_atari = self.is_atari(env_id)
        # self.vec_env_kwargs = {} if vec_env_type == "dummy" else {"start_method": "fork"}

        # adversarial env
        self.adv_fraction = adv_fraction
        self.requested_adv_fraction = adv_fraction
        self.explicit_adv_fraction_override = bool(
            getattr(args, "adv_fraction_override", False)
            or getattr(args, "alpha_override", False)
            or getattr(args, "adv_fraction_explicit", False)
        )
        self.adv_impact = adv_impact
        self.adv_index_list = adv_index_list
        self.adv_force_dim = adv_force_dim

        # training
        self.n_timesteps = n_timesteps
        self.save_freq = save_freq
        self.device = device

        # evaluation
        self.n_eval_episodes = n_eval_episodes
        self.n_eval_envs = n_eval_envs
        self.eval_freq = eval_freq
        self.save_replay_buffer = save_replay_buffer

        # callbacks
        self.specified_callbacks = []
        self.callbacks = []
        self.protagonist_callbacks = []
        self.adversary_callbacks = []

        # hyperparameters
        self.hyperparameter_path = hyperparameter_path
        self.custom_hyperparams = hyperparams
        self._hyperparams = {}
        
        # hyperparameter optimization config
        self.optimize_hyperparameters = optimize_hyperparameters
        self.optimize_hyperparameters_path = os.path.join(optimization_log_path, self.algo_tag, env_id)
        self.storage = storage
        self.study_name = study_name
        self.no_optim_plots = no_optim_plots
    
        self.n_opt_trials = n_opt_trials    # maximum number of trials for finding the best hyperparams
        self.n_jobs = n_jobs     # number of parallel jobs when doing hyperparameter search
        self.sampler = sampler
        self.pruner = pruner
        self.n_startup_trials = n_startup_trials
        self.n_evaluations_opt = n_evaluations_opt
        self.deterministic_eval = not self.is_atari(self.env_id)

        # logging
        self.verbose = verbose
        self.tensorboard_log = None if tensorboard_log == "" else os.path.join(tensorboard_log, env_id)
        self.log_interval = log_interval

        # paths
        self.model_path = model_path
        if pretrained_model == "":
            self.save_path = os.path.join(self.model_path, f"{self.env_id}_{get_latest_run_id(self.model_path, self.env_id) + 1}")
            self.continue_training = False
        else: 
            self.save_path = os.path.join(self.model_path, pretrained_model)
            self.continue_training = True
        self.params_path = os.path.join(self.save_path, self.env_id)

        # RARL
        self.adv_delay = adv_delay
        self.rarl_update_mode = rarl_update_mode
        self.clean_warmup = clean_warmup
        self.composite_outer_mode = composite_outer_mode
        self.composite_outer_warmup = int(composite_outer_warmup)
        self.composite_critic_updates = int(composite_critic_updates)
        self.composite_batch_size = int(composite_batch_size)
        self.composite_state_batch_size = int(composite_state_batch_size)
        self.composite_actor_lr = float(composite_actor_lr)
        self.N_mu = N_mu
        self.N_nu = N_nu


    def setup_experiment(self) -> Optional[BaseAlgorithm]:
        """Prepares experiment by creating environment, loading and preprocessing hyperparameters, etc.

        Returns:
            Optional[BaseAlgorithm]: return RL algorithm
        """
        hyperparams, sorted_hyperparams = self.read_hyperparameters()
        hyperparams, self.env_wrapper, self.callbacks = self._preprocess_hyperparams(hyperparams)

        # set up paths
        if not os.path.exists(self.params_path):
            os.makedirs(self.params_path)
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        if not os.path.exists(self.optimize_hyperparameters_path): 
            os.makedirs(self.optimize_hyperparameters_path)

        # create callbacks for train and test environments
        self.create_callbacks()

        # create environments
        n_envs = self.n_envs

        env = self.create_envs(n_envs, no_log=False)
        
        # preprocess action noise
        self._hyperparams = self._preprocess_action_noise(hyperparams, env)
        
        # define model and account for pre-trained model
        if self.continue_training:
            model = self._load_pretrained_agent(self._hyperparams, env)
        elif self.optimize_hyperparameters:
            return None
        else:
            # Train an agent from scratch
            model = ALGOS[self.algo](
                env=env,
                tensorboard_log=self.tensorboard_log,
                seed=self.seed,
                verbose=self.verbose,
                **self._hyperparams,
            )
        
        # save configs
        self._save_config(sorted_hyperparams)
        
        return model 


    def read_hyperparameters(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Read hyperparameters from yaml file and order to for storage later

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: (parsed hyperparameters, ordered hyperparameters for storage)
        """
        # load hyperparameters from yaml file
        hyperparameter_file = f"{self.algo}.yml"
        if self.algo == "rarl":
            rarl_files = {
                "trpo": "TRPO-rarl.yml",
                "ppo": "PPO-rarl.yml",
            }
            try:
                hyperparameter_file = rarl_files[self.rarl_config]
            except KeyError as exc:
                raise ValueError(f"Unsupported rarl_config={self.rarl_config!r}, expected one of {sorted(rarl_files.keys())}") from exc

        hyperparameter_filepath = os.path.join(self.hyperparameter_path, hyperparameter_file)

        if self.verbose > 0:
            print(f"Loading hyperparameters from: {hyperparameter_filepath}")

        with open(hyperparameter_filepath, "r") as f:
            hyperparams_dict = yaml.safe_load(f)
            
            # validate hyperparameter
            if self.env_id in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[self.env_id]
            elif self._is_atari:
                hyperparams = hyperparams_dict["atari"]
            else:
                raise ValueError(f"Hyperparameters not found for {self.algo}-{self.env_id}")


        if self.custom_hyperparams is not None:
            # overwrite hyperparams if needed
            hyperparams.update(self.custom_hyperparams)

        # sort hyperparams that will be saved
        sorted_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

        if self.verbose > 0:
            print("Default hyperparameters for environment (ones being tuned will be overridden):")
            pprint(sorted_hyperparams)

        return hyperparams, sorted_hyperparams
    

    def _preprocess_hyperparams(self, hyperparams: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Callable], List[BaseCallback]]:
        """Preprocess hyperparmeters

        Args:
            hyperparams (Dict[str, Any]): parsed hyperparameters

        Returns:
            Tuple[Dict[str, Any], Optional[Callable], List[BaseCallback]]: (hyperparameter that can be passed to model constructor,
                                                                                environment wrapper,
                                                                                list of callbacks
                                                                            )
        """
        self.n_envs = hyperparams.get("n_envs", 1)

        if self.verbose > 0:
            print(f"Using {self.n_envs} environments")

        # convert schedule strings to objects
        hyperparams = self._preprocess_schedules(hyperparams)

        # pre-process train_freq
        if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
            hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

        # overwrite number of timesteps
        if self.n_timesteps > 0:
            if self.verbose > 0:
                print(f"Overwriting n_timesteps with n={self.n_timesteps}")
        else:
            self.n_timesteps = int(hyperparams["n_timesteps"])
        
        # rarl - overwrite number of timesteps of protagonist and adversary
        if self.algo == "rarl":
            if self.N_mu > 0 and self.N_nu > 0:
                if self.verbose > 0:
                    print(f"Overwriting total_steps_protagonist with n_mu={self.N_mu}")
                    print(f"Overwriting total_steps_adversary with n_nu={self.N_nu}")
            else:
                self.N_mu = int(hyperparams["N_mu"])
                self.N_nu = int(hyperparams["N_nu"])

        yaml_adv_fraction = hyperparams.get("adv_fraction")
        if "adv_fraction" in hyperparams.keys():
            if not self.explicit_adv_fraction_override:
                self.adv_fraction = hyperparams["adv_fraction"]
            del hyperparams["adv_fraction"]

        if self.verbose > 0 and yaml_adv_fraction is not None:
            source = "explicit_override" if self.explicit_adv_fraction_override else "yaml"
            print(
                f"Resolved adv_fraction={self.adv_fraction} "
                f"(requested={self.requested_adv_fraction}, yaml={yaml_adv_fraction}, source={source})"
            )

        if hasattr(self.args, "__dict__"):
            setattr(self.args, "requested_adv_fraction", self.requested_adv_fraction)
            setattr(self.args, "resolved_adv_fraction", self.adv_fraction)
            setattr(self.args, "adv_fraction_override", self.explicit_adv_fraction_override)

        # pre-process normalization config
        hyperparams = self._preprocess_normalization(hyperparams)

        # pre-process policy/buffer keyword arguments
        for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs", "protagonist_kwargs", "protagonist_policy_kwargs", "adversary_kwargs", "adversary_policy_kwargs", "protagonist_optimizer_kwargs", "adversary_optimizer_kwargs"}:
            if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
                hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

        if self.algo == "rarl":
            hyperparams = self._configure_rarl_optimizers(hyperparams)

        # delete keys so the dict can be pass to the model constructor
        if "n_envs" in hyperparams.keys():
            del hyperparams["n_envs"]
        del hyperparams["n_timesteps"]

        if self.algo == "rarl":
            del hyperparams["N_mu"]
            del hyperparams["N_nu"]

        if "frame_stack" in hyperparams.keys():
            self.frame_stack = hyperparams["frame_stack"]
            del hyperparams["frame_stack"]

        # obtain a class object from a wrapper name string in hyperparams and delete the entry
        env_wrapper = get_wrapper_class(hyperparams)
        if "env_wrapper" in hyperparams.keys():
            del hyperparams["env_wrapper"]

        callbacks = get_callback_list(hyperparams)
        if "callback" in hyperparams.keys():
            self.specified_callbacks = hyperparams["callback"]
            del hyperparams["callback"]

        # manage devide
        if "device" in hyperparams.keys():
            self.device = hyperparams["device"]

        return hyperparams, env_wrapper, callbacks


    def _configure_rarl_optimizers(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        for role in ("protagonist", "adversary"):
            role_kwargs_key = f"{role}_kwargs"
            role_policy_kwargs_key = f"{role}_policy_kwargs"
            role_optimizer_key = f"{role}_optimizer"
            role_optimizer_kwargs_key = f"{role}_optimizer_kwargs"

            algo_kwargs = hyperparams.get(role_kwargs_key, {})
            if not isinstance(algo_kwargs, dict):
                algo_kwargs = dict(algo_kwargs)

            policy_kwargs = hyperparams.get(role_policy_kwargs_key, {})
            if not isinstance(policy_kwargs, dict):
                policy_kwargs = dict(policy_kwargs)

            optimizer_name = getattr(self, role_optimizer_key)
            if optimizer_name is None:
                optimizer_name = hyperparams.get(role_optimizer_key)
            if isinstance(optimizer_name, str):
                optimizer_name = optimizer_name.lower()

            optimizer_kwargs = hyperparams.get(role_optimizer_kwargs_key, {})
            if optimizer_kwargs is None:
                optimizer_kwargs = {}
            optimizer_kwargs = dict(optimizer_kwargs)

            cli_optimizer_kwargs = getattr(self, role_optimizer_kwargs_key)
            if cli_optimizer_kwargs:
                optimizer_kwargs.update(cli_optimizer_kwargs)

            lr_override = getattr(self, f"{role}_lr")
            if lr_override is not None:
                algo_kwargs["learning_rate"] = lr_override

            max_grad_norm_override = getattr(self, f"{role}_max_grad_norm")
            if max_grad_norm_override is not None:
                algo_kwargs["max_grad_norm"] = max_grad_norm_override

            vf_coef_override = getattr(self, f"{role}_vf_coef")
            if vf_coef_override is not None:
                algo_kwargs["vf_coef"] = vf_coef_override

            algo_name = str(hyperparams.get(f"{role}_algo", "")).lower()
            if algo_name == "ppo":
                algo_kwargs["optimizer_scope"] = self.optimizer_scope
                algo_kwargs["training_metrics_csv_path"] = os.path.join(self.save_path, "analysis", f"{role}_training_metrics.csv")
                algo_kwargs["optimizer_role"] = role

            if optimizer_name is not None:
                if optimizer_name in {"proposed_qp", "proposed_nog", "proposed_noG".lower()}:
                    optimizer_kwargs.setdefault("optimizer_scope", self.optimizer_scope)
                    optimizer_kwargs.setdefault("qp_normalization", self.qp_normalization)
                    optimizer_kwargs.setdefault("qp_g_alpha", self.qp_g_alpha)
                    optimizer_kwargs.setdefault("max_update_norm", self.max_update_norm)
                    optimizer_kwargs.setdefault("qp_eps", self.qp_eps)
                    optimizer_kwargs.setdefault("diagnostics_csv_path", os.path.join(self.save_path, f"{role}_{optimizer_name}_diagnostics.csv"))
                    optimizer_kwargs.setdefault("role", role)
                if optimizer_name in {"proposed_qp_new", "proposed_nog_new", "proposed_noG_new".lower()}:
                    optimizer_kwargs.setdefault("optimizer_scope", self.optimizer_scope)
                    optimizer_kwargs.setdefault("qp_normalization", self.qp_normalization)
                    optimizer_kwargs.setdefault("qp_alpha", self.qp_alpha)
                    optimizer_kwargs.setdefault("qp_beta_max", self.qp_beta_max)
                    optimizer_kwargs.setdefault("qp_gamma_max", self.qp_gamma_max)
                    optimizer_kwargs.setdefault("qp_max_update_norm", self.max_update_norm)
                    optimizer_kwargs.setdefault("qp_step_grid", self.qp_step_grid)
                    optimizer_kwargs.setdefault("qp_objective", self.qp_objective)
                    optimizer_kwargs.setdefault("qp_accept_rule", self.qp_accept_rule)
                    optimizer_kwargs.setdefault("qp_min_g_contribution", self.qp_min_g_contribution)
                    optimizer_kwargs.setdefault("qp_critic_weight", self.qp_critic_weight)
                    optimizer_kwargs.setdefault("qp_g_sign", self.qp_g_sign)
                    optimizer_kwargs.setdefault("qp_eps", self.qp_eps)
                    optimizer_kwargs.setdefault("diagnostics_csv_path", os.path.join(self.save_path, f"{role}_{optimizer_name}_diagnostics.csv"))
                    optimizer_kwargs.setdefault("role", role)
                if optimizer_name in {"proposed_qp_new_v2", "proposed_nog_new_v2", "proposed_noG_new_v2".lower()}:
                    optimizer_kwargs.setdefault("optimizer_scope", self.optimizer_scope)
                    optimizer_kwargs.setdefault("qp_normalization", self.qp_normalization)
                    optimizer_kwargs.setdefault("qp_fd_eps", self.qp_fd_eps)
                    optimizer_kwargs.setdefault("qp_beta_probe", self.qp_beta_probe)
                    optimizer_kwargs.setdefault("qp_gamma_probe", self.qp_gamma_probe)
                    optimizer_kwargs.setdefault("qp_ridge", self.qp_ridge)
                    optimizer_kwargs.setdefault("qp_actor_weight", self.qp_actor_weight)
                    optimizer_kwargs.setdefault("qp_logstd_weight", self.qp_logstd_weight)
                    optimizer_kwargs.setdefault("qp_critic_weight", self.qp_critic_weight)
                    optimizer_kwargs.setdefault("qp_beta_max", self.qp_beta_max)
                    optimizer_kwargs.setdefault("qp_gamma_max", self.qp_gamma_max)
                    optimizer_kwargs.setdefault("qp_max_update_norm", self.max_update_norm)
                    optimizer_kwargs.setdefault("qp_eps", self.qp_eps)
                    optimizer_kwargs.setdefault("qp_step_solver", self.qp_step_solver)
                    optimizer_kwargs.setdefault("diagnostics_csv_path", os.path.join(self.save_path, f"{role}_{optimizer_name}_diagnostics.csv"))
                    optimizer_kwargs.setdefault("role", role)
                if optimizer_name in {"proposed_qp_rawfg", "proposed_nog_rawfg", "proposed_noG_rawFG".lower()}:
                    optimizer_kwargs.setdefault("optimizer_scope", self.optimizer_scope)
                    optimizer_kwargs.setdefault("qp_fd_eps", self.qp_fd_eps)
                    optimizer_kwargs.setdefault("qp_beta_probe", self.qp_beta_probe)
                    optimizer_kwargs.setdefault("qp_gamma_probe", self.qp_gamma_probe)
                    optimizer_kwargs.setdefault("qp_ridge", self.qp_ridge)
                    optimizer_kwargs.setdefault("qp_actor_weight", self.qp_actor_weight)
                    optimizer_kwargs.setdefault("qp_logstd_weight", self.qp_logstd_weight)
                    optimizer_kwargs.setdefault("qp_critic_weight", self.qp_critic_weight)
                    optimizer_kwargs.setdefault("qp_beta_max", self.qp_beta_max)
                    optimizer_kwargs.setdefault("qp_gamma_max", self.qp_gamma_max)
                    optimizer_kwargs.setdefault("qp_max_update_norm", self.max_update_norm)
                    optimizer_kwargs.setdefault("qp_eps", self.qp_eps)
                    optimizer_kwargs.setdefault("diagnostics_csv_path", os.path.join(self.save_path, f"{role}_{optimizer_name}_diagnostics.csv"))
                    optimizer_kwargs.setdefault("role", role)
                if optimizer_name in {"proposed_qp_perflyap", "proposed_nog_perflyap", "proposed_noG_perfLyap".lower()}:
                    optimizer_kwargs.setdefault("qp_fd_eps", self.qp_fd_eps)
                    optimizer_kwargs.setdefault("qp_beta_probe", self.qp_beta_probe)
                    optimizer_kwargs.setdefault("qp_gamma_probe", self.qp_gamma_probe)
                    optimizer_kwargs.setdefault("qp_ridge", self.qp_ridge)
                    optimizer_kwargs.setdefault("qp_beta_max", self.qp_beta_max)
                    optimizer_kwargs.setdefault("qp_gamma_max", self.qp_gamma_max)
                    optimizer_kwargs.setdefault("qp_max_update_norm", self.max_update_norm)
                    optimizer_kwargs.setdefault("qp_eps", self.qp_eps)
                    optimizer_kwargs.setdefault("diagnostics_csv_path", os.path.join(self.save_path, f"{role}_{optimizer_name}_diagnostics.csv"))
                    optimizer_kwargs.setdefault("role", role)
                policy_kwargs["optimizer_class"] = get_optimizer_class(optimizer_name)
                policy_kwargs["optimizer_kwargs"] = optimizer_kwargs

            hyperparams[role_kwargs_key] = algo_kwargs
            hyperparams[role_policy_kwargs_key] = policy_kwargs

            if role_optimizer_key in hyperparams:
                del hyperparams[role_optimizer_key]
            if role_optimizer_kwargs_key in hyperparams:
                del hyperparams[role_optimizer_kwargs_key]

        return hyperparams
    

    @staticmethod
    def _preprocess_schedules(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess schedules for learning

        Args:
            hyperparams (Dict[str, Any]): parsed hyperparameters

        Returns:
            Dict[str, Any]: hyperparameters with appropriate scheduler
        """
        # create schedules
        for key in ["learning_rate", "clip_range", "clip_range_vf", "delta_std"]:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split("_")
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constant_fn(float(hyperparams[key]))
            else:
                raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
        return hyperparams
    

    def _preprocess_normalization(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess normalization

        Args:
            hyperparams (Dict[str, Any]): parsed hyperparameters

        Returns:
            Dict[str, Any]: hyperparameters with appropriate normalization parameters
        """
        if "normalize" in hyperparams.keys():
            self.normalize = hyperparams["normalize"]

            # Special case, instead of both normalizing
            # both observation and reward, we can normalize one of the two.
            # in that case `hyperparams["normalize"]` is a string
            # that can be evaluated as python,
            # ex: "dict(norm_obs=False, norm_reward=True)"
            if isinstance(self.normalize, str):
                self.normalize_kwargs = eval(self.normalize)
                self.normalize = True

            # Use the same discount factor as for the algorithm
            if "gamma" in hyperparams:
                self.normalize_kwargs["gamma"] = hyperparams["gamma"]

            del hyperparams["normalize"]
        return hyperparams


    def create_callbacks(self):
        """Create callbacks for train and test environment
        """
        self.callbacks = []
        self.protagonist_callbacks = []
        self.adversary_callbacks = []

        if self.save_freq > 0:
            # Account for the number of parallel environments
            self.save_freq = max(self.save_freq // self.n_envs, 1)
            if self.algo == "rarl":
                self.protagonist_callbacks.append(
                    CheckpointCallback(
                        save_freq=self.save_freq,
                        save_path=self.save_path,
                        name_prefix="pro_model",
                        save_vecnormalize=True,
                        verbose=1,
                    )
                )
                self.adversary_callbacks.append(
                    CheckpointCallback(
                        save_freq=self.save_freq,
                        save_path=self.save_path,
                        name_prefix="adv_model",
                        save_vecnormalize=True,
                        verbose=1,
                    )
                )
            else:
                self.callbacks.append(
                    CheckpointCallback(
                        save_freq=self.save_freq,
                        save_path=self.save_path,
                        name_prefix="rl_model",
                        verbose=1,
                    )
                )

        # Create test env if needed, do not normalize reward
        if self.eval_freq > 0 and not self.optimize_hyperparameters:
            # Account for the number of parallel environments
            self.eval_freq = max(self.eval_freq // self.n_envs, 1)

            if self.verbose > 0:
                print("Creating test environment...")

            save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=self.params_path)
            eval_callback = EvalCallback(
                self.create_envs(self.n_eval_envs, eval_env=True),
                callback_on_new_best=save_vec_normalize,
                best_model_save_path=self.save_path,
                n_eval_episodes=self.n_eval_episodes,
                log_path=self.save_path,
                eval_freq=self.eval_freq,
                deterministic=self.deterministic_eval,
            )

            self.callbacks.append(eval_callback)

            if self.algo == "rarl":
                adv_eval_path = os.path.join(self.save_path, "adv_eval")
                adv_best_path = os.path.join(self.save_path, "adv_best")
                adv_eval_callback = AdversarialEvalCallback(
                    self.create_envs(self.n_eval_envs, eval_env=True, with_adversarial_wrapper=True),
                    best_model_save_path=adv_best_path,
                    log_path=adv_eval_path,
                    n_eval_episodes=self.n_eval_episodes,
                    eval_freq=self.eval_freq,
                    deterministic=self.deterministic_eval,
                )
                self.protagonist_callbacks.append(adv_eval_callback)

                if self.control_proxy_eval and self.adv_impact.lower() == "force":
                    control_eval_path = os.path.join(self.save_path, "control_proxy_eval")
                    control_best_path = os.path.join(self.save_path, "control_proxy_best")
                    control_eval_callback = AdversarialEvalCallback(
                        self.create_envs(
                            self.n_eval_envs,
                            eval_env=True,
                            with_adversarial_wrapper=True,
                            adv_impact_override="control",
                        ),
                        best_model_save_path=control_best_path,
                        log_path=control_eval_path,
                        n_eval_episodes=self.n_eval_episodes,
                        eval_freq=self.eval_freq,
                        deterministic=self.deterministic_eval,
                    )
                    self.protagonist_callbacks.append(control_eval_callback)

        if self.algo == "rarl":
            norms_dir = os.path.join(self.save_path, "analysis")
            self.protagonist_callbacks.append(
                ProtagonistEpisodeReturnCallback(
                    csv_path=os.path.join(norms_dir, "protagonist_episode_returns.csv")
                )
            )
            self.protagonist_callbacks.append(
                ParameterNormLoggingCallback(
                    csv_path=os.path.join(norms_dir, "protagonist_param_norms.csv"),
                    agent_name="protagonist",
                )
            )
            self.adversary_callbacks.append(
                ParameterNormLoggingCallback(
                    csv_path=os.path.join(norms_dir, "adversary_param_norms.csv"),
                    agent_name="adversary",
                )
            )


    def create_envs(
        self,
        n_envs: int,
        eval_env: bool = False,
        no_log: bool = False,
        with_adversarial_wrapper: Optional[bool] = None,
        adv_impact_override: Optional[str] = None,
    ) -> VecEnv:
        """
        Create the environment and wrap it if necessary.
        :param n_envs: number of environments in stack
        :param eval_env: Whether is it an environment used for evaluation or not
        :param no_log: Do not log training when doing hyperparameter optim (issue with writing the same file)
        :return: the vectorized environment, with appropriate wrappers
        """
        # do not log eval env (issue with writing the same file)
        log_dir = None if eval_env or no_log else self.save_path

        monitor_kwargs = {}
        # special case for GoalEnvs: log success rate too
        if "Neck" in self.env_id or self.is_robotics_env(self.env_id) or "parking-v0" in self.env_id:
            monitor_kwargs = dict(info_keywords=("is_success",))

        # if adversarial environment, adapt action space by wrapping into adversarial wrapper
        wrapper_kwargs = {}

        wrapper_class = self.env_wrapper
        use_adversarial_wrapper = with_adversarial_wrapper
        if use_adversarial_wrapper is None:
            use_adversarial_wrapper = (not eval_env) and (self.algo == "rarl" or self.adv_env)

        if use_adversarial_wrapper:
            if self.verbose > 0:
                print("Using adversarial environment wrapper...")
            adv_impact = (adv_impact_override or self.adv_impact).lower()
            if adv_impact == "control":
                wrapper_class = AdversarialClassicControlWrapper
                wrapper_kwargs.update(dict(adv_fraction=self.adv_fraction))
                if adv_impact_override == "control" and self.adv_impact.lower() == "force":
                    wrapper_kwargs.update(dict(adv_action_dim=self.adv_force_dim))
                if self.device:
                    wrapper_kwargs.update(dict(device=self.device))
            elif adv_impact == "force": 
                wrapper_class = AdversarialMujocoWrapper
                wrapper_kwargs.update(dict(adv_fraction=self.adv_fraction, index_list=self.adv_index_list, force_dim=self.adv_force_dim))
                if self.device:
                    wrapper_kwargs.update(dict(device=self.device))

        # on most env, SubprocVecEnv does not help and is quite memory hungry, therefore we use DummyVecEnv by default
        env = make_vec_env(
            env_id=self.env_id,
            n_envs=n_envs,
            seed=self.seed,
            env_kwargs=self.env_kwargs,
            monitor_dir=log_dir,
            wrapper_class=wrapper_class,
            vec_env_cls=self.vec_env_class,
            vec_env_kwargs=self.vec_env_kwargs,
            monitor_kwargs=monitor_kwargs,
            wrapper_kwargs=wrapper_kwargs
        )

        # wrap the env into a VecNormalize wrapper if needed and load saved statistics when present
        env = self._maybe_normalize(env, eval_env)

        # optional frame-stacking
        if self.frame_stack is not None:
            n_stack = self.frame_stack
            env = VecFrameStack(env, n_stack)
            if self.verbose > 0:
                print(f"Stacking {n_stack} frames")

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, gym.spaces.Dict):
                # If even one of the keys is a image-space in need of transpose, apply transpose
                # If the image spaces are not consistent (for instance one is channel first,
                # the other channel last), VecTransposeImage will throw an error
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (is_image_space(space) and not is_image_space_channels_first(space))
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space)

            if wrap_with_vectranspose:
                if self.verbose > 0:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env


    def _maybe_normalize(self, env: VecEnv, eval_env: bool) -> VecEnv:
        """
        Wrap the env into a VecNormalize wrapper if needed and load saved statistics when present.
        :param env: environment
        :param eval_env: True if evaluation mode, False otherwise
        :return: normalized environment
        """
        # pretrained model, load normalization
        path_ = os.path.dirname(self.save_path)
        path_ = os.path.join(path_, "vecnormalize.pkl")

        if os.path.exists(path_):
            print("Loading saved VecNormalize stats")
            env = VecNormalize.load(path_, env)
            # deactivate training and reward normalization while evaluating
            if eval_env:
                env.training = False
                env.norm_reward = False

        elif self.normalize:
            # copy to avoid changing default values by reference
            local_normalize_kwargs = self.normalize_kwargs.copy()
            # do not normalize reward for env used for evaluation
            if eval_env:
                if len(local_normalize_kwargs) > 0:
                    local_normalize_kwargs["norm_reward"] = False
                else:
                    local_normalize_kwargs = {"norm_reward": False}

            if self.verbose > 0:
                if len(local_normalize_kwargs) > 0:
                    print(f"Normalization activated: {local_normalize_kwargs}")
                else:
                    print("Normalizing input and reward")
            env = VecNormalize(env, **local_normalize_kwargs)
        return env


    def _preprocess_action_noise(self, hyperparams: Dict[str, Any], env: VecEnv) -> Dict[str, Any]:
        """Preprocesses action noise for exploration

        Args:
            hyperparams (Dict[str, Any]): parsed hyperparameter dict
            sorted_hyperparams (Dict[str, Any]): hyperparameters sorted
            env (VecEnv): environment

        Returns:
            Dict[str, Any]: tidied hyperparameter dict
        """
        # parse noise string - Note: only off-policy algorithms are supported
        if hyperparams.get("noise_type") is not None:
            noise_type = hyperparams["noise_type"].strip()
            noise_std = hyperparams["noise_std"]

            # save for later (hyperparameter optimization)
            self.n_actions = env.action_space.shape[0]

            if "normal" in noise_type:
                hyperparams["action_noise"] = NormalActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            elif "ornstein-uhlenbeck" in noise_type:
                hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            else:
                raise RuntimeError(f'Unknown noise type "{noise_type}"')

            print(f"Applying {noise_type} noise with std: {noise_std}")

            del hyperparams["noise_type"]
            del hyperparams["noise_std"]

        return hyperparams


    def _save_config(self, sorted_hyperparams: Dict[str, Any]) -> None:
        """
        Save unprocessed hyperparameters, this can be used to reproduce the experiment
        :param sorted_hyperparams: dict of sorted hyperparameters 
        """
        # Save hyperparams
        with open(os.path.join(self.params_path, "config.yml"), "w") as f:
            yaml.dump(sorted_hyperparams, f)

        # save command line arguments
        with open(os.path.join(self.params_path, "args.yml"), "w") as f:
            ordered_args = OrderedDict([(key, vars(self.args)[key]) for key in sorted(vars(self.args).keys())])
            yaml.dump(ordered_args, f)

        print(f"Save hyperparameters for reproducing results to: {self.params_path}")


    def learn(self, model: BaseAlgorithm) -> None:
        """ Trains given model
        :param model: an initialized RL model
        """
        kwargs = {}
        outer_correction = None
        if self.algo == "rarl" and self.composite_outer_mode != "none":
            from models.ppo_composite_outer import PPOCompositeOuterDiagnostic

            outer_correction = PPOCompositeOuterDiagnostic(
                output=os.path.join(self.save_path, "analysis", "composite_outer_diagnostics.csv"),
                device=self.device,
                warmup_outer=self.composite_outer_warmup,
                critic_updates=self.composite_critic_updates,
                batch_size=self.composite_batch_size,
                state_batch_size=self.composite_state_batch_size,
                mode=self.composite_outer_mode,
                actor_lr=self.composite_actor_lr,
            )
        if self.log_interval > -1:
            kwargs = {"log_interval": self.log_interval}

        if len(self.callbacks) > 0:
            kwargs["callback"] = self.callbacks

            if self.algo == "rarl": 
                kwargs["callback_protagonist"] = self.callbacks + self.protagonist_callbacks
                kwargs["callback_adversary"] = [callback for callback in self.callbacks if not isinstance(callback, EvalCallback)] + self.adversary_callbacks

        try:
            if self.algo == "rarl": 
                model.learn(
                    self.n_timesteps,
                    N_mu=self.N_mu,
                    N_nu=self.N_nu,
                    adv_delay=self.adv_delay,
                    update_mode=self.rarl_update_mode,
                    clean_warmup=self.clean_warmup,
                    outer_correction=outer_correction,
                    **kwargs,
                )
            else:
                model.learn(self.n_timesteps, **kwargs)
        except KeyboardInterrupt:
            # this allows to save the model when interrupting training
            pass
        finally:
            # Release resources
            try:
                model.env.close()
            except EOFError:
                pass

    
    def _load_pretrained_agent(self, hyperparams: Dict[str, Any], env: VecEnv) -> BaseAlgorithm:
        # continue training
        print("Loading pretrained agent...")
        # policy should not be changed
        if self.algo == "rarl":
            del hyperparams["protagonist_policy"]
            del hyperparams["adversary_policy"]

            if "protagonist_policy_kwargs" in hyperparams.keys():
                del hyperparams["protagonist_policy_kwargs"]
                
            if "adversary_policy_kwargs" in hyperparams.keys():
                del hyperparams["adversary_policy_kwargs"]
        else:
            del hyperparams["policy"]

            if "policy_kwargs" in hyperparams.keys():
                del hyperparams["policy_kwargs"]


        model = ALGOS[self.algo].load(
            os.path.join(self.save_path, self.env_id),
            env=env,
            seed=self.seed,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose,
            **hyperparams,
        )

        replay_buffer_path = os.path.join(os.path.dirname(self.save_path), "replay_buffer.pkl")

        if os.path.exists(replay_buffer_path):
            print("Loading replay buffer...")
            model.load_replay_buffer(replay_buffer_path)
        return model


    def hyperparameters_optimization(self) -> None:
        """Optimize for hyperparameters
        """
        if self.verbose > 0:
            print("Optimizing hyperparameters")

        if self.storage is not None and self.study_name is None:
            warnings.warn(
                f"You passed a remote storage: {self.storage} but no `--study-name`."
                "The study name will be generated by Optuna, make sure to re-use the same study name "
                "when you want to do distributed hyperparameter optimization."
            )

        if self.tensorboard_log is not None:
            warnings.warn("Tensorboard log is deactivated when running hyperparameter optimization")
            self.tensorboard_log = None

        # TODO: eval each hyperparams several times to account for noisy evaluation
        sampler = self._create_sampler(self.sampler)
        pruner = self._create_pruner(self.pruner)

        if self.verbose > 0:
            print(f"Sampler: {self.sampler} - Pruner: {self.pruner}")

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=True,
            direction="maximize",
        )

        try:
            study.optimize(self.objective, n_trials=self.n_opt_trials, n_jobs=self.n_jobs)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)

        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        report_name = (
            f"report_{self.env_id}_{self.n_opt_trials}-trials-{self.n_timesteps}"
            f"-{self.sampler}-{self.pruner}_{int(time.time())}"
        )

        optimization_log_path = self.optimize_hyperparameters_path

        if self.verbose:
            print(f"Writing report to {optimization_log_path}")

        # Write report
        os.makedirs(os.path.dirname(optimization_log_path), exist_ok=True)
        study.trials_dataframe().to_csv(f"{optimization_log_path}.csv")

        # Save python object to inspect/re-use it later
        with open(f"{optimization_log_path}.pkl", "wb+") as f:
            pkl.dump(study, f)

        # Skip plots
        if self.no_optim_plots:
            return

        # Plot optimization result
        try:
            fig1 = plot_optimization_history(study)
            fig2 = plot_param_importances(study)

            fig1.savefig(os.path.join(optimization_log_path, "optimization_history.jpg"))
            fig2.savefig(os.path.join(optimization_log_path, "param_importances.jpg"))

        except (ValueError, ImportError, RuntimeError):
            pass


    def _create_sampler(self, sampler_method: str) -> BaseSampler:
        """Create sampler for hyperparameter optimization

        Args:
            sampler_method (str): sampler name

        Returns:
            BaseSampler: sampler
        """
        # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
        if sampler_method == "random":
            sampler = RandomSampler(seed=self.seed)
        elif sampler_method == "tpe":
            sampler = TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed, multivariate=True)
        elif sampler_method == "skopt":
            # cf https://scikit-optimize.github.io/#skopt.Optimizer
            # GP: gaussian process
            # Gradient boosted regression: GBRT
            sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
        else:
            raise ValueError(f"Unknown sampler: {sampler_method}")
        return sampler


    def _create_pruner(self, pruner_method: str) -> BasePruner:
        """Create pruner for hyperparameter optimization

        Args:
            pruner_method (str): pruner name

        Returns:
            BasePruner: pruner
        """
        if pruner_method == "halving":
            pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif pruner_method == "median":
            pruner = MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_evaluations_opt // 3)
        elif pruner_method == "none":
            # Do not prune
            pruner = MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_evaluations_opt)
        else:
            raise ValueError(f"Unknown pruner: {pruner_method}")
        return pruner


    def save_trained_model(self, model: BaseAlgorithm) -> None:
        """
        Save trained model optionally with its replay buffer and ``VecNormalize`` statistics
        :param model: trained model
        """
        print(f"Saving to {self.save_path}")
        model.save(os.path.join(self.save_path, self.env_id))

        if hasattr(model, "save_replay_buffer") and self.save_replay_buffer:
            print("Saving replay buffer")
            model.save_replay_buffer(os.path.join(self.save_path, "replay_buffer.pkl"))

        if self.normalize:
            # Important: save the running average, for testing the agent we need that normalization
            model.get_vec_normalize_env().save(os.path.join(self.params_path, "vecnormalize.pkl"))


    def objective(self, trial: optuna.Trial) -> float:
        """Objective for hyperparameter optimization

        Args:
            trial (optuna.Trial): optuna trial handler

        Returns:
            float: reward for config
        """
        kwargs = self._hyperparams.copy()

        # Hack to use DDPG/TD3 noise sampler
        trial.n_actions = self.n_actions

        # Sample candidate hyperparameters
        sampled_hyperparams = HYPERPARAMS_SAMPLER[self.algo](trial)

        if "adv_fraction" in sampled_hyperparams.keys():
            if not self.explicit_adv_fraction_override:
                self.adv_fraction = sampled_hyperparams["adv_fraction"]
            del sampled_hyperparams["adv_fraction"]
        if "N_mu" in sampled_hyperparams.keys():
            self.N_mu = sampled_hyperparams["N_mu"]
            del sampled_hyperparams["N_mu"]
        if "N_nu" in sampled_hyperparams.keys():
            self.N_nu = sampled_hyperparams["N_nu"]
            del sampled_hyperparams["N_nu"]

        kwargs.update(sampled_hyperparams)

        # Create environment
        n_envs = self.n_envs
        env = self.create_envs(n_envs, no_log=True)

        # Define model
        model = ALGOS[self.algo](
            env=env,
            tensorboard_log=None,
            # We do not seed the trial
            seed=None,
            verbose=0,
            **kwargs,
        )

        # Create evaluation environment
        eval_env = self.create_envs(n_envs=self.n_eval_envs, eval_env=True)
        
        if self.algo == "rarl": 
            self.n_timesteps = int(500000 / (self.N_mu * model.protagonist.n_steps * model.protagonist.env.num_envs))
            optuna_eval_freq = int((self.n_timesteps * self.N_mu * model.protagonist.n_steps * model.protagonist.env.num_envs) / self.n_evaluations_opt)
        else:
            optuna_eval_freq = int(self.n_timesteps / self.n_evaluations_opt)
        
        print(f"n_timesteps: {self.n_timesteps}")
        print(f"N_mu: {self.N_mu}")
        print(f"N_nu: {self.N_nu}")
        print(f"protagonist n_steps: {model.protagonist.n_steps}")
        print(f"adversary n_steps: {model.adversary.n_steps}")
        print(f"eval_frequency: {optuna_eval_freq}")

        # account for parallel envs
        optuna_eval_freq = max(optuna_eval_freq // self.n_envs, 1)

        # use non-deterministic eval for Atari
        path = None
        if self.optimize_hyperparameters_path is not None:
            path = os.path.join(self.optimize_hyperparameters_path, f"trial_{str(trial.number)}")
        callbacks = get_callback_list({"callback": self.specified_callbacks})
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            best_model_save_path=path,
            log_path=path,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=optuna_eval_freq,
            deterministic=self.deterministic_eval,
        )
        callbacks.append(eval_callback)

        learn_kwargs = {}

        if self.algo == "rarl": 
            learn_kwargs["callback_protagonist"] = callbacks + self.protagonist_callbacks
            learn_kwargs["callback_adversary"] = [callback for callback in callbacks if not isinstance(callback, EvalCallback)] + self.adversary_callbacks

        try:
            if self.algo == "rarl":
                model.learn(
                    self.n_timesteps,
                    N_mu=self.N_mu,
                    N_nu=self.N_nu,
                    adv_delay=self.adv_delay,
                    update_mode=self.rarl_update_mode,
                    clean_warmup=self.clean_warmup,
                    **learn_kwargs,
                )
            else:
                model.learn(self.n_timesteps, callback=callbacks, **learn_kwargs)
            # free memory
            model.env.close()
            eval_env.close()
        except (AssertionError, ValueError) as e:
            # sometimes, random hyperparams can generate NaN -> free memory
            model.env.close()
            eval_env.close()

            # prune hyperparams that generate NaNs
            print(e)
            print("============")
            print("Sampled hyperparams:")
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned()

        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward


    @staticmethod
    def is_atari(env_id: str) -> bool:
        """Check whether environment is Atari environment

        Args:
            env_id (str): environment string description

        Returns:
            bool: True if environment is Atari, False otherwise
        """
        entry_point = gym.spec(env_id).entry_point
        return "AtariEnv" in str(entry_point)

    
    @staticmethod
    def is_robotics_env(env_id: str) -> bool:
        entry_point = gym.spec(env_id).entry_point
        return "gym.envs.robotics" in str(entry_point) or "panda_gym.envs" in str(entry_point)
