import os, sys
import argparse
import difflib
import gymnasium as gym
import numpy as np
import torch 

from stable_baselines3.common.utils import set_random_seed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from models.algorithms import ALGOS
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict

# ===========================================================================================
#   Training an RL agent on an environment specified. Extends RL Baselines3 Zoo training 
#   script for training the Robust Aversarial RL agent (RARL).
#   To promote standardization, the script is as close to RL Baselines3 Zoo as possible.
#   (https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/train.py)   
# ===========================================================================================

if __name__ == "__main__": 
    # Fetch CLI arguments
    parser = argparse.ArgumentParser("Robust Adversarial RL (RARL)")

    # General configs
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1], help="Verbose mode (0: no output, 1: INFO)")
    parser.add_argument("--seed", type=int, default=-1, help="Random generator seed")
    parser.add_argument("--num-exps", type=int, default=1, help="Number of experiments")
    parser.add_argument("--num-threads", type=int, default=-1, help="Number of threads for PyTorch")

    # Configs about environment
    parser.add_argument('--env', type=str, default="BipedalWalker-v3", help='Name of gym environment')
    parser.add_argument("--n-envs", type=int, default=1, help="Number of environments for stack")
    parser.add_argument("--vec-env-type", type=str, default="dummy", choices=["dummy", "subproc"], help="VecEnv type")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor")
    parser.add_argument("--adv-env", action="store_true", default=False, help="Adversarial gym environment")

    # Configs about model
    parser.add_argument("--algo", type=str, default="ppo", choices=list(ALGOS.keys()), help="RL Algorithm")
    parser.add_argument(
        "--rarl-config",
        type=str,
        default="trpo",
        choices=["trpo", "ppo"],
        help="Select which RARL hyperparameter family to load when --algo rarl is used",
    )
    parser.add_argument("--saved-models-path", type=str, default=os.path.join("saved_models"), help="Path to a saved models")
    parser.add_argument("--pretrained-model", type=str, default="", help="Path to a pretrained agent to continue training")
    parser.add_argument("--save-replay-buffer", default=False, action="store_true", help="Save the replay buffer (when applicable)")

    # Configs about hyperparameter
    parser.add_argument("-params", "--hyperparameter", type=str, nargs="+", action=StoreDict, help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)")
    parser.add_argument("-optimize", "--optimize-hyperparameters", action="store_true", default=False, help="Run hyperparameters search")
    parser.add_argument("--hyperparameter-path", type=str, default=os.path.join("hyperparameter"), help="Path to a saved hyperparameters")
    parser.add_argument("--storage", type=str, default=None, help="Database storage path if distributed optimization should be used")
    parser.add_argument("--study-name", type=str, default=None, help="Study name for distributed optimization")
    parser.add_argument("--sampler", type=str, default="tpe", choices=["random", "tpe", "skopt"], help="Sampler to use when optimizing hyperparameters") 
    parser.add_argument("--pruner", type=str, default="median", choices=["halving", "median", "none"], help="Pruner to use when optimizing hyperparameters")
    parser.add_argument("--optimization-log-path", type=str, default=os.path.join("hyperparam_optimization"), help="Path to save the evaluation log and optimal policy for "
                                                                                                                        "each hyperparameter tried during optimization.")
    parser.add_argument("--n-opt-trials", type=int, default=10, help="Number of trials for optimizing hyperparameters.")
    parser.add_argument("--no-optim_plots", action="store_true", default=False, help="Disable hyperparameter optimization plots")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs when optimizing hyperparameters")
    parser.add_argument("--n-startup-trials", type=int, default=10, help="Number of trials before using optuna sampler")
    parser.add_argument("--n-evaluations-opt", type=int, default=20, help="Training policies are evaluated every n-timesteps during hyperparameter optimization")

    # Configs about training
    parser.add_argument("-n", "--n-timesteps", type=int, default=-1, help="Number of timesteps")
    parser.add_argument("--save-freq", type=int, default=-1, help="Save model every k steps (if negative, no checkpoint)")
    parser.add_argument("--log-interval", type=int, default=-1, help="Override log interval (default: -1, no change)")
    parser.add_argument("--device", type=str, default="cpu", help="Specify device")

    # Configs about evaluation
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluate the agent every e steps (if negative, no evaluation).")
    parser.add_argument("--n-eval-envs", type=int, default=1, help="Number of environments for evaluation")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="Number of episodes to use for evaluation")
    parser.add_argument("--control-proxy-eval", action="store_true", default=False, help="Record an additional control-proxy adversarial eval curve during force-trained RARL runs")

    # Configs about logging results 
    parser.add_argument("-tb", "--tensorboard-log", type=str, default=os.path.join("tb_logging"), help="Tensorboard log dir")
    parser.add_argument("-f", "--log-folder", type=str, default=os.path.join("logging"), help="Log folder")

    # Configs for RARL
    parser.add_argument('--protagonist-policy', type=str, default="MlpPolicy", help='Policy of protagonist')
    parser.add_argument('--adversary-policy', type=str, default="MlpPolicy", help='Policy of adversary')
    optimizer_choices = [
        "adam",
        "sgd",
        "egm",
        "ppm",
        "proposed_qp",
        "proposed_noG",
        "proposed_qp_new",
        "proposed_noG_new",
        "proposed_qp_new_v2",
        "proposed_noG_new_v2",
        "proposed_qp_rawFG",
        "proposed_noG_rawFG",
        "proposed_qp_perfLyap",
        "proposed_noG_perfLyap",
    ]
    parser.add_argument('--protagonist-optimizer', type=str, default=None, choices=optimizer_choices, help='Optimizer used by protagonist policy')
    parser.add_argument('--adversary-optimizer', type=str, default=None, choices=optimizer_choices, help='Optimizer used by adversary policy')
    parser.add_argument('--protagonist-optimizer-kwargs', type=str, nargs="+", action=StoreDict, help='Optional keyword arguments for protagonist optimizer')
    parser.add_argument('--adversary-optimizer-kwargs', type=str, nargs="+", action=StoreDict, help='Optional keyword arguments for adversary optimizer')
    parser.add_argument('--protagonist-lr', type=float, default=None, help='Optional learning-rate override for protagonist sub-algorithm')
    parser.add_argument('--adversary-lr', type=float, default=None, help='Optional learning-rate override for adversary sub-algorithm')
    parser.add_argument('--protagonist-max-grad-norm', type=float, default=None, help='Optional max_grad_norm override for protagonist sub-algorithm')
    parser.add_argument('--adversary-max-grad-norm', type=float, default=None, help='Optional max_grad_norm override for adversary sub-algorithm')
    parser.add_argument('--protagonist-vf-coef', type=float, default=None, help='Optional vf_coef override for protagonist sub-algorithm')
    parser.add_argument('--adversary-vf-coef', type=float, default=None, help='Optional vf_coef override for adversary sub-algorithm')
    parser.add_argument('--optimizer-scope', type=str, default="full_policy", choices=["full_policy", "actor_game"], help='Scope of manual PPO optimizers such as proposed_qp')
    parser.add_argument('--qp-normalization', type=str, default="none", choices=["none", "global", "block"], help='Normalization mode for proposed_qp directions')
    parser.add_argument('--qp-g-alpha', type=float, default=1e-3, help='Finite-difference alpha for proposed_qp G direction')
    parser.add_argument('--max-update-norm', type=float, default=float("inf"), help='Optional update norm cap for proposed_qp')
    parser.add_argument('--qp-eps', type=float, default=1e-8, help='Numerical epsilon used by proposed_qp')
    parser.add_argument('--qp-alpha', type=float, default=0.3, help='Half-step scale used to build the proposed_*_new correction direction')
    parser.add_argument('--qp-beta-max', type=float, default=1.0, help='Maximum beta candidate for proposed_*_new')
    parser.add_argument('--qp-gamma-max', type=float, default=1.0, help='Maximum gamma candidate for proposed_*_new')
    parser.add_argument('--qp-step-grid', type=str, default='0,0.1,0.3,1.0,3.0', help='Comma-separated candidate step grid for proposed_*_new beta/gamma search')
    parser.add_argument('--qp-objective', type=str, default='loss', choices=['loss', 'field_energy', 'mixed'], help='Same-minibatch objective used by proposed_*_new candidate selection')
    parser.add_argument('--qp-accept-rule', type=str, default='none', choices=['none', 'same_minibatch_loss'], help='Acceptance safeguard for proposed_*_new')
    parser.add_argument('--qp-min-g-contribution', type=float, default=0.0, help='Minimum G contribution ratio required for proposed_*_new gamma candidates')
    parser.add_argument('--qp-critic-weight', type=float, default=1.0, help='Explicit critic block weight used by proposed_*_new')
    parser.add_argument('--qp-g-sign', type=str, default='plus', choices=['plus', 'minus', 'auto_probe'], help='Sign convention for proposed_*_new G_raw')
    parser.add_argument('--qp-fd-eps', type=float, default=1e-3, help='Finite-difference step used by proposed_*_new_v2 to estimate G = J_F f')
    parser.add_argument('--qp-beta-probe', type=float, default=1e-3, help='Finite-difference probe scale for Lyapunov drift along -f')
    parser.add_argument('--qp-gamma-probe', type=float, default=1e-3, help='Finite-difference probe scale for Lyapunov drift along +g')
    parser.add_argument('--qp-ridge', type=float, default=1e-8, help='Ridge added to unstable Lyapunov quadratic coefficients in proposed_*_new_v2')
    parser.add_argument('--qp-actor-weight', type=float, default=1.0, help='Lyapunov block weight for actor parameters in proposed_*_new_v2')
    parser.add_argument('--qp-logstd-weight', type=float, default=1.0, help='Lyapunov block weight for log_std parameters in proposed_*_new_v2')
    parser.add_argument('--qp-step-solver', type=str, default='lyapunov_quadratic_bound', choices=['lyapunov_quadratic_bound', 'loss_grid_debug'], help='Step solver for proposed_*_new_v2; main method must use lyapunov_quadratic_bound')
    parser.add_argument('--N-mu', type=int, default=-1, help='Number of steps to run for each environment per protagonist update')
    parser.add_argument('--N-nu', type=int, default=-1, help='Number of steps to run for each environment per adversary update')

    # Configs for adversarial environment
    parser.add_argument('--adv-impact', type=str, default="control", choices=["control", "force"], help='Define how adversary impacts agent')
    parser.add_argument('--adv-delay', type=int, default=-1, help='Delay of adversary')
    parser.add_argument(
        '--rarl-update-mode',
        type=str,
        default='alternating',
        choices=['alternating', 'synchronized_lagged'],
        help='Alternating RARL or pseudo-joint lagged-opponent updates',
    )
    parser.add_argument(
        '--clean-warmup',
        action='store_true',
        default=False,
        help='Use an exact zero-force adversary before adv-delay expires',
    )
    parser.add_argument('--composite-outer-mode', choices=['none', 'diagnostic', 'gda'], default='none')
    parser.add_argument('--composite-outer-warmup', type=int, default=10)
    parser.add_argument('--composite-critic-updates', type=int, default=100)
    parser.add_argument('--composite-batch-size', type=int, default=256)
    parser.add_argument('--composite-state-batch-size', type=int, default=256)
    parser.add_argument('--composite-actor-lr', type=float, default=1e-6)
    parser.add_argument('--adv-fraction', type=float, default=1.0, help='Force-scaling for adversary')
    parser.add_argument('--adv-index-list', nargs='+', type=str, default=["torso"], help='Contact point for adversarial forces (for Mujoco environments)')
    parser.add_argument('--adv-force-dim', type=int, default=2, help='Dimension of adversarial force')


    args = parser.parse_args()
    # Preserve an explicit CLI force-scale override instead of silently replacing
    # it with the environment YAML value during hyperparameter preprocessing.
    args.adv_fraction_explicit = any(
        token == "--adv-fraction" or token.startswith("--adv-fraction=") for token in sys.argv[1:]
    )

    # iterate for number of experiments
    for exp in range(args.num_exps):
        # check gym environment
        env_id = args.env
        registered_envs = set(gym.registry.keys())

        # if environment is not found, suggest the closest match
        if env_id not in registered_envs:
            try:
                closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
            except IndexError:
                closest_match = "'no close environment match found...'"
            raise ValueError(f"{env_id} not found in gym registry, did you maybe mean {closest_match}?")
        
        # set random seed
        if args.seed < 0:
            args.seed = np.random.randint(2**32 - 1, dtype="int64").item()
        
        set_random_seed(args.seed)

        # set number of threads 
        if args.num_threads > 0: 
            if args.verbose == 1: 
                print(f"Setting torch.num_threads to {args.num_threads}")
            torch.set_num_threads(args.num_threads)
        
        algo_folder = args.algo.lower()
        if args.algo.lower() == "rarl":
            algo_folder = f"rarl-{args.rarl_config.lower()}"

        model_path = os.path.join(args.saved_models_path, algo_folder, env_id)

        # check path
        if args.pretrained_model != "":
            assert os.path.isfile(os.path.join(model_path, args.pretrained_model, env_id + ".zip")), f"The pretrained_agent must be a valid path to a .zip file. Check path to {os.path.join(model_path, args.pretrained_model)}"
        
        # print information
        print("=" * 10, env_id, "=" * 10)
        print(f"Seed: {args.seed}")

        # experiment manager
        exp_manager = ExperimentManager(
            args, 
            algo=args.algo,
            env_id=env_id,
            log_folder=args.log_folder,
            tensorboard_log=args.tensorboard_log,
            n_timesteps=args.n_timesteps,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            save_freq=args.save_freq,
            hyperparameter_path=args.hyperparameter_path,
            hyperparams=args.hyperparameter,
            env_kwargs=args.env_kwargs,
            model_path=model_path,
            pretrained_model = args.pretrained_model,
            optimize_hyperparameters=args.optimize_hyperparameters,
            storage=args.storage,
            study_name=args.study_name,
            n_opt_trials=args.n_opt_trials,
            n_jobs=args.n_jobs,
            sampler=args.sampler,
            pruner=args.pruner,
            optimization_log_path=args.optimization_log_path,
            n_startup_trials=args.n_startup_trials,
            n_evaluations_opt=args.n_evaluations_opt,
            seed=args.seed,
            log_interval=args.log_interval,
            save_replay_buffer=args.save_replay_buffer,
            verbose=args.verbose,
            vec_env_type=args.vec_env_type,
            n_envs=args.n_envs,
            n_eval_envs=args.n_eval_envs,
            no_optim_plots=args.no_optim_plots,
            adv_env=args.adv_env, 
            adv_impact=args.adv_impact,
            adv_fraction=args.adv_fraction,
            adv_delay=args.adv_delay,
            rarl_update_mode=args.rarl_update_mode,
            clean_warmup=args.clean_warmup,
            composite_outer_mode=args.composite_outer_mode,
            composite_outer_warmup=args.composite_outer_warmup,
            composite_critic_updates=args.composite_critic_updates,
            composite_batch_size=args.composite_batch_size,
            composite_state_batch_size=args.composite_state_batch_size,
            composite_actor_lr=args.composite_actor_lr,
            adv_index_list=args.adv_index_list,
            adv_force_dim=args.adv_force_dim,
            N_mu=args.N_mu,
            N_nu=args.N_nu,
            device=args.device,
            rarl_config=args.rarl_config,
            protagonist_optimizer=args.protagonist_optimizer,
            adversary_optimizer=args.adversary_optimizer,
            protagonist_optimizer_kwargs=args.protagonist_optimizer_kwargs,
            adversary_optimizer_kwargs=args.adversary_optimizer_kwargs,
            protagonist_lr=args.protagonist_lr,
            adversary_lr=args.adversary_lr,
            protagonist_max_grad_norm=args.protagonist_max_grad_norm,
            adversary_max_grad_norm=args.adversary_max_grad_norm,
            protagonist_vf_coef=args.protagonist_vf_coef,
            adversary_vf_coef=args.adversary_vf_coef,
            optimizer_scope=args.optimizer_scope,
            qp_normalization=args.qp_normalization,
            qp_g_alpha=args.qp_g_alpha,
            max_update_norm=args.max_update_norm,
            qp_eps=args.qp_eps,
            qp_alpha=args.qp_alpha,
            qp_beta_max=args.qp_beta_max,
            qp_gamma_max=args.qp_gamma_max,
            qp_step_grid=args.qp_step_grid,
            qp_objective=args.qp_objective,
            qp_accept_rule=args.qp_accept_rule,
            qp_min_g_contribution=args.qp_min_g_contribution,
            qp_critic_weight=args.qp_critic_weight,
            qp_g_sign=args.qp_g_sign,
            qp_fd_eps=args.qp_fd_eps,
            qp_beta_probe=args.qp_beta_probe,
            qp_gamma_probe=args.qp_gamma_probe,
            qp_ridge=args.qp_ridge,
            qp_actor_weight=args.qp_actor_weight,
            qp_logstd_weight=args.qp_logstd_weight,
            qp_step_solver=args.qp_step_solver,
            control_proxy_eval=args.control_proxy_eval,
        )

        # Prepare experiment and launch hyperparameter optimization if needed
        model = exp_manager.setup_experiment()

        # Normal training
        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)
        else:
            exp_manager.hyperparameters_optimization()
        
        # increase seed
        args.seed += 1
