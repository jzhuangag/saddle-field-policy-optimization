import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np
import torch
from typing import List

from stable_baselines3.common.utils import obs_as_tensor, get_device
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper, VecEnvStepReturn

# ================================================
#   Define and customize gym environment wrapper
# ================================================


def _compat_reset(env, *args, **kwargs):
    result = env.reset(*args, **kwargs)
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _compat_step(env, action):
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 5:
        return result
    obs, reward, done, info = result
    return obs, reward, done, False, info


class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.

    Taken from RL Baselines3 Zoo 
    <https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/wrappers.py>

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

    def __init__(self, env: Env, reward_offset: float = 0.0, n_successes: int = 1):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset
        self.n_successes = n_successes
        self.current_successes = 0

    def reset(self, *args, **kwargs):
        self.current_successes = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = _compat_step(self.env, action)
        if info.get("is_success", False):
            self.current_successes += 1
        else:
            self.current_successes = 0
        # number of successes in a row
        terminated = terminated or self.current_successes >= self.n_successes
        reward += self.reward_offset
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset


class AdversarialWrapper(gym.Wrapper):
    """
    Adapts the action space of the gym environment for the adversary
    """
    def __init__(self, env: Env, adv_fraction: float = 1.0):
        super(AdversarialWrapper, self).__init__(env)
        self.env = env
        # define adversarial impact
        adv_magnitude = env.action_space.high[0] * adv_fraction
        high_adv = np.ones(env.action_space.shape[0]) * adv_magnitude

        self._adv_action_space = self.convert_gym_space(env.action_space, low_val=-high_adv, high_val=high_adv)
        self._action_space = self.convert_gym_space(env.action_space, low_val=env.action_space.low, high_val=env.action_space.high)
        self._observation_space = self.convert_gym_space(env.observation_space, low_val=env.observation_space.low, high_val=env.observation_space.high)
        

    @property
    def observation_space(self):
        return self._observation_space


    @property
    def action_space(self):
        return self._action_space


    @property
    def adv_action_space(self):
        return self._adv_action_space


    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)


    def step(self, action):
        return _compat_step(self.env, action)

    def render(self):
        self.env.render()


    def set_action_space(self, updated_action_space): 
        self._action_space = updated_action_space


    def set_action_space(self, updated_action_space): 
        self._action_space = updated_action_space


    def convert_gym_space(self, space, low_val, high_val):
        """Converts gym space into appropriate categories

        Args:
            space (gym.spaces): gym space

        Returns:
            gym.spaces: converted gym space
        """
        if isinstance(space, gym.spaces.Box):
            return spaces.box.Box(low=low_val, high=high_val)
        elif isinstance(space, gym.spaces.Discrete):
            return spaces.discrete.Discrete(n=space.n)
        elif isinstance(space, gym.spaces.Tuple):
            return spaces.Tuple([self.convert_gym_space(x) for x in space.spaces])
        else:
            raise NotImplementedError


class NegativeRewardWrapper(gym.RewardWrapper):
    """
    Negates the reward function
    """
    def __init__(self, env):
        super().__init__(env=env)


    def reward(self, rew):
        # modify rew
        return -rew


class NegativeRewardVecEnvWrapper(VecEnvWrapper):
    """
    Negates the reward function for vector environments
    """
    def __init__(self, venv: VecEnv):
        super().__init__(venv=venv)


    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs, self.reward(reward), done, info
    
    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def reset(self) -> np.ndarray:
        return self.venv.reset()
    
    def reward(self, rew):
        # modify rew
        return -rew


class AdversaryRewardWrapper(gym.RewardWrapper):
    """
    Adapts the reward function of the adversary in RARL
    """
    def __init__(self, env):
        super().__init__(env=env)
        self.action_space = env.adv_action_space


    def reward(self, rew):
        # modify rew
        return -rew


class AdversaryRewardVecEnvWrapper(VecEnvWrapper):
    """
    Adapts the reward function of the adversary in RARL for vector environments
    """
    def __init__(self, venv: VecEnv):
        super().__init__(venv=venv, action_space=venv.get_attr("adv_action_space")[0])


    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs, -reward, done, info
    
    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def reset(self) -> np.ndarray:
        return self.venv.reset()


class AdversarialClassicControlWrapper(gym.Wrapper):
    """
    Adapts the action space of the gym environment for the adversary and couples 
    actions from protagonist and adversary during training
    """
    def __init__(self, env: Env, adv_fraction: float = 1.0, device: str = "auto", adv_action_dim: int | None = None):
        super(AdversarialClassicControlWrapper, self).__init__(env)
        self.env = env
        self.adv_fraction = adv_fraction
        # define adversarial impact
        adv_magnitude = env.action_space.high[0] * adv_fraction
        self.adv_action_dim = env.action_space.shape[0] if adv_action_dim is None else int(adv_action_dim)
        self._control_eval_compat = adv_action_dim is not None
        if self._control_eval_compat:
            high_adv = np.ones(self.adv_action_dim, dtype=np.float32)
        else:
            high_adv = np.ones(self.adv_action_dim, dtype=np.float32) * adv_magnitude

        self._adv_action_space = self.convert_gym_space(env.action_space, low_val=-high_adv, high_val=high_adv)
        self._action_space = env.action_space
        self._initial_action_space = env.action_space
        self._observation_space = env.observation_space

        self.operating_mode = None
        self._pro_policy = None
        self._adv_policy = None
        self.adv_strength = 1.0

        self.device = get_device(device)

        self._last_obs = None
        self._last_episode_starts = None
        self._episode_start_mode = None


    @property
    def observation_space(self):
        return self._observation_space


    @property
    def action_space(self):
        return self._action_space
    

    @property
    def initial_action_space(self):
        return self._initial_action_space


    @property
    def adv_action_space(self):
        return self._adv_action_space


    def set_action_space(self, updated_action_space): 
        self._action_space = updated_action_space


    def sample_action(self):
        class pro_adv_action(object):
            def __init__(self, pro_action, adv_action):
                self.pro_action = pro_action
                self.adv_action = adv_action

        return pro_adv_action(self.action_space.sample(), self.adv_action_space.sample())

    def _embed_adv_action(self, adv_action: np.ndarray) -> np.ndarray:
        embedded = np.zeros(self._action_space.shape[0], dtype=np.float32)
        width = min(len(adv_action), embedded.shape[0])
        embedded[:width] = adv_action[:width]
        return embedded

    def _clip_env_action(self, env_action: np.ndarray):
        env_action = np.asarray(env_action, dtype=np.float32)
        if isinstance(self._action_space, gym.spaces.Box):
            clipped_env_action = np.clip(env_action, self._action_space.low, self._action_space.high)
            clip_fraction = float(np.mean(np.abs(clipped_env_action - env_action) > 1e-8))
        else:
            clipped_env_action = env_action
            clip_fraction = 0.0
        return clipped_env_action, clip_fraction

    def _build_telemetry(
        self,
        info,
        obs,
        raw_adv_action=None,
        clipped_adv_action=None,
        protagonist_action=None,
        applied_disturbance=None,
        action_clip_fraction: float = 0.0,
    ):
        info = dict(info)
        raw_adv_action = np.asarray(raw_adv_action, dtype=np.float32) if raw_adv_action is not None else np.zeros(self.adv_action_dim, dtype=np.float32)
        clipped_adv_action = np.asarray(clipped_adv_action, dtype=np.float32) if clipped_adv_action is not None else np.zeros(self.adv_action_dim, dtype=np.float32)
        protagonist_action = np.asarray(protagonist_action, dtype=np.float32) if protagonist_action is not None else np.zeros(self._action_space.shape[0], dtype=np.float32)
        applied_disturbance = np.asarray(applied_disturbance, dtype=np.float32) if applied_disturbance is not None else np.zeros(self._action_space.shape[0], dtype=np.float32)
        obs_array = np.asarray(obs, dtype=np.float32)
        info.update(
            {
                "adv_impact": "control",
                "adv_strength": float(self.adv_strength),
                "protagonist_action_norm": float(np.linalg.norm(protagonist_action)),
                "adversary_action_norm_pre_clip": float(np.linalg.norm(raw_adv_action)),
                "adversary_action_norm_post_clip": float(np.linalg.norm(clipped_adv_action)),
                "applied_disturbance_norm": float(np.linalg.norm(applied_disturbance)),
                "applied_control_perturbation_norm": float(np.linalg.norm(applied_disturbance)),
                "applied_force_norm": 0.0,
                "action_clip_fraction": float(action_clip_fraction),
                "state_norm": float(np.linalg.norm(obs_array)),
                "operating_mode": self.operating_mode,
                "episode_start_mode": self._episode_start_mode,
            }
        )
        return info


    def step(self, action):
        if hasattr(action, '__dict__'):
            raw_adv_action = np.asarray(action.adv_action, dtype=np.float32)
            clipped_adv_action = np.clip(raw_adv_action, self._adv_action_space.low, self._adv_action_space.high)
            scale = self.adv_strength * (self.adv_fraction if self._control_eval_compat else 1.0)
            scaled_adv_action = scale * clipped_adv_action
            embedded_adv_action = self._embed_adv_action(scaled_adv_action)
            protagonist_action = np.asarray(action.pro_action, dtype=np.float32)
            env_action_raw = protagonist_action + embedded_adv_action
            env_action, action_clip_fraction = self._clip_env_action(env_action_raw)
            applied_disturbance = env_action - protagonist_action
            obs, rew, terminated, truncated, info = _compat_step(self.env, env_action)
            self._last_obs = obs
            self._last_episode_starts = terminated or truncated
            info = self._build_telemetry(
                info,
                obs,
                raw_adv_action,
                clipped_adv_action,
                protagonist_action,
                applied_disturbance,
                action_clip_fraction,
            )
            return obs, rew, terminated, truncated, info
        else:
            if not self.operating_mode:
                obs, rew, terminated, truncated, info = _compat_step(self.env, action)
                self._last_obs = obs
                self._last_episode_starts = terminated or truncated
                info = self._build_telemetry(info, obs, protagonist_action=action)
                return obs, rew, terminated, truncated, info

            elif self.operating_mode.lower() == "protagonist":
                with torch.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    if len(obs_tensor.shape) == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    action_sampled  = self._adv_policy._predict(obs_tensor, deterministic=True)

                raw_adv_action = action_sampled.cpu().numpy().squeeze(0)

                # Clip the actions to avoid out of bound error
                if isinstance(self._adv_action_space, gym.spaces.Box):
                    clipped_adv_action = np.clip(raw_adv_action, self._adv_action_space.low, self._adv_action_space.high)
                else:
                    clipped_adv_action = raw_adv_action
                scale = self.adv_strength * (self.adv_fraction if self._control_eval_compat else 1.0)
                scaled_adv_action = scale * clipped_adv_action
                embedded_adv_action = self._embed_adv_action(scaled_adv_action)
                
            elif self.operating_mode.lower() == "adversary":
                with torch.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    if len(obs_tensor.shape) == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    action_sampled = self._pro_policy._predict(obs_tensor, deterministic=True)

                clipped_actions = action_sampled.cpu().numpy()

                # Clip the actions to avoid out of bound error
                if isinstance(self._action_space, gym.spaces.Box):
                    clipped_actions = np.clip(clipped_actions, self._action_space.low, self._action_space.high).squeeze(0)
                raw_adv_action = np.asarray(action, dtype=np.float32)
                if isinstance(self._adv_action_space, gym.spaces.Box):
                    clipped_adv_action = np.clip(raw_adv_action, self._adv_action_space.low, self._adv_action_space.high)
                else:
                    clipped_adv_action = raw_adv_action
                scale = self.adv_strength * (self.adv_fraction if self._control_eval_compat else 1.0)
                scaled_adv_action = scale * clipped_adv_action
                embedded_adv_action = self._embed_adv_action(scaled_adv_action)
                
            else:
                raise ValueError(f"Please choose operating mode either 'protagonist' or 'adversary', not: ", self.operating_mode)

            if self.operating_mode.lower() == "protagonist":
                protagonist_action = np.asarray(action, dtype=np.float32)
                env_action_raw = protagonist_action + embedded_adv_action
            else:
                protagonist_action = np.asarray(clipped_actions, dtype=np.float32)
                env_action_raw = protagonist_action + embedded_adv_action

            env_action, action_clip_fraction = self._clip_env_action(env_action_raw)
            applied_disturbance = env_action - protagonist_action
            obs, rew, terminated, truncated, info = _compat_step(self.env, env_action)
            self._last_obs = obs
            self._last_episode_starts = terminated or truncated
            info = self._build_telemetry(
                info,
                obs,
                raw_adv_action,
                clipped_adv_action,
                protagonist_action,
                applied_disturbance,
                action_clip_fraction,
            )

            return obs, rew, terminated, truncated, info
    

    def reset(self, *args, **kwargs):
        self._last_obs, info = _compat_reset(self.env, *args, **kwargs)
        self._episode_start_mode = self.operating_mode
        return self._last_obs, info


    def convert_gym_space(self, space, low_val, high_val):
        """Converts gym space into appropriate categories

        Args:
            space (gym.spaces): gym space

        Returns:
            gym.spaces: converted gym space
        """
        if isinstance(space, gym.spaces.Box):
            return spaces.box.Box(low=low_val, high=high_val)
        elif isinstance(space, gym.spaces.Discrete):
            return spaces.discrete.Discrete(n=space.n)
        elif isinstance(space, gym.spaces.Tuple):
            return spaces.prodcut.Product([self.convert_gym_space(x) for x in space.spaces])
        else:
            raise NotImplementedError


class AdversarialMujocoWrapper(gym.Wrapper):
    """
    Modeling errors can be viewed as extra forces in the system.
    This wrapper allows the adversary to apply disturbing forces to the system in order
    to counteract the protagonist's goal.
    The wrapper replaces the customized environments defined in: 
    Lerrel Pinto: "Gym environments with adversarial disturbance agents" <https://github.com/lerrel/gym-adv>.
    """
    def __init__(self, env: Env, adv_fraction: float = 1.0, adv_low: float = 1.0, adv_high: float = 1.0, index_list: List[str] = [], force_dim: int = 2, device: str = "auto"):
        super(AdversarialMujocoWrapper, self).__init__(env)
        self.env = env
        self._base_env = env.unwrapped
        self.force_dim = force_dim
        self.adv_fraction = adv_fraction

        # define point of attack
        self._adv_force_bname = index_list
        available_bnames = [self._base_env.model.body(i).name for i in range(self._base_env.model.nbody)]

        try:
            self._adv_body_indicees = [self._base_env.model.body(i).id for i in self._adv_force_bname]
        except Exception as exc:
            raise AttributeError(
                "Environment: %s does not include all body names in list: %s\n Please use body names available from here: %s"
                % (env.spec.id, index_list, available_bnames)
            ) from exc
        
        adv_action_space_dim = force_dim*len(self._adv_body_indicees)
        # ToDo: Normalize action space and multiply by fraction in step to enable "superpowers"
        self._adv_action_space = spaces.Box(-1 * np.ones(adv_action_space_dim, dtype=np.float32), np.ones(adv_action_space_dim, dtype=np.float32))
        self._action_space = env.action_space
        self._initial_action_space = env.action_space
        self._observation_space = env.observation_space

        self.operating_mode = None
        self._pro_policy = None
        self._adv_policy = None
        self.adv_strength = 1.0

        self.device = get_device(device)

        self._last_obs = None
        self._last_episode_starts = None
        self._episode_start_mode = None


    def _apply_adv_to_xfrc(self, adv_act):
        assert adv_act.shape[0] >= self.force_dim
        # get force mask
        new_xfrc = self._base_env.data.xfrc_applied * 0.0
        # apply forces at contact points
        for i, bindex in enumerate(self._adv_body_indicees):
            if self.force_dim == 1: 
                new_xfrc[bindex] = self.adv_fraction * self.adv_strength * np.array([adv_act[i], 0., 0., 0., 0., 0.])
            elif self.force_dim == 2: 
                new_xfrc[bindex] = self.adv_fraction * self.adv_strength * np.array([adv_act[i*2], 0., adv_act[i*2+1], 0., 0., 0.])
            elif self.force_dim == 3: 
                new_xfrc[bindex] = self.adv_fraction * self.adv_strength * np.array([adv_act[i*3], adv_act[i*3+1], adv_act[i*3+2], 0., 0., 0.])
            elif self.force_dim == 4: 
                new_xfrc[bindex] = self.adv_fraction * self.adv_strength * np.array([adv_act[i*4], adv_act[i*4+1], adv_act[i*4+2], adv_act[i*4+3], 0., 0.])
            elif self.force_dim == 5: 
                new_xfrc[bindex] = self.adv_fraction * self.adv_strength * np.array([adv_act[i*5], adv_act[i*5+1], adv_act[i*5+2], adv_act[i*5+3], 0., adv_act[i*5+4]])
            elif self.force_dim == 6: 
                new_xfrc[bindex] = self.adv_fraction * self.adv_strength * np.array([adv_act[i*6], adv_act[i*6+1], adv_act[i*6+2], adv_act[i*6+3], adv_act[i*6+4], adv_act[i*6+5]])
            else: 
                raise ValueError(f"Force dimension must be within [1, 6], not: ", self.force_dim)

        self._base_env.data.xfrc_applied[:] = new_xfrc
        return float(np.linalg.norm(new_xfrc[:, :3].reshape(-1)))


    def _clear_adv_force(self):
        self._base_env.data.xfrc_applied[:] = 0.0


    def _build_telemetry(
        self,
        info,
        obs,
        raw_adv_action=None,
        clipped_adv_action=None,
        protagonist_action=None,
        applied_force_norm: float = 0.0,
    ):
        info = dict(info)
        raw_adv_action = np.asarray(raw_adv_action, dtype=np.float32) if raw_adv_action is not None else np.zeros(self._adv_action_space.shape[0], dtype=np.float32)
        clipped_adv_action = np.asarray(clipped_adv_action, dtype=np.float32) if clipped_adv_action is not None else np.zeros(self._adv_action_space.shape[0], dtype=np.float32)
        protagonist_action = np.asarray(protagonist_action, dtype=np.float32) if protagonist_action is not None else np.zeros(self._action_space.shape[0], dtype=np.float32)
        obs_array = np.asarray(obs, dtype=np.float32)
        info.update(
            {
                "adv_impact": "force",
                "adv_strength": float(self.adv_strength),
                "adversary_action_norm_pre_clip": float(np.linalg.norm(raw_adv_action)),
                "adversary_action_norm_post_clip": float(np.linalg.norm(clipped_adv_action)),
                "protagonist_action": protagonist_action.copy(),
                "adversary_action": clipped_adv_action.copy(),
                "applied_disturbance_norm": float(applied_force_norm),
                "applied_force_norm": float(applied_force_norm),
                "state_norm": float(np.linalg.norm(obs_array)),
                "operating_mode": self.operating_mode,
                "episode_start_mode": self._episode_start_mode,
            }
        )
        return info


    def sample_action(self):
        class pro_adv_action(object):
            def __init__(self, pro_action, adv_action):
                self.pro_action = pro_action
                self.adv_action = adv_action

        return pro_adv_action(self.action_space.sample(), self.adv_action_space.sample())


    def step(self, action):
        if hasattr(action, '__dict__'):
            raw_adv_action = np.asarray(action.adv_action, dtype=np.float32)
            clipped_adv_action = np.clip(raw_adv_action, self._adv_action_space.low, self._adv_action_space.high)
            # apply force to system
            applied_force_norm = self._apply_adv_to_xfrc(clipped_adv_action)
            # perform step
            obs, rew, terminated, truncated, info = _compat_step(self.env, action.pro_action)
            self._last_obs = obs
            self._last_episode_starts = terminated or truncated
            info = self._build_telemetry(
                info, obs, raw_adv_action, clipped_adv_action, action.pro_action, applied_force_norm
            )
            return obs, rew, terminated, truncated, info
        else:
            if not self.operating_mode:
                # no adversarial influence
                self._clear_adv_force()
                obs, rew, terminated, truncated, info = _compat_step(self.env, action)
                self._last_obs = obs
                self._last_episode_starts = terminated or truncated
                info = self._build_telemetry(info, obs)
                return obs, rew, terminated, truncated, info

            elif self.operating_mode.lower() == "protagonist":
                # sample action from adversary
                with torch.no_grad():
                    # convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    if len(obs_tensor.shape) == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    action_sampled  = self._adv_policy._predict(obs_tensor, deterministic=True)

                raw_adv_action = action_sampled.cpu().numpy().squeeze(0)

                # clip the actions to avoid out of bound error
                if isinstance(self._adv_action_space, gym.spaces.Box):
                    clipped_actions = np.clip(raw_adv_action, self._adv_action_space.low, self._adv_action_space.high)
                else:
                    clipped_actions = raw_adv_action
                
                # apply force to system
                applied_force_norm = self._apply_adv_to_xfrc(clipped_actions)
                # perform step
                obs, rew, terminated, truncated, info = _compat_step(self.env, action)

                self._last_obs = obs
                self._last_episode_starts = terminated or truncated
                info = self._build_telemetry(
                    info, obs, raw_adv_action, clipped_actions, action, applied_force_norm
                )

                return obs, rew, terminated, truncated, info
                
            elif self.operating_mode.lower() == "adversary":
                # sample action from protagonist
                with torch.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    if len(obs_tensor.shape) == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    action_sampled = self._pro_policy._predict(obs_tensor, deterministic=True)

                clipped_actions = action_sampled.cpu().numpy()

                # Clip the actions to avoid out of bound error
                if isinstance(self._action_space, gym.spaces.Box):
                    clipped_actions = np.clip(clipped_actions, self._action_space.low, self._action_space.high).squeeze(0)
                raw_adv_action = np.asarray(action, dtype=np.float32)
                if isinstance(self._adv_action_space, gym.spaces.Box):
                    clipped_adv_action = np.clip(raw_adv_action, self._adv_action_space.low, self._adv_action_space.high)
                else:
                    clipped_adv_action = raw_adv_action
                
                # apply force to system
                applied_force_norm = self._apply_adv_to_xfrc(clipped_adv_action)
                # perform step
                obs, rew, terminated, truncated, info = _compat_step(self.env, clipped_actions)

                self._last_obs = obs
                self._last_episode_starts = terminated or truncated
                info = self._build_telemetry(
                    info, obs, raw_adv_action, clipped_adv_action, clipped_actions, applied_force_norm
                )

                return obs, rew, terminated, truncated, info
                
            else:
                raise ValueError(f"Please choose operating mode either 'protagonist' or 'adversary', not: ", self.operating_mode)


    @property
    def observation_space(self):
        return self._observation_space


    @property
    def action_space(self):
        return self._action_space


    @property
    def adv_action_space(self):
        return self._adv_action_space
    

    def set_action_space(self, updated_action_space): 
        self._action_space = updated_action_space
    

    def set_adv_action_space(self, updated_adv_action_space): 
        self._adv_action_space = updated_adv_action_space


    def reset(self, *args, **kwargs):
        self._last_obs, info = _compat_reset(self.env, *args, **kwargs)
        self._episode_start_mode = self.operating_mode
        return self._last_obs, info


class ObsNoiseWrapper(gym.ObservationWrapper):
    """
    Adding Gaussian noise to observations.
    """
    def __init__(self, env: Env, mu: float = 0, std: float = 1):
        super(ObsNoiseWrapper, self).__init__(env)
        self.env = env
        self.mu = mu
        self.std = std

    def observation(self, observation):
        """Apply noise to observation signal

        Args:
            observations (array): observation signal

        Returns:
            array: observation signal with Gaussian noise
        """
        for obs in observation:
            obs += np.random.normal(self.mu, self.std)
        return observation
