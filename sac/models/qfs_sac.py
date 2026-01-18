
import sys
import os
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from gymnasium import spaces
from typing import Any, ClassVar, Optional, TypeVar, Union, NamedTuple
import warnings

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.sac.policies import SACPolicy

# Import CustomSAC (Assuming it is in the same directory as this file)
from custom_sac import CustomSAC

# --- Utility Functions ---

def calculate_smoothness(actions):
    """
    주어진 액션 시퀀스의 스무스니스 지표를 계산합니다.
    """
    if actions.dim() == 1:
        actions = actions.unsqueeze(1)

    n = actions.size(0)
    if n < 2:
        return 0.0

    fs = 1.0
    yf = th.fft.fft(actions, dim=0)
    yf = th.abs(yf[:n // 2])
    freqs = th.fft.fftfreq(n, d=1 / fs)[:n // 2].to(yf.device).unsqueeze(1)
    smoothness = 2 * th.sum(freqs * yf, dim=0) / (n * fs)
    return smoothness.mean().item()

def calculate_oscillation(actions):
    action_n = actions[1:].float()
    action_p = actions[:-1].float()
    return th.mean(th.abs(action_n-action_p)).item()


# --- GradBuffer (Reused from grad_sac.py) ---
try:
    import psutil
except ImportError:
    psutil = None

class GradBuffer_Samples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    prev_observations: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

class GradBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.
    Stores prev/next observations for temporal consistency.
    """
    observations: np.ndarray
    prev_observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        self.buffer_size = max(buffer_size // n_envs, 1)

        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
            self.prev_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )
            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes
                total_memory_usage += self.prev_observations.nbytes

            if total_memory_usage > mem_available:
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        prev_obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
            prev_obs = prev_obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
            self.observations[(self.pos - 1) % self.buffer_size] = np.array(prev_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)
            self.prev_observations[self.pos] = np.array(prev_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> GradBuffer_Samples:
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> GradBuffer_Samples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
            prev_obs = self._normalize_obs(self.observations[(batch_inds - 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
            prev_obs = self._normalize_obs(self.prev_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            prev_obs,
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return GradBuffer_Samples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        if dtype == np.float64:
            return np.float32
        return dtype


# --- QFS SAC Class ---

class QFSSAC(CustomSAC):
    """
    QFS Implementation for SAC (Soft Actor-Critic).
    Implements QFS (MPR + VFC) logic using Policy Output difference (consistent with QFS PPO).
    """
    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[GradBuffer]] = None, # Will be set to GradBuffer in setup
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        qfs_lamT = 0.1, # VFC (Temporal) weight
        qfs_lamS = 0.1, # MPR (Spatial) weight
        qfs_sigma = 0.01 # MPR noise scale
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.lam_mpr = qfs_lamS
        self.lam_vfc = qfs_lamT
        self.sigma_s = qfs_sigma

    def _setup_model(self):
        self.replay_buffer_class = GradBuffer
        return super()._setup_model()

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        mpr_losses, vfc_losses = [], []
        smoothnesses, oscillations = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer with GradBuffer_Samples (includes prev/next obs)
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # --- QFS Regularization (Calculate BEFORE Critic Update) ---
            # Paper: "We propose two regularizers for the CRITIC optimization"
            # 1. MPR: || ∇a Q(s, a+e) - ∇a Q(s, a) ||^2
            # 2. VFC: || ∇a Q(s_t, a) - ∇a Q(s_{t+1}, a) ||^2
            
            # Prepare inputs for gradient calculation
            # We need gradients w.r.t measures actions.
            
            obs_grad = replay_data.observations.detach() # No grad for obs
            act_grad = replay_data.actions.clone().detach().requires_grad_(True)
            
            # 1. MPR (Spatial)
            # -------------------------------------------------------
            # L_MPR = E [ || ∇a Q(s, a+e) - ∇a Q(s, a) ||^2 ]
            # -------------------------------------------------------
            noise = th.normal(0, self.sigma_s, size=act_grad.shape, device=self.device)
            act_noisy = act_grad + noise
            
            # Use Q1 for regularization (common practice)
            q1_clean = self.critic.q1_forward(obs_grad, act_grad)
            q1_noisy = self.critic.q1_forward(obs_grad, act_noisy)
            
            # Compute Gradients ∇a Q
            # create_graph=True because we want to minimize this gradient difference w.r.t critic weights (Theta)
            grad_q1_clean = th.autograd.grad(outputs=q1_clean.sum(), inputs=act_grad, create_graph=True)[0]
            grad_q1_noisy = th.autograd.grad(outputs=q1_noisy.sum(), inputs=act_noisy, create_graph=True)[0]
            
            mpr_loss = F.mse_loss(grad_q1_clean, grad_q1_noisy)
            mpr_losses.append(mpr_loss.item())
            
            # 2. VFC (Temporal)
            # -------------------------------------------------------
            # L_VFC = E [ || ∇a Q(s_t, a) - ∇a Q(s_{t+1}, a) ||^2 ]
            # -------------------------------------------------------
            # We need next_observations from buffer
            next_obs_grad = replay_data.next_observations.detach()
            
            # Reuse grad_q1_clean (∇a Q(s_t, a_t))
            
            # Compute ∇a Q(s_{t+1}, a_t)
            # Note: We use the SAME action a_t for both s_t and s_{t+1} to measure vector field consistency 
            # (how the gradient field changes as state moves, evaluating at same action)
            
            # Input: next_obs, current_action (act_grad)
            q1_next = self.critic.q1_forward(next_obs_grad, act_grad)
            
            grad_q1_next = th.autograd.grad(outputs=q1_next.sum(), inputs=act_grad, create_graph=True)[0]
            
            vfc_loss = F.mse_loss(grad_q1_clean, grad_q1_next)
            vfc_losses.append(vfc_loss.item())
            
            # Total QFS Loss
            qfs_loss = self.lam_mpr * mpr_loss + self.lam_vfc * vfc_loss

            # --- Critic Update (TD Loss + QFS Loss) ---
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_td_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            
            # Final Critic Loss = TD Loss + QFS Loss
            critic_loss = critic_td_loss + qfs_loss
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # --- Actor Update ---
            # Note: QFS regularization is applied to the CRITIC, not the Actor.
            
            # Standard SAC Actor Loss
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            
            # Calculate metrics for logging
            with th.no_grad():
                osc = calculate_oscillation(replay_data.actions)
                oscillations.append(osc)
                smoothness = calculate_smoothness(replay_data.actions)
                smoothnesses.append(smoothness)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/mpr_loss", np.mean(mpr_losses))
        self.logger.record("train/vfc_loss", np.mean(vfc_losses))
        self.logger.record("train/smoothness", np.mean(smoothnesses))
        self.logger.record("train/oscillation", np.mean(oscillations))
        
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _store_transition(
        self,
        replay_buffer: GradBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        Reused from GradSAC logic to store prev/next observations.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            self._last2_original_obs = self._last2_obs
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        next_obs = deepcopy(new_obs_)
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            self._last2_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
        )
        self._last2_obs = self._last_obs
        self._last_obs = new_obs
        if self._vec_normalize_env is not None:
            self._last2_original_obs = self._last_original_obs
            self._last_original_obs = new_obs_

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> tuple[int, BaseCallback]:
        total_timesteps, callback = super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        if reset_num_timesteps or self._last2_obs is None:
            assert self.env is not None
            if self._last_obs is not None:
                self._last2_obs = deepcopy(self._last_obs)
            if self._last_original_obs is not None:
                self._last2_original_obs = deepcopy(self._last_original_obs)

        return total_timesteps, callback


class QFS_CS_SAC(QFSSAC):
    """
    QFS Implementation wrapped in QFS_CS_SAC class name.
    Explicitly exposes spatial (MPR) and temporal (VFC) weights.
    """
    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[GradBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        qfs_lamT = 0.1,  # VFC Weight
        qfs_lamS = 0.1,  # MPR Weight
        qfs_sigma = 0.01 # MPR Noise
    ):
        super().__init__(
            policy, env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma,
            train_freq, gradient_steps, action_noise, replay_buffer_class, replay_buffer_kwargs,
            optimize_memory_usage, ent_coef, target_update_interval, target_entropy, use_sde,
            sde_sample_freq, use_sde_at_warmup, stats_window_size, tensorboard_log, policy_kwargs,
            verbose, seed, device, _init_setup_model,
            qfs_lamT, qfs_lamS, qfs_sigma
        )


class QFS_LS_SAC(QFSSAC):
    """
    QFS Implementation wrapped in QFS_LS_SAC class name.
    Ignores L2C2 specific parameters (qfs_lamD, qfs_lamU) and uses QFS logic.
    """
    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[GradBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        qfs_lamT = 0.1,    # VFC Weight
        qfs_sigma = 0.01,  # MPR Noise
        qfs_lamD = 0.01,   # Ignored in QFS
        qfs_lamU = 1.0,    # Ignored in QFS
    ):
        # We assume a default Spatial weight (MPR) for LS_SAC context
        qfs_lamS = 0.1 
        
        super().__init__(
            policy, env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma,
            train_freq, gradient_steps, action_noise, replay_buffer_class, replay_buffer_kwargs,
            optimize_memory_usage, ent_coef, target_update_interval, target_entropy, use_sde,
            sde_sample_freq, use_sde_at_warmup, stats_window_size, tensorboard_log, policy_kwargs,
            verbose, seed, device, _init_setup_model,
            qfs_lamT, qfs_lamS, qfs_sigma
        )
        self.qfs_lamD = qfs_lamD
        self.qfs_lamU = qfs_lamU
