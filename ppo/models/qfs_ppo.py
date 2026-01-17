import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import deque
import torch as th
from stable_baselines3 import PPO
from temporal_buffer import TemporalPPO, TemporalRolloutBuffer
import traceback
from typing import Any, ClassVar, Optional, TypeVar, Union
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from gymnasium import spaces
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from stable_baselines3.common.utils import explained_variance
from scipy.fft import fft

# --- 유틸리티 함수 ---

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


class QFSPPO(TemporalPPO):
    """
    QFS Implementation wrapped in QFSPPO class name.
    Uses QFS (MPR + VFC) logic.
    """
    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[str, Any],
        learning_rate: Union[float, Any] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Any] = 0.2,
        clip_range_vf: Optional[Union[float, Any]] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        qfs_lamT = 0.1, # Mapped to VFC Weight (Temporal)
        # Hidden QFS params for basic QFSPPO class
        qfs_lamS = 0.1, # Mapped to MPR Weight (Spatial)
        qfs_sigma = 0.01 # Mapped to MPR Noise
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        # Map parameters to QFS concepts
        self.lam_vfc = qfs_lamT
        self.lam_mpr = qfs_lamS
        self.sigma_s = qfs_sigma

    def train(self) -> None:
        self._train_qfs()

    def _train_qfs(self) -> None:
        """
        Shared training logic implementing QFS (MPR + VFC).
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        smoothnesses, oscillations = [], []
        mpr_losses, vfc_losses = [], []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                try:
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        actions = rollout_data.actions.long().flatten()

                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()
                    advantages = rollout_data.advantages
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)
                    entropy_losses.append(entropy_loss.item())

                    # --- QFS Implementation ---
                    
                    # 1. Mixed-Partial Regularization (MPR) / Spatial
                    # || pi(s) - pi(s + noise) ||^2
                    noise = th.normal(0, self.sigma_s, size=rollout_data.observations.shape, device=self.device)
                    observations_noisy = rollout_data.observations + noise
                    
                    pi_clean = self.policy._predict(rollout_data.observations, deterministic=True)
                    pi_noisy = self.policy._predict(observations_noisy, deterministic=True)
                    
                    mpr_loss = F.mse_loss(pi_clean, pi_noisy)
                    mpr_losses.append(mpr_loss.item())

                    # 2. Vector Field Consistency (VFC) / Temporal
                    # || pi(s_t) - pi(s_{t+1}) ||^2
                    # TemporalRolloutBuffer의 next_observations 사용 (temporal consistency 보장)
                    curr_obs = rollout_data.observations
                    next_obs = rollout_data.next_observations
                    
                    pi_t = self.policy._predict(curr_obs, deterministic=True)
                    pi_t1 = self.policy._predict(next_obs, deterministic=True)
                    
                    vfc_loss = F.mse_loss(pi_t, pi_t1)
                    vfc_losses.append(vfc_loss.item())

                    # Total Loss
                    loss = policy_loss + \
                           self.ent_coef * entropy_loss + \
                           self.vf_coef * value_loss + \
                           self.lam_mpr * mpr_loss + \
                           self.lam_vfc * vfc_loss

                    # Logging Metrics
                    osc = calculate_oscillation(actions)
                    oscillations.append(osc)
                    smoothness = calculate_smoothness(actions)
                    smoothnesses.append(smoothness)

                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                except Exception as e:
                    print("🔥 예외 발생! traceback ↓↓↓")
                    traceback.print_exc()
                    raise e

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        
        # QFS & Smoothness Logs
        self.logger.record("train/smoothness", np.mean(smoothnesses))
        self.logger.record("train/oscillation", np.mean(oscillations))
        self.logger.record("train/mpr_loss", np.mean(mpr_losses))
        self.logger.record("train/vfc_loss", np.mean(vfc_losses))


class QFS_CS_PPO(QFSPPO):
    """
    QFS Implementation wrapped in QFS_CS_PPO class name.
    Explicitly exposes spatial (MPR) and temporal (VFC) weights.
    """
    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[str, Any],
        learning_rate: Union[float, Any] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Any] = 0.2,
        clip_range_vf: Optional[Union[float, Any]] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        qfs_lamT = 0.1,  # VFC Weight
        qfs_lamS = 0.1,  # MPR Weight
        qfs_sigma = 0.01 # MPR Noise (Using QFS recommended scale)
    ):
        super().__init__(
            policy,
            env,
            learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda,
            clip_range, clip_range_vf, normalize_advantage, ent_coef, vf_coef,
            max_grad_norm, use_sde, sde_sample_freq, rollout_buffer_class,
            rollout_buffer_kwargs, target_kl, stats_window_size, tensorboard_log,
            policy_kwargs, verbose, seed, device, _init_setup_model,
            qfs_lamT, qfs_lamS, qfs_sigma
        )


class QFS_LS_PPO(QFSPPO):
    """
    QFS Implementation wrapped in QFS_LS_PPO class name.
    Ignores L2C2 specific parameters (qfs_lamD, qfs_lamU) and uses QFS logic.
    """
    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[str, Any],
        learning_rate: Union[float, Any] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Any] = 0.2,
        clip_range_vf: Optional[Union[float, Any]] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
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
        # We assume a default Spatial weight (MPR) for LS_PPO context if not provided, 
        # but since this class is replacing LS, we'll set a reasonable default for QFS.
        qfs_lamS = 0.1 
        
        super().__init__(
            policy,
            env,
            learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda,
            clip_range, clip_range_vf, normalize_advantage, ent_coef, vf_coef,
            max_grad_norm, use_sde, sde_sample_freq, rollout_buffer_class,
            rollout_buffer_kwargs, target_kl, stats_window_size, tensorboard_log,
            policy_kwargs, verbose, seed, device, _init_setup_model,
            qfs_lamT, qfs_lamS, qfs_sigma
        )
        # Keep these just in case they are accessed externally, though unused in QFS logic
        self.qfs_lamD = qfs_lamD
        self.qfs_lamU = qfs_lamU
