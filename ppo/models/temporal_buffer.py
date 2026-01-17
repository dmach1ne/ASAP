"""
TemporalRolloutBuffer: PPO용 커스텀 버퍼
next_observations와 prev_observations를 명시적으로 저장하여 temporal consistency를 유지합니다.
"""
from typing import Generator, NamedTuple, Optional, Union
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from copy import deepcopy


class TemporalRolloutBufferSamples(NamedTuple):
    """RolloutBufferSamples에 prev/next_observations 추가"""
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    prev_observations: th.Tensor  # 추가: 이전 observation
    next_observations: th.Tensor  # 추가: 다음 observation


class TemporalRolloutBuffer(RolloutBuffer):
    """
    RolloutBuffer를 확장하여 prev/next_observations를 저장합니다.
    
    PPO에서 temporal smoothness loss를 계산할 때 (prev_obs, obs, next_obs) 쌍이
    shuffle 후에도 유지되도록 보장합니다.
    
    Usage:
        model = PPO(
            "MlpPolicy", 
            env,
            rollout_buffer_class=TemporalRolloutBuffer,
        )
    """
    
    prev_observations: np.ndarray
    next_observations: np.ndarray
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(
            buffer_size, 
            observation_space, 
            action_space, 
            device, 
            gae_lambda, 
            gamma, 
            n_envs
        )
    
    def reset(self) -> None:
        """버퍼 초기화 시 prev/next_observations 배열도 초기화"""
        super().reset()
        self.prev_observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape), 
            dtype=self.observation_space.dtype
        )
        self.next_observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape), 
            dtype=self.observation_space.dtype
        )
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        prev_obs: Optional[np.ndarray] = None,
        next_obs: Optional[np.ndarray] = None,
    ) -> None:
        """
        한 스텝의 데이터를 버퍼에 추가합니다.
        
        :param obs: 현재 observation
        :param action: 선택한 action
        :param reward: 받은 reward
        :param episode_start: episode 시작 여부
        :param value: value function 추정값
        :param log_prob: action의 log probability
        :param prev_obs: 이전 observation (temporal smoothness용)
        :param next_obs: 다음 observation (temporal smoothness용)
        """
        # 현재 position 저장 (부모 메서드가 pos를 증가시키기 전에)
        current_pos = self.pos
        
        # 부모 클래스의 add 호출
        super().add(obs, action, reward, episode_start, value, log_prob)
        
        # prev_observations 저장
        if prev_obs is not None:
            if isinstance(self.observation_space, spaces.Discrete):
                prev_obs = prev_obs.reshape((self.n_envs, *self.obs_shape))
            self.prev_observations[current_pos] = np.array(prev_obs)
        
        # next_observations 저장
        if next_obs is not None:
            if isinstance(self.observation_space, spaces.Discrete):
                next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
            self.next_observations[current_pos] = np.array(next_obs)
    
    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[TemporalRolloutBufferSamples, None, None]:
        """
        배치를 생성합니다. (prev_obs, obs, next_obs) 쌍이 함께 shuffle됩니다.
        """
        assert self.full, "RolloutBuffer must be full before calling get()"
        
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        
        # 데이터 준비 (flat으로 변환)
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "prev_observations",  # 추가
                "next_observations",  # 추가
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # 배치 크기 설정
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> TemporalRolloutBufferSamples:
        """
        인덱스에 해당하는 샘플을 반환합니다.
        (prev_obs, obs, next_obs) 쌍이 함께 반환됩니다.
        """
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.prev_observations[batch_inds],  # 추가
            self.next_observations[batch_inds],  # 추가
        )
        return TemporalRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class TemporalPPO(PPO):
    """
    PPO를 확장하여 collect_rollouts에서 prev/next_obs를 버퍼에 저장합니다.
    TemporalRolloutBuffer와 함께 사용해야 합니다.
    """
    
    def _setup_model(self) -> None:
        """TemporalRolloutBuffer를 기본 버퍼로 설정"""
        if self.rollout_buffer_class is None:
            self.rollout_buffer_class = TemporalRolloutBuffer
        super()._setup_model()
        # prev_obs 추적용
        self._prev_obs = None
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TemporalRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        collect_rollouts를 오버라이드하여 prev/next_obs를 버퍼에 저장합니다.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        
        # prev_obs 초기화 (첫 스텝에서는 현재 obs를 prev로 사용)
        if self._prev_obs is None:
            self._prev_obs = deepcopy(self._last_obs)
        
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            # next_obs 저장 (terminal state일 경우 terminal_observation 사용)
            next_obs_to_store = deepcopy(new_obs)
            for idx, done in enumerate(dones):
                if done and infos[idx].get("terminal_observation") is not None:
                    next_obs_to_store[idx] = infos[idx]["terminal_observation"]

            # prev_obs 저장
            prev_obs_to_store = deepcopy(self._prev_obs)
            # episode 시작 시 prev_obs는 현재 obs와 동일하게 설정
            for idx, is_start in enumerate(self._last_episode_starts):
                if is_start:
                    prev_obs_to_store[idx] = self._last_obs[idx]

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                prev_obs=prev_obs_to_store,  # ← prev_obs 추가
                next_obs=next_obs_to_store,  # ← next_obs 추가
            )
            
            # 상태 업데이트
            self._prev_obs = deepcopy(self._last_obs)  # 현재 obs를 prev로
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True
