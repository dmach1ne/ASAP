"""
TemporalRolloutBuffer: PPO용 커스텀 버퍼
next_observations를 명시적으로 저장하여 temporal consistency를 유지합니다.
"""
from typing import Generator, NamedTuple, Optional, Union
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


class TemporalRolloutBufferSamples(NamedTuple):
    """RolloutBufferSamples에 next_observations 추가"""
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    next_observations: th.Tensor  # 추가


class TemporalRolloutBuffer(RolloutBuffer):
    """
    RolloutBuffer를 확장하여 next_observations를 저장합니다.
    
    PPO에서 temporal smoothness loss를 계산할 때 (obs, next_obs) 쌍이
    shuffle 후에도 유지되도록 보장합니다.
    
    Usage:
        model = PPO(
            "MlpPolicy", 
            env,
            rollout_buffer_class=TemporalRolloutBuffer,
        )
    """
    
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
        """버퍼 초기화 시 next_observations 배열도 초기화"""
        super().reset()
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
        :param next_obs: 다음 observation (temporal smoothness용)
        """
        # 현재 position 저장 (부모 메서드가 pos를 증가시키기 전에)
        current_pos = self.pos
        
        # 부모 클래스의 add 호출
        super().add(obs, action, reward, episode_start, value, log_prob)
        
        # next_observations 저장
        if next_obs is not None:
            if isinstance(self.observation_space, spaces.Discrete):
                next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
            self.next_observations[current_pos] = np.array(next_obs)
    
    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[TemporalRolloutBufferSamples, None, None]:
        """
        배치를 생성합니다. (obs, next_obs) 쌍이 함께 shuffle됩니다.
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
        (obs, next_obs) 쌍이 함께 반환됩니다.
        """
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.next_observations[batch_inds],  # 추가: 같은 인덱스로 next_obs도 가져옴
        )
        return TemporalRolloutBufferSamples(*tuple(map(self.to_torch, data)))
