"""
RolloutBuffer vs TemporalRolloutBuffer 비교 테스트

기존 RolloutBuffer는 shuffle 후 temporal 관계가 깨지지만,
TemporalRolloutBuffer는 (obs, next_obs) 쌍이 유지되는지 확인합니다.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer

from temporal_buffer import TemporalRolloutBuffer, TemporalRolloutBufferSamples


def test_original_rollout_buffer():
    """기존 RolloutBuffer 테스트 - shuffle 후 temporal 관계 유지 안됨"""
    print("=" * 70)
    print("🔴 기존 RolloutBuffer 테스트")
    print("=" * 70)
    
    env = gym.make("Pendulum-v1")
    model = PPO("MlpPolicy", env, n_steps=100, batch_size=32, verbose=0, device="cpu")
    
    # 롤아웃 수집
    model.learn(total_timesteps=100, progress_bar=False)
    
    # 원본 버퍼 저장
    original_obs = model.rollout_buffer.observations.copy()
    print(f"\n[버퍼 정보]")
    print(f"  - 버퍼 크기: {model.rollout_buffer.buffer_size}")
    print(f"  - 배치 크기: {model.batch_size}")
    
    # shuffle 후 temporal 관계 확인
    consecutive_pairs = 0
    non_consecutive_pairs = 0
    
    for rollout_data in model.rollout_buffer.get(model.batch_size):
        batch_obs = rollout_data.observations.cpu().numpy()
        
        # 배치 내에서 연속된 인덱스인지 확인
        for i in range(len(batch_obs) - 1):
            # 원본에서 현재/다음 obs의 위치 찾기
            obs_i = batch_obs[i]
            obs_next = batch_obs[i + 1]
            
            idx_i = None
            idx_next = None
            for j in range(len(original_obs)):
                if np.allclose(original_obs[j].flatten(), obs_i.flatten(), atol=1e-6):
                    idx_i = j
                if np.allclose(original_obs[j].flatten(), obs_next.flatten(), atol=1e-6):
                    idx_next = j
            
            if idx_i is not None and idx_next is not None:
                if idx_next == idx_i + 1:
                    consecutive_pairs += 1
                else:
                    non_consecutive_pairs += 1
    
    total = consecutive_pairs + non_consecutive_pairs
    if total > 0:
        consecutive_ratio = consecutive_pairs / total * 100
        print(f"\n[결과]")
        print(f"  - 연속된 쌍: {consecutive_pairs}")
        print(f"  - 비연속 쌍: {non_consecutive_pairs}")
        print(f"  - 연속 비율: {consecutive_ratio:.1f}%")
        
        if consecutive_ratio < 50:
            print(f"  ❌ observations[i+1]은 실제 다음 시간 스텝이 아닙니다!")
    
    env.close()
    return consecutive_ratio if total > 0 else 0


def test_temporal_rollout_buffer():
    """TemporalRolloutBuffer 테스트 - (obs, next_obs) 쌍 유지"""
    print("\n" + "=" * 70)
    print("🟢 TemporalRolloutBuffer 테스트")
    print("=" * 70)
    
    env = gym.make("Pendulum-v1")
    obs_space = env.observation_space
    action_space = env.action_space
    
    # TemporalRolloutBuffer 생성
    buffer = TemporalRolloutBuffer(
        buffer_size=100,
        observation_space=obs_space,
        action_space=action_space,
        device="cpu",
        n_envs=1,
    )
    
    # 수동으로 데이터 수집
    obs, _ = env.reset()
    stored_pairs = []  # (obs, next_obs) 쌍 저장
    
    for step in range(100):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # 버퍼에 저장
        buffer.add(
            obs=obs.reshape(1, -1),
            action=action.reshape(1, -1),
            reward=np.array([reward]),
            episode_start=np.array([step == 0]),
            value=th.tensor([0.0]),
            log_prob=th.tensor([0.0]),
            next_obs=next_obs.reshape(1, -1),  # next_obs 명시적 저장
        )
        
        # 원본 쌍 저장 (검증용)
        stored_pairs.append((obs.copy(), next_obs.copy()))
        
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    
    # GAE 계산 (필수)
    buffer.compute_returns_and_advantage(
        last_values=th.tensor([0.0]),
        dones=np.array([False])
    )
    
    print(f"\n[버퍼 정보]")
    print(f"  - 버퍼 크기: {buffer.buffer_size}")
    
    # shuffle 후 (obs, next_obs) 쌍 유지 확인
    correct_pairs = 0
    wrong_pairs = 0
    
    for rollout_data in buffer.get(batch_size=32):
        batch_obs = rollout_data.observations.cpu().numpy()
        batch_next_obs = rollout_data.next_observations.cpu().numpy()
        
        for i in range(len(batch_obs)):
            obs_i = batch_obs[i]
            next_obs_i = batch_next_obs[i]
            
            # 원본에서 이 쌍이 실제로 연속된 쌍인지 확인
            found_match = False
            for orig_obs, orig_next in stored_pairs:
                if np.allclose(obs_i.flatten(), orig_obs.flatten(), atol=1e-6) and \
                   np.allclose(next_obs_i.flatten(), orig_next.flatten(), atol=1e-6):
                    found_match = True
                    break
            
            if found_match:
                correct_pairs += 1
            else:
                wrong_pairs += 1
    
    total = correct_pairs + wrong_pairs
    if total > 0:
        correct_ratio = correct_pairs / total * 100
        print(f"\n[결과]")
        print(f"  - 올바른 (obs, next_obs) 쌍: {correct_pairs}")
        print(f"  - 잘못된 쌍: {wrong_pairs}")
        print(f"  - 정확도: {correct_ratio:.1f}%")
        
        if correct_ratio > 99:
            print(f"  ✅ (obs, next_obs) 쌍이 완벽하게 유지됩니다!")
    
    env.close()
    return correct_ratio if total > 0 else 0


def main():
    print("\n" + "🔬 RolloutBuffer vs TemporalRolloutBuffer 비교 테스트")
    print("=" * 70)
    
    # 테스트 실행
    original_ratio = test_original_rollout_buffer()
    temporal_ratio = test_temporal_rollout_buffer()
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("📊 최종 결과 요약")
    print("=" * 70)
    print(f"\n  RolloutBuffer (기존):          {original_ratio:.1f}% 연속")
    print(f"  TemporalRolloutBuffer (신규):  {temporal_ratio:.1f}% 정확")
    
    if original_ratio < 50 and temporal_ratio > 99:
        print(f"\n  🎉 TemporalRolloutBuffer가 temporal consistency를 보장합니다!")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
