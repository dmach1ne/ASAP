"""
QFS 하이퍼파라미터 Sweep 테스트
3개의 하이퍼파라미터(qfs_lamT, qfs_lamS, qfs_sigma)를 sweep합니다.

Usage:
    python test_qfs_sweep.py --envs pendulum lunar --lamt 0.01 0.1 --lams 0.1 0.5 --sigma 0.01 0.1 --num 3 --cnum 4
"""
import sys
import gymnasium as gym
import os
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import argparse
import signal
from torch import nn
import itertools

from stable_baselines3.common.env_util import make_vec_env

from modules.action_extractor import test_some_path
from modules.controller import train_qfs_sweep
from modules.envs import (
    make_ant_env, make_hopper_env, make_humanoid_env, 
    make_lunar_env, make_pendulum_env, make_reacher_env, make_walker_env
)
from modules.params import env_timestep, env_args

train_envs_dict = dict({
    "ant": make_ant_env,
    "hopper": make_hopper_env,
    "humanoid": make_humanoid_env,
    "lunar": make_lunar_env,
    "pendulum": make_pendulum_env,
    "reacher": make_reacher_env,
    "walker": make_walker_env
})

save_dir_root = f"./ppo/results/pths/"
logs_dir_root = f"./ppo/results/tensorboard_logs/"
seed_file_path = "./ppo/tests/seeds.txt"
result_dir_root = f"./ppo/results/"


def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def mk_param_list_sweep(lam_t_list, lam_s_list, sigma_list):
    """
    3개 하이퍼파라미터의 모든 조합을 생성합니다.
    
    Returns:
        param_list: 하이퍼파라미터 딕셔너리 리스트
        pth_list: 저장 경로명 리스트
    """
    param_list = []
    pth_list = []
    
    for lam_t, lam_s, sigma in itertools.product(lam_t_list, lam_s_list, sigma_list):
        param_list.append(dict(
            qfs_lamT=lam_t,
            qfs_lamS=lam_s,
            qfs_sigma=sigma
        ))
        pth_list.append(f"qfs_ppo_lamT{lam_t}_lamS{lam_s}_sigma{sigma}_")
    
    return param_list, pth_list


## 학습 시작
parser = argparse.ArgumentParser(description="QFS 하이퍼파라미터 Sweep 테스트")
parser.add_argument(
    "--envs",
    nargs="+",
    choices=["ant", "hopper", "lunar", "pendulum", "reacher", "walker"],
    default=["pendulum"],
    help="List of environments to train on."
)
parser.add_argument(
    "--lamt",
    nargs="+",
    type=float,
    default=[0.01, 0.05, 0.1],
    help="List of qfs_lamT (VFC/Temporal weight) values. 예: --lamt 0.01 0.05 0.1"
)
parser.add_argument(
    "--lams",
    nargs="+",
    type=float,
    default=[0.05, 0.1, 0.3],
    help="List of qfs_lamS (MPR/Spatial weight) values. 예: --lams 0.1 0.3 0.5"
)
parser.add_argument(
    "--sigma",
    nargs="+",
    type=float,
    default=[0.01, 0.05, 0.1],
    help="List of qfs_sigma (MPR noise scale) values. 예: --sigma 0.01 0.05 0.1"
)
parser.add_argument(
    "--num",
    type=int,
    default=5,
    help="Number of seeds to test for each parameter combination. 예: --num 5"
)
parser.add_argument(
    "--cnum",
    type=int,
    default=4,
    help="Maximum number of concurrent workers. 예: --cnum 4"
)

args = parser.parse_args()

# 하이퍼파라미터 조합 생성
test_params, pth_list = mk_param_list_sweep(args.lamt, args.lams, args.sigma)

print("=" * 60)
print("QFS Hyperparameter Sweep")
print("=" * 60)
print(f"Environments: {args.envs}")
print(f"qfs_lamT (VFC): {args.lamt}")
print(f"qfs_lamS (MPR): {args.lams}")
print(f"qfs_sigma: {args.sigma}")
print(f"Total combinations: {len(test_params)}")
print(f"Seeds per combination: {args.num}")
print(f"Concurrent workers: {args.cnum}")
print(f"Total jobs: {len(args.envs) * len(test_params) * args.num}")
print("=" * 60)

# 파라미터 조합 출력
print("\nParameter combinations:")
for i, (param, pth) in enumerate(zip(test_params, pth_list)):
    print(f"  {i+1}. lamT={param['qfs_lamT']}, lamS={param['qfs_lamS']}, sigma={param['qfs_sigma']}")

train_envs = args.envs
max_num = args.num
max_concurrent_num = args.cnum

# Seed 로드
seeds = load_seeds(seed_file_path)
if len(seeds) < max_num:
    raise ValueError(f"Not enough seeds in {seed_file_path}: required {max_num}, found {len(seeds)}")

# Job 생성
jobs = []
for env_name in train_envs:
    save_dir = os.path.join(save_dir_root, env_name) + '/'
    log_dir = os.path.join(logs_dir_root, env_name) + '/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        time.sleep(1)

    for num in range(max_num):
        seed = seeds[num]
        for test_param in test_params:
            jobs.append(
                partial(
                    train_qfs_sweep,
                    seed, 
                    env_timestep[env_name], 
                    save_dir, 
                    log_dir, 
                    train_envs_dict[env_name], 
                    env_args[env_name], 
                    test_param
                )
            )

print(f"\nStarting {len(jobs)} jobs with {max_concurrent_num} workers...")

# 병렬 실행
with ProcessPoolExecutor(max_workers=max_concurrent_num) as executor:
    future_to_job = {executor.submit(job): job for job in jobs}
    
    completed = 0
    try:
        for future in as_completed(future_to_job):
            job_func = future_to_job[future]
            completed += 1
            try:
                future.result(timeout=60*60*24)  # 24-hour timeout
                print(f"[{completed}/{len(jobs)}] Job completed successfully.")
            except Exception as exc:
                print(f"[{completed}/{len(jobs)}] Job generated an exception: {exc}")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cancelling all jobs...")
        for future in future_to_job:
            future.cancel()
        executor.shutdown(wait=False)
        raise
    finally:
        print("Cleaning up processes...")
        executor.shutdown(wait=False)

# 결과 분석
print("\n" + "=" * 60)
print("Testing completed! Analyzing results...")
print("=" * 60)

subpath = f"qfs_sweep_{args.envs[0]}/"
test_some_path(save_dir_root, True, pth_list, subpath)
