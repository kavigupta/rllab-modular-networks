from argparse import ArgumentParser
import sys
import numpy as np
import subprocess

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.block_env import BlockPushEnv
from rllab.envs.mujoco.reach_env import ReachEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite

import rllab.misc.logger as logger

from rllab.envs.mujoco.arm_env import COLORS, ALL_CONDITIONS

parser = ArgumentParser()
parser.add_argument('batch_sizes', type=str, help='sizes of the batch, in units of 10k')
parser.add_argument('path_lengths', type=str, help='path lengths')
parser.add_argument('step_sizes', type=str, help='step sizes')
args = parser.parse_args(sys.argv[1:])

def run_task(env, max_path_length=150, n_itr=100, step_size=1e-3, batch_size=10**4):
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        max_path_length=max_path_length,
        n_itr=n_itr,
        step_size=step_size,
        batch_size=batch_size
    )
    algo.train()

environments = [(env, dict(batch_size=int(batch_size * 1e4), max_path_length=int(max_path_length), n_itr=100, step_size=float(step_size)))
                    for batch_size in eval(args.batch_sizes)
                    for max_path_length in eval(args.path_lengths)
                    for step_size in eval(args.step_sizes)
                    for cls in (ReachEnv,)
                    for condition in range(len(ALL_CONDITIONS))
                    for num_links in (3, 4, 5)
                    for env in cls.all_envs(is_3d=True, condition=condition, number_links=num_links)]

np.random.shuffle(environments)

already_run = subprocess.run("ls data/local/experiment | cat", shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')

def run_exp(env, **kwargs):
    env = env()
    exp_name = f"hypersearch_{env}_" + "_".join(sorted(f"{k}_{v}" for k, v in kwargs.items()))
    if exp_name in already_run:
        print("Skip %s" % exp_name)
        return
    run_experiment_lite(
        lambda *_: run_task(env, **kwargs),
        n_parallel=10,
        snapshot_mode="last",
        exp_name=exp_name,
        seed=1
    )

for env, kwargs in environments:
    run_exp(env, **kwargs)
