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

TRPO_INDEX = 0

parser = ArgumentParser()
parser.add_argument('batch_size', type=str,
                    help='size of the batch, in units of 10k')
args = parser.parse_args(sys.argv[1:])

def run_task(env):
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        max_path_length=300,
        n_itr=100,
        step_size=1e-2,
        batch_size=int(float(args.batch_size) * 10**4)
    )
    algo.train()

environments = [env
                    for cls in (BlockPushEnv, ReachEnv)
                    for condition in range(len(ALL_CONDITIONS))
                    for num_links in (3, 4, 5)
                    for env in cls.all_envs(is_3d=True, condition=condition, number_links=num_links)]

np.random.shuffle(environments)

already_run = subprocess.run("ls data/local/experiment | cat", shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')

def run_exp(env):
    env = env()
    exp_name = f"trpo_{TRPO_INDEX}_{env}_{args.batch_size}"
    if exp_name in already_run:
        print("Skip %s" % exp_name)
        return
    run_experiment_lite(
        lambda *_: run_task(env),
        n_parallel=10,
        snapshot_mode="last",
        exp_name=exp_name,
        seed=1
    )

for env in environments:
    run_exp(env)
