from argparse import ArgumentParser
import sys
import numpy as np

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

for env in environments:
    env = env()
    run_experiment_lite(
        lambda *_: run_task(env),
        n_parallel=10,
        snapshot_mode="last",
        exp_name=f"{env}_{args.batch_size}",
        seed=1
    )
