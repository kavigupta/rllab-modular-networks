
from os.path import exists
from os import system

import numpy as np
import matplotlib.pyplot as plt

import subprocess
from pickle import load

def folder(env):
    return f"data/local/experiment/trpo_{env}_4"
def params_path(env):
    return f"{folder(env)}/params.pkl"
def log_path(env):
    return f"{folder(env)}/debug.log"
def rollout_path(env):
    return f"{folder(env)}/rollout.pkl"
def total_train_benefit(env):
    shell = f"cat {log_path(env)} | grep AverageR | tail -1"
    last_average_r = subprocess.run(shell, stdout=subprocess.PIPE, shell=True).stdout.decode()
    try:
        return float(last_average_r.split(" ")[-1])
    except ValueError:
        if last_average_r == "":
            return -float('inf')
        raise RuntimeError(f"Bad line: {last_average_r}, produced by command: {shell}")
def good_environments(environments, plot=True, cutoff=-20):
    extant_envs = [env for env in environments if exists(log_path(env()))]
    train_benefits = [total_train_benefit(env()) for env in extant_envs]
    if plot:
        plt.hist(train_benefits, color="black", bins=25)
        plt.axvline(cutoff, color="red")
        plt.show()
    return [env for env, train in zip(extant_envs, train_benefits) if train > cutoff]

def run_sim_policy(env):
    if not exists(rollout_path(env)):
        system(f"python scripts/sim_policy.py {params_path(env)} --rollout {rollout_path(env)}")
    with open(rollout_path(env), "rb") as f:
        return load(f)

def get_components(n_angles, rollout, images):
    for obs, act, image in zip(rollout["observations"], rollout["actions"], images):
        block_locations = obs[-12:]
        robot_end_effector = obs[-15:-12]
        number_qpos = len(obs[:-15]) // 2
        joint_angles = np.concatenate([obs[:n_angles], obs[number_qpos:number_qpos+n_angles]])
        yield block_locations, robot_end_effector, joint_angles, image, act, images[-1]

def get_rollouts(env):
    rollout, images = run_sim_policy(env)
    return list(np.array(list(x)) for x in zip(*get_components(env.n_joints, rollout, images)))
