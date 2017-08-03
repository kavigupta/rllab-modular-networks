from rllab.envs.mujoco.arm_env import ArmEnv
from rllab.envs.mujoco.arm_env import COLORS, ALL_CONDITIONS
from rllab.core.serializable import Serializable

from numpy.linalg import norm
import numpy as np

class ReachEnv(ArmEnv):
    is_push = True
    ctrl_coeff = 1e-7
    unmoved_coeff = 0.1
    l1_weight = 0.1
    l2_weight = 10
    alpha = 1e-5
    def __init__(self, reach_block, *args, **kwargs):
        ArmEnv.__init__(self, *args, **kwargs)
        self.reach_block = reach_block
        Serializable.__init__(self, reach_block, *args, **kwargs)

    def cost(self, end_eff, block_locations):
        init_locs = ALL_CONDITIONS[self.condition]
        target_loc = init_locs[self.reach_block]
        blocks_unmoved_cost = sum(norm(block_locations[color] - init_locs[color])
                                        for color in init_locs)
        dist2 = norm(end_eff - target_loc) ** 2
        dist_cost = (0.5 * self.l2_weight * dist2) + (self.l1_weight * np.sqrt(self.alpha + dist2))
        return dist_cost + blocks_unmoved_cost * self.unmoved_coeff

    @property
    def task_type_key(self):
        return self.reach_block,

    @staticmethod
    def all_envs(*args, **kwargs):
        for color in COLORS:
            yield lambda color=color: ReachEnv(color, *args, **kwargs)

    @property
    def description(self):
        return "color_reach_{reach_block}".format(reach_block=self.reach_block)
