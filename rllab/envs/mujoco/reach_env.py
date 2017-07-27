from rllab.envs.mujoco.arm_env import ArmEnv
from rllab.envs.mujoco.arm_env import COLORS, ALL_CONDITIONS
from rllab.core.serializable import Serializable

from numpy.linalg import norm

class ReachEnv(ArmEnv):
    is_push = True
    ctrl_coeff = 1e-7
    unmoved_coeff = 0.1
    def __init__(self, reach_block, *args, **kwargs):
        ArmEnv.__init__(self, *args, **kwargs)
        self.reach_block = reach_block
        Serializable.__init__(self, reach_block, *args, **kwargs)

    def cost(self, end_eff, block_locations):
        init_locs = ALL_CONDITIONS[self.condition]
        target_loc = init_locs[self.reach_block]
        blocks_unmoved_cost = sum(norm(block_locations[color] - init_locs[color])
                                        for color in init_locs)
        dist = norm(end_eff - target_loc)
        return dist + blocks_unmoved_cost * self.unmoved_coeff

    @staticmethod
    def all_envs(*args, **kwargs):
        for color in COLORS:
            yield lambda color=color: ReachEnv(color, *args, **kwargs)

    @property
    def description(self):
        return f"color_reach_{self.reach_block}"
