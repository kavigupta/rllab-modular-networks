from rllab.envs.mujoco.arm_env import ArmEnv
from vendor.mujoco_models.generalizations.generalize_model import COLORS, ALL_CONDITIONS
from rllab.core.serializable import Serializable

import numpy as np

class ReachEnv(ArmEnv):
    is_push = True
    def __init__(self, reach_block, *args, **kwargs):
        ArmEnv.__init__(self, *args, **kwargs)
        self.reach_block = reach_block
        Serializable.__init__(self, reach_block, *args, **kwargs)

    def cost(self, end_eff, block_locations):
        target_loc = ALL_CONDITIONS[self.condition][self.reach_block]
        return np.linalg.norm(end_eff - target_loc)

    @staticmethod
    def all_envs(*args, **kwargs):
        for color in COLORS:
            yield lambda color=color: ReachEnv(color, *args, **kwargs)

    @property
    def description(self):
        return f"color_reach_{self.reach_block}"
