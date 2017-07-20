from rllab.envs.mujoco.arm_env import ArmEnv
from vendor.mujoco_models.generalizations.generalize_model import COLORS, ALL_CONDITIONS
from rllab.core.serializable import Serializable

import numpy as np
from numpy.linalg import norm

class BlockPushEnv(ArmEnv):
    is_push = True
    def __init__(self, moving_block, target_block, *args, **kwargs):
        ArmEnv.__init__(self, *args, **kwargs)
        self.moving_block, self.target_block = moving_block, target_block
        Serializable.__init__(self, moving_block, target_block, *args, **kwargs)

    def cost(self, end_eff, block_locations):
        movepos = block_locations[self.moving_block]
        targetpos = block_locations[self.target_block]
        init_locs = ALL_CONDITIONS[self.condition]
        cost_items_adjacent = norm(targetpos - movepos)
        cost_shaping = min(norm(end_eff - movepos), norm(end_eff - targetpos))
        cost_other_blocks_unmoved = sum(norm(block_locations[color] - init_locs[color])
                                        for color in init_locs
                                        if color not in {self.moving_block, self.target_block})
        return cost_shaping + cost_items_adjacent + cost_other_blocks_unmoved

    def all_envs(*args, **kwargs):
        for moving in COLORS:
            for target in COLORS:
                if moving < target:
                    yield lambda moving=moving, target=target: BlockPushEnv(moving, target, *args, **kwargs)

    @property
    def description(self):
        return f"color_push_{self.moving_block}_to_{self.target_block}"
