from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger
from vendor.mujoco_models.generalizations.generalize_model import mujoco_xml, COLORS

from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import numpy as np
import math

class ArmEnv(MujocoEnv, Serializable):

    ORI_IND = 3

    def __init__(self, is_3d, condition, number_links, *args, **kwargs):
        self.is_3d = is_3d
        self.condition = condition
        self.number_links = number_links
        super(ArmEnv, self).__init__(file_path=self.file, *args, **kwargs)
        Serializable.__init__(self, is_3d, *args, **kwargs)

    @property
    def file(self):
        return mujoco_xml(self.number_links, self.is_push, self.is_3d, self.condition)

    def block_locations(self):
        return [self.get_body_com("goal%s" % color) for color in COLORS]

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_com("end").flat
        ] + self.block_locations()).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        end_eff = self.get_body_com("end")
        goal_cost = self.cost(end_eff, {color : self.get_body_com("goal%s" % color) for color in COLORS})
        lb, ub = self.action_bounds
        ctrl_cost = 1e-2 * np.linalg.norm(action / (ub - lb))
        reward = - goal_cost - ctrl_cost
        return Step(self.get_current_obs(), float(reward), not np.isfinite(self._state).all())

    @overrides
    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.model.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))

    def __str__(self):
        return f"{self.number_links}_{self.description}_{self.condition}"
