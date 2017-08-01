from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger
from vendor.mujoco_models.generalizations.generalize_model import mujoco_xml

from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import numpy as np
import math
from itertools import permutations

COLORS = "red", "green", "yellow", "blue"
ALL_BLOCK_LOCATIONS = [[-0.4 + 0.7 * np.cos(theta), 0, -0.2 + 0.7 * np.sin(theta)] for theta in (-1.5, -1, -0.5, 0.5, 1, 1.5)]
ALL_CONDITIONS = [dict(zip(COLORS, x)) for x in permutations(ALL_BLOCK_LOCATIONS, 4)]

class ArmEnv(MujocoEnv, Serializable):

    ORI_IND = 3
    ctrl_coeff = 1e-2
    image_list = None

    def __init__(self, is_3d, condition, number_links, *args, **kwargs):
        self.is_3d = is_3d
        self.condition = condition
        self.number_links = number_links
        super(ArmEnv, self).__init__(file_path=self.file, *args, **kwargs)
        Serializable.__init__(self, is_3d, *args, **kwargs)

    @property
    def file(self):
        return mujoco_xml(self.number_links, self.is_push, self.is_3d)

    @property
    def n_joints(self):
        return self.is_3d + self.number_links

    @property
    def block_locations(self):
        return [self.block_location_dict[color] for color in COLORS]
    @property
    def block_location_dict(self):
        return {color : self.get_body_com("goal%s" % color) + ALL_CONDITIONS[self.condition][color] for color in COLORS}

    @property
    def tensorcloud_key(self):
        return self.task_type_key + (self.number_links,)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_com("end").flat
        ] + self.block_locations).reshape(-1)

    def step(self, action):
        if self.image_list is not None:
            self.image_list += [self.get_image()]
        self.forward_dynamics(action)
        end_eff = self.get_body_com("end")
        goal_cost = self.cost(end_eff, self.block_location_dict)
        lb, ub = self.action_bounds
        ctrl_cost = self.ctrl_coeff * np.linalg.norm(action / (ub - lb))
        reward = - goal_cost - ctrl_cost
        return Step(self.get_current_obs(), float(reward), not np.isfinite(self._state).all())

    def get_image(self):
        data, width, height = self.get_viewer().get_image()
        return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:]

    @overrides
    def reset_mujoco(self, init_state=None):
        geom_pos = np.array(self.model.geom_pos)
        geom_pos[-4:] = [ALL_CONDITIONS[self.condition][color] for color in COLORS]
        self.model.geom_pos = geom_pos

    @overrides
    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.model.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

    def setup_camera(self):
        self.image_list = []
        self.get_viewer().cam.distance = 3.5
        self.get_viewer().cam.elevation = 0

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
