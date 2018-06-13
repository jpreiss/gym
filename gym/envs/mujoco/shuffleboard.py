import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

ACTIVE = 2.0

class ShuffleboardEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, model_path='shuffleboard.xml', min_radius=0.0, max_radius=0.1):
        self.min_radius = min_radius
        self.max_radius = max_radius
        # penalize control effort less when links are heavier
        # TODO: deal with this by changing actuator strength to match links
        # in mj model, and normalize control range to [-1, 1]
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)
        self.obj_moved = False
        self.puck_init = self.get_body_com("puck")

    def _step(self, a):
        puck = self.get_body_com("puck") - self.get_body_com("target")
        if abs(puck[0]) >= 0.35 or abs(puck[1]) >= 0.95:
            reward = 0
        else:
            reward_dist = -puck[1] ** 2
            if np.linalg.norm(puck - self.puck_init) > 0.01:
                self.obj_moved = True
            reward_vel = - 0.001 * np.square(self.model.data.qvel.flat[:2]).sum()
            reward_ctrl = - 0.01 * np.square(a).sum()
            reward = reward_dist + reward_vel + reward_ctrl + ACTIVE * self.obj_moved
            self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        return ob, reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = -0.4
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.distance *= 1.15

    def reset_model(self):
        # don't want very tight initial angle on elbow
        #joint_angles = self.np_random.uniform(low=[-np.pi, -2.0], high=[np.pi, 2.0]) + self.init_qpos[:2]

        minr = max(self.min_radius, 0.05)
        angle = self.np_random.uniform(-np.pi, np.pi)
        radius = self.np_random.uniform(minr, self.max_radius)
        self.goal = radius * np.array([np.cos(angle), np.sin(angle)])

        # TODO move puck

        #qpos = np.concatenate([joint_angles, self.goal])
        qpos = self.init_qpos
        qpos[2:] += self.np_random.uniform(-0.02, 0.02, size=(2,))
        qvel = self.init_qvel #+ self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        #qvel[-2:] = 0
        self.set_state(qpos, qvel)
        self.obj_moved = False
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("puck"),
            self.get_body_com("target") - self.get_body_com("puck"),
        ])
