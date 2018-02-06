import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, model_path='reacher.xml', min_radius=0.0, max_radius=0.2):
        self.min_radius = min_radius
        self.max_radius = max_radius
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)

    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_vel = - 0.0010 * np.square(self.model.data.qvel.flat[:2]).sum()
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_vel + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-1.0, high=1.0, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-self.max_radius, high=self.max_radius, size=2)
            dist = np.linalg.norm(self.goal)
            if self.min_radius < dist and dist < self.max_radius:
                break
        #self.goal = np.array([-0.1, 0.1])
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
