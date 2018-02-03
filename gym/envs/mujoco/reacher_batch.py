import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.reacher_xml import ReacherXML

class ReacherBatchEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.batch = False
        self.N = 1
        self.sysid_dim = 4
        utils.EzPickle.__init__(self)
        self.tick = 0
        self.xml_randomizer = ReacherXML()
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
        self.sample_sysid()

    def sysid_values(self):
        vals = self.xml_randomizer.sysid_values()
        return vals[None,:] if self.batch else vals

    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 1.00 * reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        self.tick += 1
        done = self.tick >= 64
        if done:
            self.reset()
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        self.tick = 0
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        lo, hi = self.xml_randomizer.min_rad, self.xml_randomizer.max_rad
        while True:
            self.goal = self.np_random.uniform(-hi, hi, size=2)
            dist = np.linalg.norm(self.goal)
            if lo < dist and dist < hi:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def sample_sysid(self):
        if self.tick != 0:
            print("sampling sysid with tick =", self.tick)
            assert False
        self.xml_randomizer.randomize(self.np_random)
        path = self.xml_randomizer.get_path()
        mujoco_env.MujocoEnv.__init__(self, path, 2, clear_viewer=False)
        print("randomized: new sysid", self.xml_randomizer.sysid_values())
        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer_setup()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        vals = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target"),
            self.sysid_values().flatten()
        ])
        return vals[None,:] if self.batch else vals
