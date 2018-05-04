import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from collections import OrderedDict
from copy import deepcopy

logger = logging.getLogger(__name__)

# Point-mass environment. Mass starts at random location.
# Must push to center.
# Action: force vector (2d)
# State: position, velocity (2d)

class PointMassBatchEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.N = 32
        self.state = np.zeros((self.N, 4))
        self.fixedlen = True

        self.box = 2.0
        self.maxvel = 1000.0
        self.maxgain = 4.0
        self.dt = 1.0 / 50
        self.seconds = 3.0
        self.ep_len = int(self.seconds / self.dt)

        ob_high = np.array([
            self.box,
            self.box,
            self.maxvel,
            self.maxvel,
            self.maxgain
        ])
        self.observation_space = spaces.Box(-ob_high, ob_high)
        self.sysid_dim = 1
        self.sysid_names = ["gain"]

        # act: between -1 and 1, will be scaled by force_mag.
        # extra push for policy to learn SysID of force_mag.
        ac_high = np.ones(2)
        self.action_space = spaces.Box(-ac_high, ac_high)

        self.viewer = None
        self.tick = 0

        self.gain = np.ones((self.N))
        self._seed()
        self.sample_sysid()

    def sysid_values(self):
        return self.gain[:,None]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert action.shape == (self.N,2)
        ac_penalty = 0.0 * np.sum(action ** 2, axis=1)
        action[action > 1] = 1
        action[action < -1] = -1

        state = self.state
        x, dx = self.state[:,:2], self.state[:,2:4]
        force = self.gain[:,None] * action

        damping = 0.998 # lose 10% of velocity in one second
        dx *= damping
        dx += self.dt * force
        x += self.dt * dx

        self.state = np.hstack([x, dx, self.gain[:,None]])

        goaldist = np.linalg.norm(x, axis=1)
        reward = -goaldist - ac_penalty
        # compensate for gain. derived from integrating optimal bang-bang policy
        # reward = -goaldist * np.sqrt(np.abs(self.gain))

        self.tick += 1
        self.done = np.full(self.N, False)
        if self.tick >= self.ep_len:
            self.done = np.full(self.N, True)
            self.tick = 0
            self._reset()

        self.reward = reward

        # after self.reset_done(), all states that were done
        # have already been reset to new states
        return self.state, reward, self.done, {}

    def sample_sysid(self):
        abs_gain = self.np_random.uniform(0.05 * self.maxgain, self.maxgain, size=(self.N))
        sgn_gain = (-1) ** self.np_random.randint(1, 3, size=(self.N))
        self.gain = sgn_gain * abs_gain
        self.state[:,-1] = self.gain

    def _reset(self):
        x = self.np_random.uniform(-self.box, self.box, size=(self.N, 2))
        x[:,0] = 1
        x[:,1] = 1
        dx = 0 * x
        self.state = np.hstack([x, dx, self.gain[:,None]])
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(600, 600)
            lim = 1.5 * self.box
            self.viewer.set_bounds(-lim, lim, -lim, lim)

            goal_rad = self.box / 20.0
            goal = rendering.make_circle(goal_rad)

            mass = rendering.make_circle(goal_rad / 2.0)
            mass.set_color(0.2, 0.2, 1.0)
            self.mass_trans = rendering.Transform()
            mass.add_attr(self.mass_trans)

            self.viewer.add_geom(goal)
            self.viewer.add_geom(mass)

        #if self.state is None: return None

        x = self.state[0,:2]
        self.mass_trans.set_translation(x[0], x[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
