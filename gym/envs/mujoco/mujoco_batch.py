import gym
from gym import utils

import numpy as np

from copy import deepcopy
import itertools as it


class MujocoBatchEnv(gym.Env):

    def __init__(self, *args):
        self.init_args = args

    # rand_gen should be an infinite generator function
    # that takes an np_random object
    # and yields (scalar Mujoco env, sysid vector) pairs

    def _init_after_seed(self, n_parallel, n_total, rand_gen, ep_len):
        self.N = n_parallel
        self.N_RAND = n_total

        self.tick = 0
        self.ep_len = ep_len

        # construct a bunch of randomized models
        envs_all, sysids_all = zip(
            *it.islice(rand_gen(self.np_random), self.N_RAND))
        assert len(sysids_all[0].shape) == 1
        self.sysid_dim = sysids_all[0].shape[0]
        self.envs_all = np.array(envs_all)
        self.sysids_all = np.row_stack(sysids_all)
        print("sysid dimension:", self.sysids_all[0].shape)

        env0 = self.envs_all[0]
        self.obs_dim = env0.observation_space.low.shape
        self.action_space = env0.action_space

        def expand_obs_by(obs, n):
            infs = np.full(n, np.inf)
            low = np.concatenate([obs.low, -infs])
            high = np.concatenate([obs.high, infs])
            return gym.spaces.Box(low, high)

        self.observation_space = expand_obs_by(
            env0.observation_space, self.sysid_dim)
        self.envs = None
        self.sample_sysid()

        # rendering stuff
        self.metadata = deepcopy(env0.metadata)

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self._init_after_seed(*self.init_args)
        return [seed]

    def sample_sysid(self):
        prev_env0 = None
        if self.envs is not None:
            prev_env0 = self.envs[0]
        selection = self.np_random.choice(self.N_RAND, self.N)
        self.envs = self.envs_all[selection]
        self.sysid = self.sysids_all[selection,:]
        assert self.sysid.shape == (self.N, self.sysid_dim)
        if prev_env0 is not None and prev_env0.viewer is not None:
            self.envs[0]._take_viewer(prev_env0)

    def sysid_values(self):
        return deepcopy(self.sysid)

    def _step(self, a):
        obs = np.zeros([self.N] + list(self.obs_dim))
        rewards = np.zeros(self.N)
        # TODO merge reward dicts
        for i in range(self.N):
            ob, reward, done, rew_dict = self.envs[i]._step(a[i,:])
            obs[i] = ob
            rewards[i] = reward

        self.tick += 1
        done = self.tick >= self.ep_len
        dones = np.full(self.N, done)
        if done:
            self.tick = 0
            for env in self.envs:
                env.reset()

        obs = np.concatenate([obs, self.sysid], axis=1)

        return obs, rewards, dones, None

    def _render(self, mode='human', close=False):
        self.envs[0].render(mode=mode, close=close)

    def _reset(self):
        for env in self.envs:
            env.reset()
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros([self.N] + list(self.obs_dim))
        for i in range(self.N):
            obs[i] = self.envs[i]._get_obs()
        return np.concatenate([obs, self.sysid], 1)
