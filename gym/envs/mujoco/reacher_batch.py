import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import ReacherEnv
from gym.envs.mujoco.mujoco_batch import *
import os

xml_path = os.path.join(os.path.dirname(__file__), "assets", "reacher.xml")

def reacher_mutate(np_random, tree, randomness):

    shoulder = tree.find("worldbody/body")
    assert shoulder.attrib["name"] == "body0"
    params = randomize_chain(np_random, shoulder, randomness)
    params.extend(randomize_gear_ratios(np_random, tree, randomness))
    return np.array(params)


class ReacherBatchEnv(MujocoBatchEnv):

    def __init__(self, n_batch, n_total, randomness, tweak=1.0):

        def mutate(np_random, tree):
            sysid = reacher_mutate(np_random, tree, randomness)
            if tweak > 0:
                return reacher_mutate(np_random, tree, tweak)
            return sysid

        ep_len = 50
        super().__init__(ReacherEnv, xml_path, mutate, n_batch, n_total, ep_len)
