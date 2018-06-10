import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import AntEnv
from gym.envs.mujoco.mujoco_batch import *

from copy import deepcopy
import tempfile
import os
import sys
import operator as op


xml_path = os.path.join(os.path.dirname(__file__), "assets", "ant.xml")

RANDOMNESS = 1.3

def mutate_ant_tree(np_random, tree):

    torso = tree.find("worldbody/body")
    assert torso.attrib["name"] == "torso"

    sysid_params = []
    for origin in torso.findall("body"):
        bodies = origin.findall("body")
        assert len(bodies) == 1
        leg = bodies[0]
        sysid_params.extend(randomize_chain(np_random, leg, RANDOMNESS))

    sysid_params.extend(randomize_gear_ratios(np_random, tree, RANDOMNESS))
    return np.array(sysid_params)


class AntBatchEnv(MujocoBatchEnv):

    def __init__(self, N=64):
        total_envs = 256
        ep_len = 196
        super().__init__(AntEnv, xml_path,
            mutate_ant_tree, N, total_envs, ep_len)
