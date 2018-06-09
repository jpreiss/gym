import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import HalfCheetahEnv
from gym.envs.mujoco.mujoco_batch import *

from copy import deepcopy
import tempfile
import os
import sys
import operator as op


xml_path = os.path.join(os.path.dirname(__file__), "assets", "half_cheetah.xml")


RANDOMNESS = 1.5
TORSO_LEN_RATIO = RANDOMNESS
GEAR_RATIO = RANDOMNESS


def mutate_cheetah_tree(np_random, tree):
    npr = np_random

    torso = tree.find("worldbody/body")
    assert torso.attrib["name"] == "torso"

    # TODO: randomize head? probably doesn't matter much,
    # except it influcences succeptibility to flip-overs

    lenscale = rand_ratio(npr, TORSO_LEN_RATIO)
    torso_geom = named_child(torso, "geom", "torso")
    fn_attr(torso_geom, "fromto", op.mul, lenscale) # depending on being zero-centered
    head_geom = named_child(torso, "geom", "head")
    head_shift = (lenscale / 2.0) - 0.5
    fn_attr(head_geom, "pos", op.add, [head_shift, 0, 0])

    sysid_params = [lenscale]

    for thigh_name in ("bthigh", "fthigh"):
        thigh = named_child(torso, "body", thigh_name)
        fn_attr(thigh, "pos", op.mul, lenscale)
        sysid_params.extend(randomize_chain(np_random, thigh, RANDOMNESS))

    randomize_gear_ratios(npr, tree, GEAR_RATIO)

    return np.array(sysid_params)


class HalfCheetahBatchEnv(MujocoBatchEnv):

    def __init__(self, N=64):
        total_envs = 256
        ep_len = 196
        super().__init__(HalfCheetahEnv, xml_path,
            mutate_cheetah_tree, N, total_envs, ep_len)
