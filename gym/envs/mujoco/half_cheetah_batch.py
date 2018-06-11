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

def mutate_cheetah_tree(np_random, tree, randomness):

    torso = tree.find("worldbody/body")
    assert torso.attrib["name"] == "torso"

    lenscale = rand_ratio(np_random, randomness)
    torso_geom = named_child(torso, "geom", "torso")
    fn_attr(torso_geom, "fromto", op.mul, lenscale) # depending on being zero-centered
    head_geom = named_child(torso, "geom", "head")
    head_shift = (lenscale / 2.0) - 0.5
    fn_attr(head_geom, "pos", op.add, [head_shift, 0, 0])

    sysid_params = [lenscale]

    for thigh_name in ("bthigh", "fthigh"):
        thigh = named_child(torso, "body", thigh_name)
        fn_attr(thigh, "pos", op.mul, lenscale)
        sysid_params.extend(randomize_chain(np_random, thigh, randomness))

    sysid_params.extend(randomize_gear_ratios(np_random, tree, randomness))

    return np.array(sysid_params)


class HalfCheetahBatchEnv(MujocoBatchEnv):

    # tweak: construct the envs with the normal amount of randomness
    # using the same seed, but then slightly modify them
    def __init__(self, n_batch, n_total, randomness, tweak=1.0):

        print("1/2-cheetah with randomness", randomness)

        def mutate(np_random, tree):
            prng_state = np_random.get_state()
            assert prng_state[0] == "MT19937"
            #print("mutating w/ state",
                #hash(prng_state[1].tostring() + bytes(prng_state[2])))
            #print("randomness {}, tweak {}".format(randomness, tweak))
            mutate_cheetah_tree(np_random, tree, randomness)
            return mutate_cheetah_tree(np_random, tree, tweak)

        ep_len = 196
        super().__init__(HalfCheetahEnv, xml_path,
            mutate, n_batch, n_total, ep_len)