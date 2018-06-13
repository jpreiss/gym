import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import ShuffleboardEnv
from gym.envs.mujoco.mujoco_batch import *
import os

xml_path = os.path.join(os.path.dirname(__file__), "assets", "shuffleboard.xml")

def reacher_mutate(np_random, tree, randomness):

    shoulder = named_child(tree, "worldbody/body", "body0")
    params = randomize_chain(np_random, shoulder, randomness)
    params.extend(randomize_gear_ratios(np_random, tree, randomness))

    puck = named_child(tree, "worldbody/body", "puck")
    joints = puck.findall("joint")
    #damping = np_random.uniform(0.1, 1.0)
    #for j in joints:
        #j.attrib["damping"] = str(damping)

    return np.array(params)


class ShuffleboardBatchEnv(MujocoBatchEnv):

    def __init__(self, n_batch, n_total, randomness, tweak=1.0):

        def mutate(np_random, tree):
            sysid = reacher_mutate(np_random, tree, randomness)
            if tweak > 1.0:
                return reacher_mutate(np_random, tree, tweak)
            return sysid

        ep_len = 256
        super().__init__(ShuffleboardEnv, xml_path, mutate, n_batch, n_total, ep_len)
