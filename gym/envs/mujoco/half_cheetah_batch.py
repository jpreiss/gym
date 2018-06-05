import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import HalfCheetahEnv
from gym.envs.mujoco.mujoco_batch import MujocoBatchEnv

from copy import deepcopy
import xml.etree.cElementTree as ElementTree
import tempfile
import os
import sys
import operator as op

def parsevec(attrib):
    assert isinstance(attrib, str)
    parsed = np.array([float(s) for s in attrib.split(" ")])
    return parsed

def formatvec(v, precision=4):
    #print("formatvec:", list(v))
    return " ".join("{:.{}f}".format(x, precision) for x in v)

def rand_ratio(np_random, ratio):
    return ratio ** np_random.uniform(-1, 1)

def named_child(node, tag, name):
    children = node.findall(tag)
    filtered = [c for c in children if c.attrib["name"] == name]
    assert len(filtered) == 1
    return filtered[0]

# apply the given function to the attribute
# (handles lists with scalar fn and scalar or list params)
def fn_attr(node, attrib, fn, params):
    vals = parsevec(node.attrib[attrib])
    if isinstance(params, list):
        assert len(vals) == len(params)
        newvals = list(fn(x, p) for x, p in zip(vals, params))
    else:
        newvals = list(fn(x, params) for x in vals)
    node.attrib[attrib] = formatvec(newvals)
    return newvals


RANDOMNESS = 2.0

TORSO_LEN_RATIO = RANDOMNESS
JOINT_LEN_RATIO = RANDOMNESS
JOINT_RANGE_ADD = 0.3 * (RANDOMNESS - 1.0)
JOINT_STIFFNESS_RATIO = RANDOMNESS
JOINT_DAMPING_RATIO = RANDOMNESS
GEAR_RATIO = RANDOMNESS


# input must be a kinematic chain (sequence of bodies separated by hinge joints)
# randomizes in-place to create a new XML tree describing a randomized version of the original.
# returns a list of the SysID parameters
def randomize_chain(np_random, body):

    npr = np_random

    def rand_length_withchild(geom, child):

        assert geom.attrib["type"] == "capsule", "TODO: support other geom types"
        # TODO I think this just rotates the capsule itself - in that case, not needed
        #fn_attr(geom, "axisangle", rand_add, [0, 0, 0, angleplusminus])
        cpos = parsevec(child.attrib["pos"])
        body_len = np.linalg.norm(cpos)

        geom_size = parsevec(geom.attrib["size"])
        geom_len_delta = geom_size[1] - body_len
        #assert geom_len_delta >= 0
        geom_len_delta = 0

        #print("child pos old", child.attrib["pos"])

        ratio = rand_ratio(npr, JOINT_LEN_RATIO)
        geom_size[1] = ratio * 0.5 * body_len + geom_len_delta 

        geom.attrib["pos"] = formatvec(0.5 * cpos * ratio)
        geom.attrib["size"] = formatvec(geom_size)
        child.attrib["pos"] = formatvec(ratio * cpos)

        #print("child pos new", child.attrib["pos"])

        return [ratio * body_len]

    def rand_length_end(geom):
        assert geom.attrib["type"] == "capsule", "TODO: support other geom types"
        # TODO I think this just rotates the capsule itself - in that case, not needed
        #fn_attr(geom, "axisangle", rand_add, [0, 0, 0, angleplusminus])
        scale = rand_ratio(npr, JOINT_LEN_RATIO)
        fn_attr(geom, "pos", op.mul, scale)
        geom_size = parsevec(geom.attrib["size"])
        fn_attr(geom, "size", op.mul, [1, scale])
        return [geom_size[1] * scale]

    sysid_params = []

    # randomize the joint connecting me to my parent
    joints = body.findall("joint")
    assert len(joints) == 1, "must be kinematic chain"
    damping_mul = rand_ratio(npr, JOINT_DAMPING_RATIO)
    stiffness_mul = rand_ratio(npr, JOINT_STIFFNESS_RATIO)
    range_add = list(npr.uniform(-JOINT_RANGE_ADD, JOINT_RANGE_ADD, size=(2)))
    damping = fn_attr(joints[0], "damping", op.mul, damping_mul)
    stiffness = fn_attr(joints[0], "stiffness", op.mul, stiffness_mul)
    range = fn_attr(joints[0], "range", op.add, range_add)
    sysid_params.extend(damping + stiffness + range)

    # randomize my child
    geom = body.findall("geom")
    assert len(geom) == 1
    geom = geom[0]
    children = body.findall("body")
    assert len(children) < 2, "must be kinematic chain"
    if children:
        sysid_params.extend(rand_length_withchild(geom, children[0]))
        # recurse
        sysid_params.extend(randomize_chain(npr, children[0]))
    else:
        sysid_params.extend(rand_length_end(geom))

    return sysid_params


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
        sysid_params.extend(randomize_chain(np_random, thigh))

    ac = tree.find("actuator")
    for motor in ac.findall("motor"):
        gear_ratio = fn_attr(motor, "gear", op.mul, rand_ratio(np_random, GEAR_RATIO))
        sysid_params.extend(gear_ratio)

    return np.array(sysid_params)


def half_cheetah_gen(np_random):

    npr = np_random

    path = os.path.join(os.path.dirname(__file__), "assets", "half_cheetah.xml")
    tree = ElementTree.parse(path)

    i = 1
    while True:
        treecopy = deepcopy(tree)
        sysid_vec = mutate_cheetah_tree(npr, treecopy)

        tf = tempfile.NamedTemporaryFile(delete=True)
        write_path = tf.name
        treecopy.write(write_path)
        #for line in open(write_path):
            #if "bshin" in line:
                #sys.stdout.write(line)

        #print("constructing env", i)
        env = HalfCheetahEnv(model_path=write_path)
        s = np_random.randint(100000)
        env.seed(s)
        yield env, sysid_vec
        #return env, sysid_vec
        #print("yielded env", i)
        i += 1


class HalfCheetahBatchEnv(MujocoBatchEnv):

    def __init__(self, N=64):
        total_envs = 256
        ep_len = 196
        super().__init__(N, total_envs, half_cheetah_gen, ep_len)
