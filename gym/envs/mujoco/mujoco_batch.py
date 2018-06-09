import gym
from gym import utils

import numpy as np

from copy import deepcopy
import itertools as it
import operator as op
import tempfile
import xml.etree.cElementTree as ElementTree


class MujocoBatchEnv(gym.Env):
    """
    Wraps a collection of Gym Mujoco environments and presents
    a single batch environment with the ability to randomize SysID parameters.
    The user provides a function that randomizes the Mujoco XML model
    in-place as an ElementTree object.
    The functions below this class provide tools to make this easier.
    """

    def __init__(self, *args):
        self.init_args = args

    #
    # EnvClass: gym Mujoco environment class name.
    # mutator: function that modifies an ElementTree xml tree in place
    #          and returns a 1D NumPy ndarray of the SysID params.
    # n_parallel: number of environments to execute at same time (using batch-Gym)
    # n_total: total number of random environments to construct.
    #          at each iteration, a random choice of n_parallel is drawn.
    # ep_len: episode length. EnvClass should not end episodes on its own.
    #
    def _init_after_seed(self, EnvClass, xml_path, mutator, n_parallel, n_total, ep_len):
        self.N = n_parallel
        self.N_RAND = n_total
        self.mean_rews = np.zeros(n_total)

        self.tick = 0
        self.ep_len = ep_len

        # construct a bunch of randomized models
        envs_all, sysids_all = zip(*list(it.islice(
            _generate_envs(self.np_random, EnvClass, xml_path, mutator),
            self.N_RAND)))
        assert len(sysids_all[0].shape) == 1
        self.sysid_dim = sysids_all[0].shape[0]
        self.envs_all = np.array(envs_all)
        self.sysids_all = np.row_stack(sysids_all)
        s = self.sysids_all.flatten()
        print("sysid params: max {}, min {}, mean {}, std {}".format(
            np.amax(s), np.amin(s), np.mean(s), np.std(s)))


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
        selection = self.np_random.choice(self.N_RAND, self.N, replace=False)
        self.selection = selection
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

        if False:
            beta = 0.9995
            self.mean_rews[self.selection] = (
                beta * self.mean_rews[self.selection] +
                (1.0 - beta) * rewards)
            rewards -= self.mean_rews[self.selection]

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


def _generate_envs(npr, EnvClass, xml_path, mutator):

    tree = ElementTree.parse(xml_path)
    while True:
        treecopy = deepcopy(tree)
        sysid_vec = mutator(npr, treecopy)

        tf = tempfile.NamedTemporaryFile(delete=True)
        write_path = tf.name
        treecopy.write(write_path)

        env = EnvClass(model_path=write_path)
        s = npr.randint(100000)
        env.seed(s)
        yield env, sysid_vec


# parses the space-separated values used in Mujoco XML attributes
# into an np.array of floats.
def parsevec(attrib):
    assert isinstance(attrib, str)
    parsed = np.array([float(s) for s in attrib.split(" ")])
    return parsed

# formats iterable of floats into space-separated string for attributes.
def formatvec(v, precision=4):
    return " ".join("{:.{}f}".format(x, precision) for x in v)

# return a log-uniformly distributed value between 1/ratio and ratio.
def rand_ratio(np_random, ratio):
    return ratio ** np_random.uniform(-1, 1)

# find the child of this node with type tag and attrib "name" == name.
def named_child(node, tag, name):
    children = node.findall(tag)
    filtered = [c for c in children if c.attrib["name"] == name]
    assert len(filtered) == 1
    return filtered[0]

# apply the given function to the attribute
# (handles lists with scalar fn and scalar or list params)
# returns the new values as a list
# e.g.:
#   fn_attr(<node attr="1.0 2.0">, op.mul, 2.0) -> [2.0, 4.0]
#
def fn_attr(node, attrib, fn, params):
    vals = parsevec(node.attrib[attrib])
    if isinstance(params, list):
        assert len(vals) == len(params)
        newvals = list(fn(x, p) for x, p in zip(vals, params))
    else:
        newvals = list(fn(x, params) for x in vals)
    node.attrib[attrib] = formatvec(newvals)
    return newvals


# input must be a kinematic chain (sequence of bodies separated by hinge joints)
# randomizes in-place to create a new XML tree describing a randomized version of the original.
# returns a list of the SysID parameters
def randomize_chain(np_random, body, randomness):

    npr = np_random

    JOINT_LEN_RATIO = randomness
    JOINT_RANGE_ADD = 0.3 * (randomness - 1.0)
    JOINT_STIFFNESS_RATIO = randomness
    JOINT_DAMPING_RATIO = randomness

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
    sysid_params.extend([0.1 * damping[0]] + [0.01 * stiffness[0]] + range)

    # randomize my child
    geom = body.findall("geom")
    assert len(geom) == 1
    geom = geom[0]
    children = body.findall("body")
    assert len(children) < 2, "must be kinematic chain"
    if children:
        sysid_params.extend(rand_length_withchild(geom, children[0]))
        # recurse
        sysid_params.extend(randomize_chain(npr, children[0], randomness))
    else:
        sysid_params.extend(rand_length_end(geom))

    return sysid_params


def randomize_gear_ratios(np_random, tree, ratio):
    ac = tree.find("actuator")
    sysid_params = []
    for motor in ac.findall("motor"):
        gear_ratio = fn_attr(motor, "gear", op.mul, rand_ratio(np_random, ratio))
        sysid_params.append(0.01 * gear_ratio[0]) # keep all sysid params near N(0,1)
    return sysid_params
