from copy import deepcopy
import itertools as it
import operator as op
import tempfile
import xml.etree.cElementTree as ElementTree

import gym
from gym import utils
import numpy as np


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
    # seeds: list of seeds for sub-environments.
    # ep_len: episode length. EnvClass should not end episodes on its own.
    #
    def _init_after_seed(self, EnvClass, xml_path, mutator, n_parallel, seeds, ep_len):

        #print("mujoco_batch ctor with seeds:")
        #print(seeds)

        self.N = n_parallel
        self.seeds = seeds
        self.ep_len = ep_len
        self.np_random = np.random.RandomState(self.seeds[0])

        # construct a bunch of randomized models
        # little complicated to deal with N_RAND < N case,
        # where we need to construct multiple copies of the
        # we can't just sample with replacement because mujoco envs have reference semantics
        def env_repeater():
            while True:
                yield from _generate_envs(seeds, EnvClass, xml_path, mutator)
        n_envs = max(self.N, len(seeds))
        envs_all, sysids_all = zip(*list(it.islice(env_repeater(), n_envs)))
        unique_ids = set(id(env) for env in envs_all)
        assert len(unique_ids) == n_envs
        self.envs_all = np.array(envs_all)

        assert len(sysids_all[0].shape) == 1
        self.sysid_dim = sysids_all[0].shape[0]
        self.sysids_all = np.row_stack(sysids_all)

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
        self.tick = 0

        # rendering stuff
        self.metadata = deepcopy(env0.metadata)

    def seed(self, *args):
        self._init_after_seed(*self.init_args)
        return self.seeds

    def sample_sysid(self):
        prev_env0 = None
        if self.envs is not None:
            prev_env0 = self.envs[0]
        selection = self.np_random.choice(len(self.envs_all), self.N, replace=False)
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


def _generate_envs(seeds, EnvClass, xml_path, mutator):

    tree = ElementTree.parse(xml_path)
    for seed in seeds:
        # print("generating mujoco_batch sub-env with seed", seed)
        treecopy = deepcopy(tree)
        npr_gen = np.random.RandomState(seed)
        sysid_vec = mutator(npr_gen, treecopy)

        tf = tempfile.NamedTemporaryFile(delete=True)
        write_path = tf.name
        treecopy.write(write_path)

        env = EnvClass(model_path=write_path)
        env.seed(seed)
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
    RATIO_MIN = 0.25

    def rand_length_withchild(geom, child):
        assert geom.attrib["type"] == "capsule", "TODO: support other geom types"

        assert not ("axisangle" in geom.attrib and "fromto" in geom.attrib)

        if "axisangle" in geom.attrib:
            cpos = parsevec(child.attrib["pos"])
            body_len = np.linalg.norm(cpos)

            geom_size = parsevec(geom.attrib["size"])
            #geom_len_delta = geom_size[1] - body_len
            geom_len_delta = 0

            ratio = rand_ratio(npr, JOINT_LEN_RATIO)
            #ratio = npr.uniform(RATIO_MIN, JOINT_LEN_RATIO)
            geom_size[1] = ratio * 0.5 * body_len + geom_len_delta 

            geom.attrib["pos"] = formatvec(0.5 * cpos * ratio)
            geom.attrib["size"] = formatvec(geom_size)
            child.attrib["pos"] = formatvec(ratio * cpos)

            return [ratio * body_len]

        elif "fromto" in geom.attrib:
            cpos = parsevec(child.attrib["pos"])
            body_len = np.linalg.norm(cpos)

            fromto = parsevec(geom.attrib["fromto"])
            if not np.all(fromto[:3] == 0):
                print("invalid fromto:", fromto)
            assert np.all(fromto[:3] == 0)
            assert np.all(fromto[3:] == cpos)

            ratio = rand_ratio(npr, JOINT_LEN_RATIO)
            #ratio = npr.uniform(RATIO_MIN, JOINT_LEN_RATIO)

            geom.attrib["fromto"] = formatvec(ratio * fromto)
            child.attrib["pos"] = formatvec(ratio * cpos)

            return [ratio * body_len]
        else:
            raise NotImplementedError


    def rand_length_end(geom):

        g_type = geom.attrib["type"]

        # special case for reacher fingertip. we could change radius 
        if g_type == "sphere":
            assert geom.attrib["name"] == "fingertip"
            return []

        assert g_type == "capsule", "TODO: support other geom types"

        assert not ("axisangle" in geom.attrib and "fromto" in geom.attrib)

        if "axisangle" in geom.attrib:

            scale = rand_ratio(npr, JOINT_LEN_RATIO)
            #scale = npr.uniform(RATIO_MIN, JOINT_LEN_RATIO)
            fn_attr(geom, "pos", op.mul, scale)
            geom_size = parsevec(geom.attrib["size"])
            fn_attr(geom, "size", op.mul, [1, scale])
            return [geom_size[1] * scale]

        elif "fromto" in geom.attrib:

            fromto = parsevec(geom.attrib["fromto"])
            if not np.all(fromto[:3] == 0):
                print("invalid fromto:", fromto)
            assert np.all(fromto[:3] == 0)

            ratio = rand_ratio(npr, JOINT_LEN_RATIO)
            #ratio = npr.uniform(RATIO_MIN, JOINT_LEN_RATIO)
            geom.attrib["fromto"] = formatvec(ratio * fromto)
            body_len = np.linalg.norm(fromto[3:])

            return [ratio * body_len]

    sysid_params = []

    # randomize the joint connecting me to my parent
    joints = body.findall("joint")
    assert len(joints) < 2, "must be kinematic chain"

    if len(joints) == 1:
        joint = joints[0]

        # not all envs define these for all joints.
        # TODO: find the <default> values and mutate those

        try:
            damping_mul = rand_ratio(npr, JOINT_DAMPING_RATIO)
            damping = fn_attr(joint, "damping", op.mul, damping_mul)
            sysid_params.extend([0.1 * damping[0]])
        except KeyError:
            pass

        try:
            stiffness_mul = rand_ratio(npr, JOINT_STIFFNESS_RATIO)
            stiffness = fn_attr(joint, "stiffness", op.mul, stiffness_mul)
            sysid_params.extend([0.01 * stiffness[0]])
        except KeyError:
            pass

        try:
            range = parsevec(joint.attrib["range"])
            is_degrees = np.any(np.abs(range) > 5.0)
            delta = np.radians(JOINT_RANGE_ADD) if is_degrees else JOINT_RANGE_ADD
            range_add = list(npr.uniform(-delta, delta, size=(2)))
            range = fn_attr(joint, "range", op.add, range_add)
            if is_degrees:
                range = np.radians(range)
            sysid_params.extend(range)
        except KeyError:
            pass
    else:
        # rigid connection, e.g. reacher fingertip, do nothing
        pass

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


def randomize_gear_ratios(np_random, tree, ratio, disable=False):
    ac = tree.find("actuator")
    sysid_params = []
    motors = ac.findall("motor")
    # with 50% chance, disable exactly one motor
    ind_disabled = np_random.randint(len(motors))
    if not disable or np_random.uniform() > 0.5:
        ind_disabled = -1
    for i, motor in enumerate(motors):
        if i == ind_disabled:
            gear_ratio = fn_attr(motor, "gear", op.mul, 0.0)
        else:
            gear_ratio = fn_attr(motor, "gear", op.mul, rand_ratio(np_random, ratio))
        sysid_params.append(0.01 * gear_ratio[0]) # keep all sysid params near N(0,1)
    return sysid_params
