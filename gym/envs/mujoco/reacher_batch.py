import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import ReacherEnv
from gym.envs.mujoco.mujoco_batch import MujocoBatchEnv

from copy import deepcopy
import xml.etree.cElementTree as ElementTree
import tempfile
import os

def reacher_gen(np_random):

    RATIO = 2.0
    MIN_LEN = 0.1 / RATIO
    MAX_LEN = 0.1 * RATIO
    MIN_FORCE = 1 / RATIO
    MAX_FORCE = 1 * RATIO

    path = os.path.join(os.path.dirname(__file__), "assets", "reacher.xml")
    tree = ElementTree.parse(path)
    up = tree.find("worldbody/body")
    fore = up.find("body")
    fingertip = fore.find("body").attrib
    up = up.find("geom").attrib
    fore_attrib = fore.attrib
    fore_geom = fore.find("geom").attrib
    actuators = [motor.attrib for motor in
            tree.find('actuator').findall('motor')]

    while True:
        up_len, fore_len = np_random.uniform(MIN_LEN, MAX_LEN, size=(2,))
        fmt_str = "0 0 0 {} 0 0"
        up["fromto"] = fmt_str.format(up_len)
        fore_attrib["pos"] = "{} 0 0".format(up_len)
        fore_geom["fromto"] = fmt_str.format(fore_len)
        fingertip["pos"] = "{} 0 0".format(fore_len + 0.01)

        max_rad = up_len + fore_len
        min_rad = np.abs(up_len - fore_len)

        up_force, fore_force = np_random.uniform(MIN_FORCE, MAX_FORCE, size=(2,))
        actuators[0]["ctrlrange"] = "{} {}".format(-up_force, up_force)
        actuators[1]["ctrlrange"] = "{} {}".format(-fore_force, fore_force)

        tf = tempfile.NamedTemporaryFile(delete=True)
        write_path = tf.name
        tree.write(write_path)

        env = ReacherEnv(model_path=write_path, min_radius=min_rad, max_radius=max_rad)
        sysid_vec = np.array([up_len, fore_len, up_force, fore_force])
        yield env, sysid_vec


class ReacherBatchEnv(MujocoBatchEnv):

    def __init__(self, N=32):
        print("HELLO from ReacherBatch2")
        total_envs = min(1000, 100*N)
        ep_len = 64
        super().__init__(N, total_envs, reacher_gen, ep_len)
