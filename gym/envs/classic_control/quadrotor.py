"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering3d as r3d
import numpy as np
from numpy.linalg import norm
import traceback
import sys
from copy import deepcopy

logger = logging.getLogger(__name__)

GRAV = 9.81

def quadrotor_3dmodel(diameter):
    r = diameter / 2
    prop_r = 0.3 * diameter
    prop_h = prop_r / 20.0

    # "X" propeller configuration, start fwd left, go clockwise
    rr = r * np.sqrt(2)/2
    deltas = ((rr, rr, 0), (rr, -rr, 0), (-rr, -rr, 0), (-rr, rr, 0))
    colors = ((1,0,0), (1,0,0), (0,1,0), (0,1,0))
    def disc(translation, color):
        color = 0.3 * np.array(list(color)) + 0.2
        disc = r3d.Cylinder2(prop_r, prop_h, 32).translate(*translation)
        disc.set_color(*color)
        return disc
    props = [disc(d, c) for d, c in zip(deltas, colors)]

    arm_thicc = diameter / 20.0
    arm1 = r3d.Box(diameter, diameter/10, arm_thicc).translate(0, 0, -arm_thicc)
    arm1.set_rotation(r3d.rotz(np.pi/4)).set_color(0.5, 0.5, 0.5)
    arm2 = r3d.Box(diameter, diameter/10, arm_thicc).translate(0, 0, -arm_thicc)
    arm2.set_rotation(r3d.rotz(3*np.pi/4)).set_color(0.5, 0.5, 0.5)
    arrow = r3d.Arrow(0.12*prop_r, 2.5*prop_r, 16)
    arrow.set_color(0.3, 0.3, 1.0)

    bodies = props + [arm1, arm2, arrow]
    return r3d.Compound(bodies)

def is_orthonormal(m):
    return np.max(np.abs(np.matmul(m, m.T) - np.eye(3)).flatten()) < 0.00001

def normalize(x):
    n = norm(x)
    if n < 0.00001:
        return x, 0
    return x / n, n

# gram-schmidt orthogonalization
def orth_cols(m):
    assert len(m.shape) == 2
    _, nc = m.shape
    m[:,0] = m[:,0] / norm(m[:,0])
    for i in range(1, nc):
        c = m[:,i]
        left = m[:,:i]
        proj = np.matmul(left.T, c)
        c = c - np.matmul(left, proj)
        clen = norm(c)
        assert clen > 0.95
        m[:,i] = c / clen
    return m

def rand_uniform_rot3d(np_random):
    randunit = lambda: normalize(np_random.normal(size=(3,)))[0]
    up = randunit()
    fwd = randunit()
    while np.dot(fwd, up) > 0.95:
        fwd = randunit()
    left = normalize(np.cross(up, fwd))
    up = np.cross(fwd, left)
    rot = np.hstack([fwd, left, up])

def npa(*args):
    return np.array(args)

def omega_mat_deriv(omega):
    x, y, z = omega
    dRdt = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return dRdt

class QuadrotorDynamics(object):
    # thrust_to_weight is the total, it will be divided among the 4 props
    # torque_to_thrust is ratio of torque produced by prop to thrust
    def __init__(self, mass, arm_length, inertia, thrust_to_weight=2.0, torque_to_thrust=0.05):
        assert np.isscalar(mass)
        assert np.isscalar(arm_length)
        assert inertia.shape == (3,)
        self.mass = mass
        self.arm = arm_length
        self.inertia = inertia
        self.thrust = GRAV * mass * thrust_to_weight / 4.0
        self.torque = torque_to_thrust * self.thrust
        scl = arm_length / norm([1,1,0])
        self.prop_pos = scl * np.array([
            [1,  1, -1, -1],
            [1, -1, -1,  1],
            [0,  0,  0,  0]]).T # row-wise easier with np
        self.prop_ccw = np.array([1, -1, 1, -1])

    # pos, vel, in world coords
    # rotation is (body coords) -> (world coords)
    # omega in body coords
    def set_state(self, position, velocity, rotation, omega, thrusts=np.zeros((4,))):
        for v in (position, velocity, omega):
            assert v.shape == (3,)
        assert thrusts.shape == (4,)
        assert rotation.shape == (3,3)
        assert is_orthonormal(rotation)
        self.pos = deepcopy(position)
        self.vel = deepcopy(velocity)
        self.rot = deepcopy(rotation)
        self.omega = deepcopy(omega)
        self.thrusts = deepcopy(thrusts)
        self.crashed = False

    # generate a random state (meters, meters/sec, radians/sec)
    def random_state(self, np_random, box, vel_max=15.0, omega_max=2*np.pi):
        pos = np_random.uniform(low=-box, high=box, size=(3,))
        vel = np_random.uniform(low=-vel_max, high=vel_max, size=(3,))
        omega = np_random.uniform(low=-omega_max, high=omega_max, size=(3,))
        rot = rand_uniform_rot3d(np_random)
        self.set_state(pos, vel, rot, omega)

    def step(self, thrust_cmds, dt):

        if self.pos[2] <= self.arm:
            # crashed, episode over
            self.pos[2] = self.arm
            self.vel *= 0
            self.omega *= 0
            self.crashed = True
            return

        assert np.all(thrust_cmds >= 0)
        assert np.all(thrust_cmds <= 1)
        thrusts = self.thrust * thrust_cmds
        thrust = npa(0,0,np.sum(thrusts))
        torques = np.cross(self.prop_pos, npa(0,0,1)) * thrusts[:,None]
        torques[:,2] += self.torque * self.prop_ccw * thrust_cmds
        torque = np.sum(torques, axis=0)

        # TODO add noise

        vel_damp = 0.99
        omega_damp = 0.99

        # rotational dynamics
        omega_dot = ((1.0 / self.inertia) *
            (np.cross(-self.omega, self.inertia * self.omega) + torque))
        self.omega = omega_damp * self.omega + dt * omega_dot

        dRdt = np.matmul(omega_mat_deriv(self.omega), self.rot)
        rot = self.rot + dt * dRdt;
        self.rot = orth_cols(rot)

        # translational dynamics
        g = npa(0, 0, -GRAV)
        acc = g + (1.0 / self.mass) * np.matmul(self.rot, thrust)
        self.vel = vel_damp * self.vel + dt * acc
        self.pos = self.pos + dt * self.vel

    def state_vector(self):
        return np.concatenate([
            self.pos, self.vel, self.rot.flatten(), self.omega])

class ChaseCamera(object):
    def __init__(self, pos=npa(0,0,0), vel=npa(0,0,0)):
        self.pos_smooth = pos
        self.vel_smooth = vel
        self.view_dist = 3

    def step(self, pos, vel):
        # lowpass filter
        ap = 0.6
        av = 0.999
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos
        self.vel_smooth = av * self.vel_smooth + (1 - av) * vel

    # return eye, center, up suitable for gluLookAt
    def look_at(self):
        veln, n = normalize(self.vel_smooth)
        up = npa(0, 0, 1)
        if np.abs(veln[2]) > 0.95 or n < 0.1 or True:
            # look over quadrotor's right shoulder, like we're in passenger seat
            right, _ = normalize(npa(-0.9, -0.3, 0))
            #fwd = npa(0, 1, 0)
        else:
            right, _ = normalize(np.cross(veln, up))
            #fwd = np.cross(up, right)
        eye = self.pos_smooth + self.view_dist * (right + 0.3 * up)
        center = self.pos_smooth
        return eye, center, up


class QuadrotorEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        np.seterr(all='raise')
        mass = 0.5
        arm_length = 0.33 / 2.0
        inertia = mass * npa(0.01, 0.01, 0.02)
        self.thrust_to_weight = 2.0
        self.dynamics = QuadrotorDynamics(mass, arm_length, inertia,
            thrust_to_weight=self.thrust_to_weight)

        self.ep_len = 256
        self.tick = 0
        self.dt = 1.0 / 50.0

        # make it so the zero action corresponds to hovering
        low = -1.0 * np.ones(4)
        high = (self.thrust_to_weight - 1.0) * np.ones(4)
        self.action_space = spaces.Box(low, high)
        print("action space:", low, high)
        # pos, vel, rot, omega
        obs_dim = 3 + 3 + 9 + 3
        # TODO tighter bounds on some variables
        obs_high = 100 * np.ones(obs_dim)
        # rotation mtx guaranteed to be orthogonal
        obs_high[6:-3] = 1
        self.observation_space = spaces.Box(-obs_high, obs_high)

        self._seed()
        self.reset()
        self.viewer = None
        self.model = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # noop for gfx testing
        #return self.dynamics.state_vector(), 0, False, {}
        action = (action + 1.0) / self.thrust_to_weight
        action[action < 0] = 0
        action[action > 1] = 1
        #action[:] = 0.5
        #action = npa(0.6, 0.4, 0.6, 0.4)
        self.dynamics.step(action, self.dt)
        self.camera.step(self.dynamics.pos, self.dynamics.vel)

        self.tick += 1
        done = self.tick >= self.ep_len

        loss_pos = norm(self.goal - self.dynamics.pos)
        loss_spin = 0.5 * np.sum(np.abs(self.dynamics.omega))
        loss_crash = 50 * self.dynamics.crashed
        if self.tick % 30 == 0 and False:
            print("losses:")
            print("  pos =", loss_pos)
            print(" spin =", loss_spin)
            print("crash =", loss_crash)
        reward = -self.dt * (loss_pos + loss_spin + loss_crash)

        sv = self.dynamics.state_vector()
        return sv, reward, done, {}

    def _reset(self):
        pos = npa(-20, 0, 2)
        vel = omega = npa(0, 0, 0)
        rotation = np.eye(3)
        self.dynamics.set_state(pos, vel, rotation, omega)
        self.camera = ChaseCamera(pos=pos, vel=vel)
        self.goal = npa(0, 0, 2)
        self.tick = 0
        return self.dynamics.state_vector()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            self.viewer = r3d.Viewer(screen_width, screen_height)
            diameter = 2 * self.dynamics.arm
            self.model = quadrotor_3dmodel(diameter).set_rotation(np.eye(3)).translate(*self.dynamics.pos)
            self.viewer.add_geom(self.model)

            floor = r3d.Rect((1000, 1000), (0, 100), (0, 100))
            floor.attrs = []
            floor.add_attr(r3d.CheckerTexture())
            self.viewer.add_geom(floor)

            goal = r3d.Sphere(diameter/3.0, 18).translate(*self.goal)
            goal.set_color(0.5,0.4,0)
            self.viewer.add_geom(goal)

        self.model.set_translate(*self.dynamics.pos)
        self.model.set_rotation(self.dynamics.rot)

        eye, center, up = self.camera.look_at()
        self.viewer.look_at(eye, center, up)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
