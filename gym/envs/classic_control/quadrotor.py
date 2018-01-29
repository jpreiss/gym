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
import traceback
import sys

logger = logging.getLogger(__name__)


def quadrotor_3dmodel(diameter):
    r = diameter / 2
    prop_r = 0.3 * diameter
    prop_h = prop_r / 20.0
    deltas = ((r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0))
    def disc(d):
        color = np.array(d)
        if np.any(color < 0):
            color = 0.6 * (0.2 - 0.7 * color)
        else:
            color = 0.2 + 0.7 * color
        disc = r3d.Cylinder(prop_r, prop_h, 32).translate(*d)
        disc.set_color(*color)
        return disc
    arm1 = r3d.Box(diameter, diameter/10, diameter/20).translate(0, 0, -diameter/10)
    arm1.set_color(0.6, 0.6, 0.6)
    arm2 = r3d.Box(diameter/10, diameter, diameter/20).translate(0, 0, -diameter/10)
    arm2.set_color(0.3, 0.3, 0.3)
    arrow = r3d.Arrow(0.12*prop_r, 2.5*prop_r, 16).translate(0, 0, -diameter/10)
    arrow.set_color(0.3, 0.3, 1.0)
    bodies = [disc(d) for d in deltas] + [arm1, arm2, arrow]
    return r3d.Compound(bodies)

class QuadrotorEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.x_threshold = 2.4

        self.ep_len = 256
        self.tick = 0
        self.view_theta = 0

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            100,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None
        self.model = None
        self.state = None
        self.pos = np.zeros(3)


        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = 0.995 * x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = 0.995 * theta_dot + self.tau * thetaacc
        self.state = (x,x_dot,theta,theta_dot)

        theta_wrap = (theta + np.pi) % (2 * np.pi) - np.pi
        vel_rew = - 0.05 * np.abs(theta_dot) - 0.02 * np.abs(x_dot)
        pos_rew = -(1.0/np.pi) * np.abs(theta_wrap) - 0.05*np.abs(x)
        reward = -100 * (np.abs(x) > self.x_threshold) + pos_rew + vel_rew

        self.pos[0] += 0.03
        if self.model:
            self.model.set_translate(*self.pos)

        self.tick += 1
        done = False
        if self.tick >= self.ep_len:
            done = True

        return np.array(self.state), reward, done, {}


    def _reset(self):
        self.tick = 0
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state[2] += np.pi #theta
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            self.viewer = r3d.Viewer(screen_width, screen_height)
            #b = rendering3d.Box3D(1, 1, 1)
            #b = rendering3d.Cone(0.5, 0.9, 32)
            #b = rendering3d.Arrow(0.1, 1, 32)
            self.model = quadrotor_3dmodel(1.33)
            #b.set_color(1, 0, 0)
            self.viewer.add_geom(self.model)

            #glShadeModel (GL_SMOOTH);

            #glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
            #glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);


            #l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            #axleoffset =cartheight/4.0
            #cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            #self.carttrans = rendering.Transform()
            #cart.add_attr(self.carttrans)
            #self.viewer.add_geom(cart)
            #l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            #pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            #pole.set_color(.8,.6,.4)
            #self.poletrans = rendering.Transform(translation=(0, axleoffset))
            #pole.add_attr(self.poletrans)
            #pole.add_attr(self.carttrans)
            #self.viewer.add_geom(pole)
            #self.axle = rendering.make_circle(polewidth/2)
            #self.axle.add_attr(self.poletrans)
            #self.axle.add_attr(self.carttrans)
            #self.axle.set_color(.5,.5,.8)
            #self.viewer.add_geom(self.axle)
            #self.track = rendering.Line((0,carty), (screen_width,carty))
            #self.track.set_color(0,0,0)
            #self.viewer.add_geom(self.track)

        if self.state is None: return None

        #x = self.state
        #cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        #self.carttrans.set_translation(cartx, carty)
        #self.poletrans.set_rotation(-x[2])

        cx = -2.5 #2 * np.cos(self.view_theta)
        cy = -1.0 #2 * np.sin(self.view_theta)
        self.view_theta += 0.02
        self.viewer.look_at((cx,cy,0.7), (0,0,0), (0,0,1))
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
