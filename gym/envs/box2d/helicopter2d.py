# -*- coding: utf-8 -*-
"""
OpenAI Gym environment for a 2D helicopter.

Environment is random hilly terrain.
Sensors are 16 fixed-angle range sensors ("Lidar"),
gyroscope, and accelerometer.
Controls are two nonnegative, bounded motor thrusts.
Reward is for +x velocity, with penalty for flying too high
above the terrain, crashing, or going out of bounds.

Author: James Preiss, University of Southern California
"""

import math
import random
import numpy as np
import Box2D

import gym
from gym import spaces
from gym.utils import seeding


# physical constants
GRAV = 9.8
LINEAR_DRAG = 0.5 # proportional to linear velocity (TODO: squared?)
ANGULAR_DRAG = 0.2 # proportional to angular velocity
INERTIA_MOMENT = 0.25 # TODO make sure units are right...
ARM_LENGTH = 0.25 # meters - distance from center of helicopter to motors

# environment limits
X_MAX = 500 # meters
X_MIN = -20 # meters
Z_MAX = 20  # meters
Z_MIN = 0   # meters

# dynamic limits
THRUST_MAX_PER_MOTOR = GRAV         # meters/sec^2.
THRUST_MAX = 2*THRUST_MAX_PER_MOTOR # meters/sec^2
# TODO: we define these limits so we can create a "Box" observation space 
# for Gym, but we do not actually enforce them in the simulation... 
# can we make the observation space unbounded instead?
W_MAX = 10               # radians/sec
ACC_MAX = 2 * THRUST_MAX # meters/sec^2

# sensor configuration & limits
N_LIDAR_RAYS = 16
LIDAR_THETA_MIN = -math.pi / 2 # radians, 0 is forward
LIDAR_THETA_MAX = math.pi / 4  # radians
LIDAR_RANGE = 10            # meters
LIDAR_NOHIT_SENTINEL = 1000 # lidar observation value when ray hits nothing

# TODO: heteroschedastic noise?
LIDAR_NOISE = 0.03  # meters stddev. from Velodyne Puck-Lite datasheet.
GYRO_NOISE = 0.0017 # radians/sec stddev. from Invensense MPU-9250 datasheet.
ACC_NOISE = 0.078   # meters/sec^2 stddev. from Invensense MPU-9250 datasheet.

# simulation parameters
FPS = 30
DT = 1.0 / FPS
TERRAIN_NPTS = 2048
TERRAIN_Z_RANGE = 10.0
TERRAIN_FREQ = 16.0 # higher: more small (high-frequency) bumps

# reward parameters
HEIGHT_PENALIZE_ABOVE = 2.5 # above this ground altitude, linear penalty


# helper functions
def np_duplicate(a):
	return np.concatenate([a, a])

def add_noise(a, stddev):
	return a + stddev * np.random.randn(*a.shape)

def clamp(x, xmin, xmax):
	return min(max(x, xmin), xmax)


class Helicopter2DEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': FPS
	}

	def __init__(self):

		self._seed()
		self.viewer = None
		self.terrain = Terrain()

		# observations: gyroscope, accelerometer, lidar, x2 for now and previous
		low_state  = np.array([-W_MAX, -ACC_MAX, -ACC_MAX] + [0 for i in range(N_LIDAR_RAYS)])
		high_state = -low_state
		high_state[3:] = LIDAR_NOHIT_SENTINEL
		self.observation_space = spaces.Box(
			np_duplicate(low_state), np_duplicate(high_state))

		# control inputs: back motor, front motor
		# inputs are shifted so the zero-action corresponds to hover
		low_action = np.array([-GRAV/2, -GRAV/2])
		high_action = np.array([THRUST_MAX_PER_MOTOR - GRAV/2, THRUST_MAX_PER_MOTOR - GRAV/2])
		self.action_space = spaces.Box(low_action, high_action)

		# used for rendering
		self.last_motors = np.array([0, 0])
		self.last_ddth = 0
		self.last_lidar = []
		self.last_obs = None

		# randomize the terrain
		self.reset()

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		print "seeded with:", seed
		return [seed]

	def _step(self, action):

		# clip - it seems that TRPO can generate actions outside the Box action space
		for i in [0, 1]:
			if action[i] <= -GRAV/2:
				action[i] = -GRAV/2
			if action[i] >= (THRUST_MAX_PER_MOTOR - GRAV/2):
				action[i] = (THRUST_MAX_PER_MOTOR - GRAV/2)

		# undo our shift that made 0 action correspond to hover
		m0 = action[0] + GRAV/2
		m1 = action[1] + GRAV/2
		self.dynamics.step_motors(m0, m1)

		# compute reward assuming no crash or finish
		height_above_terrain = (self.dynamics.pos[1] -
			self.terrain.height(self.dynamics.pos[0]))
		aloft_reward = 0 #0.1

		# this scale requires careful tuning - if too high, the policy learns
		# to gamble, going very fast but sometimes crashing
		speed_reward = 0.01 * self.dynamics.vel[0]

		# hinge loss
		non_terrain_hug_penalty = max(0, height_above_terrain - HEIGHT_PENALIZE_ABOVE)
		reward = aloft_reward + speed_reward - non_terrain_hug_penalty

		# check for crash or finish, reward accordingly
		done = False
		crash = height_above_terrain <= 0
		out_of_bounds = self.dynamics.pos[0] < X_MIN or self.dynamics.pos[1] > Z_MAX
		success = self.dynamics.pos[0] > X_MAX

		if crash or out_of_bounds:
			done = True
			reward = -100

		if success:
			done = True
			reward = 1000

		# store controls for rendering
		self.last_motors = action

		self.state = self._sensors()
		return self.state, reward, done, {}

	def _sensors(self):
		gyro, acc_imu = self.dynamics.imu()

		# simulate lidar
		[pts, hits, dists] = self.terrain.lidar(
			self.dynamics.pos, self.dynamics.th,
			N_LIDAR_RAYS, LIDAR_THETA_MIN, LIDAR_THETA_MAX)
		self.last_lidar = zip(pts, hits, dists)

		# add noise
		obs_now = np.concatenate([
			add_noise(gyro, GYRO_NOISE),
			add_noise(acc_imu, ACC_NOISE),
			add_noise(np.array(dists), LIDAR_NOISE)])

		# concatenate with previous measurement to make velocity observable
		cat_obs = obs_now if self.last_obs is None else self.last_obs
		obs = np.concatenate([cat_obs, obs_now])
		self.last_obs = obs_now
		return obs

	def _reset(self):
		# re-randomize the terrain, re-initialize state
		self.terrain.randomize(TERRAIN_NPTS, X_MIN, X_MAX, self.np_random)
		self.terrain_dirty_flag = True
		z = self.terrain.height(0) + 2.0
		self.dynamics = Heli2DDynamics([0, z])
		self.state = self._sensors()
		return self.state

	def _render(self, mode='human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return

		screen_width = 600
		screen_height = 400

		view_wide = 15
		view_tall = (screen_height / float(screen_width)) * view_wide

		if self.viewer is None:
			# set up viewer
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)

			# draw terrain
			self.terrain_polyline = rendering.make_polyline(
				zip(self.terrain.x, self.terrain.z))
			self.terrain_polyline.set_linewidth(2)
			self.viewer.add_geom(self.terrain_polyline)

			# draw quad
			l,r,t,b = -ARM_LENGTH, ARM_LENGTH, ARM_LENGTH/8, -ARM_LENGTH/8
			quad_polygon = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.quadtrans = rendering.Transform()

			# draw arrows to represent motor thrust
			arrow_xy = [(0, 0), (0, 1), (-0.2, 0.8), (0, 1), (0.2, 0.8)]
			m0_arrow = rendering.make_polyline(arrow_xy)
			m0_arrow.set_color(0, 0.8, 0.4)
			self.m0_trans = rendering.Transform()
			m0_arrow.add_attr(self.m0_trans)

			m1_arrow = rendering.make_polyline(arrow_xy)
			m1_arrow.set_color(0, 0.8, 0.4)
			self.m1_trans = rendering.Transform()
			m1_arrow.add_attr(self.m1_trans)

			quad_group = rendering.Compound([quad_polygon, m0_arrow, m1_arrow])
			quad_group.add_attr(self.quadtrans)
			self.viewer.add_geom(quad_group)

			# note: lidar rendering is done in "immediate mode" below

		# terrain has been re-randomized, update
		if self.terrain_dirty_flag:
			self.terrain_polyline.v = zip(self.terrain.x, self.terrain.z)
			self.terrain_dirty_flag = False

		# draw quad
		pos = self.dynamics.pos

		self.quadtrans.set_translation(pos[0], pos[1])
		self.quadtrans.set_rotation(self.dynamics.th)

		thrusts = self.last_motors + GRAV/2
		self.m0_trans.set_scale(thrusts[0] / 9.8, thrusts[0] / 9.8)
		self.m0_trans.set_translation(-ARM_LENGTH, 0)
		self.m1_trans.set_scale(thrusts[1] / 9.8, thrusts[1] / 9.8)
		self.m1_trans.set_translation(ARM_LENGTH, 0)

		self.viewer.set_bounds(-view_wide/2 + pos[0], view_wide/2 + pos[0],
			-view_tall/2 + pos[1], view_tall/2 + pos[1])

		# draw lidar rays
		for pt, hit, dist in self.last_lidar:
			if hit:
				self.viewer.draw_line(pos, hit, color=(1, 0, 0))
			else:
				self.viewer.draw_line(pos, pt,  color=(.5, .5, .5))

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

class Terrain:
	"""random terrain generation and ray-terrain intersection"""

	def randomize(self, N, x0, x1, rng):
		# randomize how hilly the terrain is. lower number = softer hills
		terrain_freq = 10 + 14 * rng.rand()
		# construct random terrain using frequency domain low-freq noise
		modulus = np.exp(-np.arange(N) / terrain_freq) * rng.randn(N)
		argument = rng.randn(N)
		freqspace = modulus * np.cos(argument) + 1j * modulus * np.sin(argument)
		z = np.fft.fft(freqspace).real
		zmin = np.min(z)
		zmax = np.max(z)
		self.z = (TERRAIN_Z_RANGE / (zmax - zmin)) * (z - zmin)
		self.x = np.linspace(x0, x1, N)

		# construct box2d world for lidar.
		self.world = Box2D.b2World()
		verts = zip(self.x, self.z)
		nverts = len(verts)
		for i in range(nverts - 1):
			p0, p1 = verts[i], verts[i + 1]
			edge = Box2D.b2EdgeShape(vertices=[p0, p1])
			body = Box2D.b2BodyDef(
				type=Box2D.b2_staticBody,
				fixtures=Box2D.b2FixtureDef(shape=edge, friction=0),
			)
			self.world.CreateBody(body)

	def height(self, x):
		"""terrain height at given x coordinate."""
		return np.interp(x, self.x, self.z)

	def lidar(self, x, body_th, N, th0, th1):
		"""simulate lidar."""
		class RaycastCallback(Box2D.b2RayCastCallback):
			def ReportFixture(self, fixture, point, normal, fraction):
				self.point = point
				return fraction

		cb = RaycastCallback()

		def bvec(x):
			return Box2D.b2Vec2(x[0], x[1])

		def cast(dir):
			cb.point = None
			self.world.RayCast(cb, bvec(x), bvec(dir))
			return cb.point

		def calc_dist(pt, hit):
			if hit:
				return np.linalg.norm(np.array(pt) - np.array(hit))
			else:
				return LIDAR_NOHIT_SENTINEL

		thetas = np.linspace(body_th + th0, body_th + th1, N)
		cast_pts = [(x[0] + 10 * math.cos(theta), x[1] + 10 * math.sin(theta)) for theta in thetas]
		hits = [cast(pt) for pt in cast_pts]
		dists = [calc_dist(pt, hit) for pt, hit in zip(cast_pts, hits)]

		return cast_pts, hits, dists

class Heli2DDynamics:

	def __init__(self, pos):
		self.pos = np.array([float(pos[0]), float(pos[1])])
		self.vel = np.array([0.0, 0.0])
		self.acc = np.array([0.0, 0.0])
		self.th  = np.array([0.0])
		self.dth = np.array([0.0])
		self.thrusts = np.array([1.0, 1.0])

	# m0: back motor, m1: front motor
	def step_motors(self, m0, m1):
		des_thrusts = np.array([m0, m1])
		self.thrusts = 0.6 * self.thrusts + 0.4 * des_thrusts
		ddth = (1.0 / INERTIA_MOMENT) * ARM_LENGTH * (self.thrusts[1] - self.thrusts[0])
		self.dth += DT * (ddth - ANGULAR_DRAG * self.dth)
		self.th += DT * self.dth

		# integrate linear part
		thrust_vec = sum(self.thrusts) * np.array([-math.sin(self.th), math.cos(self.th)])
		grav_force = np.array([0, -GRAV])
		self.acc = thrust_vec - LINEAR_DRAG * self.vel + grav_force
		self.vel += DT * self.acc
		self.pos += DT * self.vel

	# returns (gyro, acc)
	def imu(self):
		cth = np.cos(self.th)
		sth = np.sin(self.th)
		rot_world_to_imu = np.array([[cth, -sth], [sth, cth]]).T
		acc_imu = rot_world_to_imu.dot(self.acc + np.array([0, GRAV]))
		return self.dth, np.squeeze(acc_imu)

