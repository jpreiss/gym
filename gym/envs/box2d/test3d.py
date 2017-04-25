import rendering3d
import numpy as np
import scipy.misc
import noise
import time

v = rendering3d.Viewer(800, 600)

def perlin2d(N):
	z = np.zeros((N,N))
	nz = noise.snoise2
	octaves = 8
	freq = 256.0
	for x in range(N):
		for y in range(N):
			z[y][x] = 3* nz(x / freq, y / freq, octaves, persistence=0.1)
	return z

z = perlin2d(256)
xrange = (-4, 4)
yrange = (-4, 4)

v.add_terrain(z, xrange, yrange)

N = 60
theta = np.linspace(0, 2*np.pi, N)
for j in range(3):
	for th in theta:
		pos = (5*np.cos(th), 5*np.sin(th), 5)
		v.lookat(pos, (0, 0, 0), (0, 0, 1))
		v.render()
		time.sleep(1.0 / 30.0)
