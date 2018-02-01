"""
3D rendering framework
"""
from __future__ import division
from copy import deepcopy
import os
import six
import sys

if "Apple" in sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

from gym.utils import reraise
from gym import error
import matplotlib.pyplot as plt

try:
    import pyglet
except ImportError as e:
    reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
    from pyglet.gl import *
except ImportError as e:
    reraise(prefix="Error occured while running `from pyglet.gl import *`",suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")

import math
import numpy as np

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

class Viewer(object):

    #
    # public interface
    #

    def __init__(self, width, height, display=None):

        self.fov = 45

        display = get_display(display)
        self.window = pyglet.window.Window(display=display,
            width=width, height=height, resizable=True
        )

        self.window.on_resize = self._gl_setup
        self.window.on_close = self.window_closed_by_user
        self.batches = []

        self._gl_setup(width, height)
        self._light_setup()
        self.set_bgcolor(0, 0, 0)

    def close(self):
        self.window.close()

    def set_bgcolor(self, r, g, b):
        glClearColor(r, g, b, 1.0)

    def set_fov(self):
        aspect = float(self.window.width) / self.window.height
        znear = 0.1
        zfar = 100.0
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, aspect, znear, zfar)

    def look_at(self, eye, target, up):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        eye, target, up = list(eye), list(target), list(up)
        gluLookAt(*(eye + target + up))

    def add_batch(self, batch):
        self.batches.append(batch)

    def render(self, return_rgb_array=False):
        self.window.switch_to()
        self.window.dispatch_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for batch in self.batches:
            batch.draw()

        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr

    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1,:,0:3]

    #
    # private methods
    #

    # called on window resize
    def _gl_setup(self, width, height):
        glViewport(0, 0, width, height)
        glFrontFace(GL_CCW)
        glEnable(GL_DEPTH_TEST)
        self.set_fov()

    # should only be called once
    def _light_setup(self):
        glShadeModel(GL_SMOOTH)
        glEnable(GL_LIGHTING)
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(40.0, 100.0, 60.0, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat * 4)(0.2, 0.2, 0.2, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 4)(0.8, 0.8, 0.8, 1))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (GLfloat * 4)(0.4, 0.4, 0.4, 1))
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT1, GL_POSITION, (GLfloat * 4)(-200, -40.0, -20.0, 1))
        glLightfv(GL_LIGHT1, GL_AMBIENT, (GLfloat * 4)(0.1, 0.1, 0.1, 1))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (GLfloat * 4)(0.5, 0.5, 0.5, 1))
        glLightfv(GL_LIGHT1, GL_SPECULAR, (GLfloat * 4)(0.2, 0.2, 0.2, 1))
        glEnable(GL_LIGHT1)

    def window_closed_by_user(self):
        self.close()

class SceneNode(object):
    def _build_children(self, batch):
        if isinstance(self.children, type([])):
            for c in self.children:
                c.build(batch, self.pyg_grp)
        else:
            self.children.build(batch, self.pyg_grp)

class World(SceneNode):
    def __init__(self, children):
        self.children = children
        self.pyg_grp = None

    def build(self, batch):
        self._build_children(batch)

class Transform(SceneNode):
    def __init__(self, transform, children):
        self.t = transform
        self.children = children

    def build(self, batch, parent):
        self.pyg_grp = _PygTransform(self.t, parent=parent)
        self._build_children(batch)
        return self.pyg_grp

    def set_transform(self, t):
        self.pyg_grp.set_matrix(t)

class BackToFront(SceneNode):
    def __init__(self, children):
        self.children = children

    def build(self, batch, parent):
        self.pyg_grp = pyglet.graphics.Group(parent=parent)
        for i, c in enumerate(self.children):
            ordering = pyglet.graphics.OrderedGroup(i, parent=self.pyg_grp)
            c.build(batch, ordering)
        return self.pyg_grp

class Color(SceneNode):
    def __init__(self, color, children):
        self.color = color
        self.children = children

    def build(self, batch, parent):
        self.pyg_grp = _PygColor(self.color, parent=parent)
        self._build_children(batch)
        return self.pyg_grp

    def set_rgb(self, r, g, b):
        self.pyg_grp.set_rgb(r, g, b)

def transform_and_color(transform, color, children):
    return Transform(transform, Color(color, children))

class CheckerTexture(SceneNode):
    def __init__(self, children):
        self.children = children

    def build(self, batch, parent):
        self.pyg_grp = _PygCheckerTexture(parent=parent)
        self._build_children(batch)
        return self.pyg_grp


#
# these functions return 4x4 rotation matrix suitable to construct Transform
# or to mutate Transform via set_matrix
#
def translate(x):
    r = np.eye(4)
    r[:3,3] = x
    return r

def trans_and_rot(t, r):
    m = np.eye(4)
    m[:3,:3] = r
    m[:3,3] = t
    return m

def rotz(theta):
    r = np.eye(4)
    r[:2,:2] = _rot2d(theta)
    return r

def roty(theta):
    r = np.eye(4)
    r2d = _rot2d(theta)
    r[[0,0,2,2],[0,2,0,2]] = _rot2d(theta).flatten()
    return r

def rotx(theta):
    r = np.eye(4)
    r[1:3,1:3] = _rot2d(theta)
    return r

class _PygTransform(pyglet.graphics.Group):
    def __init__(self, transform=np.eye(4), parent=None):
        super().__init__(parent)
        self.set_matrix(transform)

    def set_matrix(self, transform):
        assert transform.shape == (4, 4)
        assert np.all(transform[3,:] == [0, 0, 0, 1])
        self.matrix_raw = (GLfloat * 16)(*transform.T.flatten())

    def set_state(self):
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glMultMatrixf(self.matrix_raw)

    def unset_state(self):
        glPopMatrix()

class _PygColor(pyglet.graphics.Group):
    def __init__(self, color, parent=None):
        super().__init__(parent)
        if len(color) == 3:
            self.set_rgb(*color)
        else:
            self.set_rgba(*color)

    def set_rgb(self, r, g, b):
        self.color = (r, g, b, 1)

    def set_rgba(self, r, g, b, a):
        self.color = (r, g, b, a)

    def alpha(self):
        return self.color[-1]

    def set_state(self):
        if self.alpha() < 1:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (GLfloat * 4)(*self.color))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(*self.color))

    def unset_state(self):
        if self.alpha() < 1:
            glDisable(GL_BLEND)

class _PygCheckerTexture(pyglet.graphics.Group):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        pattern = pyglet.image.CheckerImagePattern(
            color1=(30,30,30,255),
            color2=(50,50,50,255))
        # higher res makes nicer mipmaps
        res = 256
        self.tex = pattern.create_image(res, res).get_texture()
        glBindTexture(GL_TEXTURE_2D, self.tex.id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

        # anisotropic texturing helps a lot with checkerboard floors
        anisotropy = (GLfloat)()
        glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy)

    def set_state(self):
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (GLfloat * 4)(1,1,1,1))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(1,1,1,1))
        glEnable(self.tex.target)
        glBindTexture(self.tex.target, self.tex.id)

    def unset_state(self):
        glDisable(self.tex.target)

class _PygAlphaBlending(pyglet.graphics.Group):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def set_state(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def unset_state(self):
        glDisable(GL_BLEND)

Batch = pyglet.graphics.Batch

#
# these are the 3d primitives that can be added to a pyglet.graphics.Batch.
# construct them with the shape functions below.
#
class BatchElement(SceneNode):
    def build(self, batch, parent):
        self.batch_args[2] = parent
        batch.add(*self.batch_args)

class Mesh(BatchElement):
    def __init__(self, verts, normals=None, st=None):
        if len(verts.shape) != 2 or verts.shape[1] != 3:
            raise ValueError('verts must be an N x 3 NumPy array')

        N = verts.shape[0]
        assert int(N) % 3 == 0

        if st is not None:
            assert st.shape == (N, 2)

        if normals is None:
            # compute normals implied by triangle faces
            normals = deepcopy(verts)

            for i in range(0, N, 3):
                v0, v1, v2 = verts[i:(i+3),:]
                d0, d1 = (v1 - v0), (v2 - v1)
                n = _normalize(np.cross(d0, d1))
                normals[i:(i+3),:] = n

        self.batch_args = [N, pyglet.gl.GL_TRIANGLES, None,
            ('v3f/static', list(verts.flatten())),
            ('n3f/static', list(normals.flatten())),
        ]
        if st is not None:
            self.batch_args.append(('t2f/static', list(st.flatten())))

class TriStrip(BatchElement):
    def __init__(self, verts, normals):
        N, dim = verts.shape
        assert dim == 3
        assert normals.shape == verts.shape

        self.batch_args = [N, pyglet.gl.GL_TRIANGLE_STRIP, None,
            ('v3f/static', list(verts.flatten())),
            ('n3f/static', list(normals.flatten()))
        ]

class TriFan(BatchElement):
    def __init__(self, verts, normals):
        N, dim = verts.shape
        assert dim == 3
        assert normals.shape == verts.shape

        self.batch_args = [N, pyglet.gl.GL_TRIANGLE_FAN, None,
            ('v3f/static', list(verts.flatten())),
            ('n3f/static', list(normals.flatten()))
        ]

# a box centered on the origin
def box(x, y, z):
    v = box_mesh(x, y, z)
    return Mesh(v)

# cylinder sitting on xy plane pointing +z
def cylinder(radius, height, sections):
    v, n = cylinder_strip(radius, height, sections)
    return TriStrip(v, n)

# cone sitting on xy plane pointing +z
def cone(radius, height, sections):
    v, n = cone_strip(radius, height, sections)
    return TriStrip(v, n)

# arrow sitting on xy plane pointing +z
def arrow(radius, height, sections):
    v, n = arrow_strip(radius, height, sections)
    return TriStrip(v, n)

# sphere centered on origin, n tris will be about TODO * facets
def sphere(radius, facets):
    v, n = sphere_strip(radius, facets)
    return TriStrip(v, n)

# square in xy plane centered on origin
# dim: (w, h)
# srange, trange: desired min/max (s, t) tex coords
def rect(dim, srange=(0,1), trange=(0,1)):
    v = np.array([
        [1, 1, 0], [-1, 1, 0], [1, -1, 0],
        [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])
    v = np.matmul(v, np.diag([dim[0] / 2.0, dim[1] / 2.0, 0]))
    n = _withz(0 * v, 1)
    s0, s1 = srange
    t0, t1 = trange
    st = np.array([
        [s1, t1], [s0, t1], [s1, t0],
        [s0, t1], [s0, t0], [s1, t0]])
    return Mesh(v, n, st)

def circle(radius, facets):
    v, n = circle_fan(radius, facets)
    return TriFan(v, n)

#
# low-level primitive builders. return vertex/normal/texcoord arrays.
# good if you want to apply transforms directly to the points, etc.
#

# box centered on origin with given dimensions.
# no normals, but Mesh ctor will estimate them perfectly
def box_mesh(x, y, z):
    vtop = np.array([[x, y, z], [x, -y, z], [-x, -y, z], [-x, y, z]])
    vbottom = deepcopy(vtop)
    vbottom[:,2] = -vbottom[:,2]
    v = 0.5 * np.concatenate([vtop, vbottom], axis=0)
    t = np.array([[1, 3, 2,], [1, 4, 3,], [1, 2, 5,], [2, 6, 5,], [2, 3, 6,], [3, 7, 6,], [3, 4, 8,], [3, 8, 7,], [4, 1, 8,], [1, 5, 8,], [5, 6, 7,], [5, 7, 8,]]) - 1
    t = t.flatten()
    v = v[t,:]
    return v

# circle in the x-y plane
def circle_fan(radius, sections):
    t = np.linspace(0, 2 * np.pi, sections + 1)[:,None]
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    v = np.hstack([x, y, 0*t])
    v = np.vstack([[0, 0, 0], v])
    n = _withz(0 * v, 1)
    return v, n

# cylinder sitting on the x-y plane
def cylinder_strip(radius, height, sections):
    t = np.linspace(0, 2 * np.pi, sections + 1)[:,None]
    x = radius * np.cos(t)
    y = radius * np.sin(t)

    base = np.hstack([x, y, 0*t])
    top = np.hstack([x, y, height + 0*t])
    strip_sides = _to_strip(np.hstack([top[:,None,:], base[:,None,:]]))
    normals_sides = _withz(strip_sides / radius, 0)

    def make_cap(circle, normal_z):
        height = circle[0,2]
        center = _withz(0 * circle, height)
        strip = _to_strip(np.hstack([center[:,None,:], circle[:,None,:]]))
        normals = _withz(0 * strip, normal_z)
        return strip, normals

    vbase, nbase = make_cap(base, -1)
    vtop, ntop = make_cap(top, 1)
    return (
        np.vstack([strip_sides, vbase, vtop]),
        np.vstack([normals_sides, nbase, ntop]))

# cone sitting on the x-y plane
def cone_strip(radius, height, sections):
    t = np.linspace(0, 2 * np.pi, sections + 1)[:,None]
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    base = np.hstack([x, y, 0*t])

    top = _withz(0 * base, height)
    vside = _to_strip(np.hstack([top[:,None,:], base[:,None,:]]))
    base_tangent = np.cross(_npa(0, 0, 1), base)
    top_to_base = base - top
    normals = _normalize(np.cross(top_to_base, base_tangent))
    nside = _to_strip(np.hstack([normals[:,None,:], normals[:,None,:]]))

    base_ctr = 0 * base
    vbase = _to_strip(np.hstack([base_ctr[:,None,:], base[:,None,:]]))
    nbase = _withz(0 * vbase, -1)

    return np.vstack([vside]), np.vstack([nside])

# sphere centered on origin
def sphere_strip(radius, resolution):
    t = np.linspace(-1, 1, resolution)
    u, v = np.meshgrid(t, t)
    vtx = []
    panel = np.zeros((resolution, resolution, 3))
    inds = list(range(3))
    for i in range(3):
        panel[:,:,inds[0]] = u
        panel[:,:,inds[1]] = v
        panel[:,:,inds[2]] = 1
        norms = np.linalg.norm(panel, axis=2)
        panel = panel / np.tile(norms[:,:,None], (1, 1, 3))
        for _ in range(2):
            for j in range(resolution - 1):
                strip = deepcopy(panel[[j,j+1],:,:].transpose([1,0,2]).reshape((-1,3)))
                degen0 = deepcopy(strip[0,:])
                degen1 = deepcopy(strip[-1,:])
                vtx.extend([degen0, strip, degen1])
            panel *= -1
        inds = [inds[-1]] + inds[:-1]

    n = np.vstack(vtx)
    v = radius * n
    return v, n

# arrow sitting on x-y plane
def arrow_strip(radius, height, facets):
    cyl_r = radius
    cyl_h = 0.75 * height
    cone_h = height - cyl_h
    cone_half_angle = np.radians(30)
    cone_r = cone_h * np.tan(cone_half_angle)
    vcyl, ncyl = cylinder_strip(cyl_r, cyl_h, facets)
    vcone, ncone = cone_strip(cone_r, cone_h, facets)
    vcone[:,2] += cyl_h
    v = np.vstack([vcyl, vcone])
    n = np.vstack([ncyl, ncone])
    return v, n


#
# private helper functions, not part of API
#
def _npa(*args):
    return np.array(args)

def _normalize(x):
    if len(x.shape) == 1:
        return x / np.linalg.norm(x)
    elif len(x.shape) == 2:
        return x / np.linalg.norm(x, axis=1)[:,None]
    else:
        assert False

def _withz(a, z):
    b = 0 + a
    b[:,2] = z
    return b

def _rot2d(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])

# add degenerate tris, convert from N x 2 x 3 to 2N+2 x 3
def _to_strip(strip):
    s0 = strip[0,0,:]
    s1 = strip[-1,-1,:]
    return np.vstack([s0, np.reshape(strip, (-1, 3)), s1])

