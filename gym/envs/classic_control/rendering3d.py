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

RAD2DEG = 57.29577951308232

def normalize(x):
    return x / np.linalg.norm(x)

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

class Viewer(object):
    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width
        self.height = height
        self.window = pyglet.window.Window(width=width, height=height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.geoms = []
        self.onetime_geoms = []
        #glEnable(GL_BLEND)
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def look_at(self, eye, target, up):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        eye, target, up = list(eye), list(target), list(up)
        gluLookAt(*(eye + target + up))

    def render(self, return_rgb_array=False):
        self.window.switch_to()
        self.window.dispatch_events()

        glClearColor(0,0,0,1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, self.width, self.height)
        glFrontFace(GL_CCW)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        fov = 60.0
        aspect = float(self.width) / self.height
        znear = 0.001
        zfar = 1000.0
        gluPerspective(fov, aspect, znear, zfar)

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

        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()

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

def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])

class Geom(object):

    def __init__(self):
        self._color=Color((0, 0, 0, 1.0))
        self.attrs = [self._color]
    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()
    def render1(self):
        raise NotImplementedError
    def add_attr(self, attr):
        self.attrs.append(attr)
    def set_color(self, r, g, b):
        self._color.vec4 = (r, g, b, 1)
        return self
    def set_translate(self, x, y, z):
        translations = [a for a in self.attrs if isinstance(a, Translate)]
        assert len(translations) <= 1
        if len(translations) == 1:
            translations[0].t = (x, y, z)
        else:
            self.translate(x, y, z)
        return self

    def set_rotation(self, mat):
        rotations = [i for i, a in enumerate(self.attrs) if isinstance(a, Rotate)]
        assert len(rotations) <= 1
        if len(rotations) == 1:
            i = rotations[0]
            self.attrs[i] = Rotate(mat)
        else:
            self.add_attr(Rotate(mat))
        return self

    def translate(self, x, y, z):
        self.add_attr(Translate(x, y, z))
        return self

class Attr(object):
    def enable(self):
        raise NotImplementedError
    def disable(self):
        pass

class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1,1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)
    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0) # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)
    def disable(self):
        glPopMatrix()
    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))
    def set_rotation(self, new):
        self.rotation = float(new)
    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))

class Translate(Attr):
    def __init__(self, x, y, z):
        self.t = (x, y, z)
    def enable(self):
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(*self.t)
    def disable(self):
        glPopMatrix()

class Rotate(Attr):
    def __init__(self, matrix):
        m1 = np.vstack([matrix, np.array([0, 0, 0])])
        matrix = np.hstack([m1, np.array([0, 0, 0, 1])[:,None]])
        mf = matrix.T.flatten()
        self.matrix_raw = (GLfloat * 16)(*matrix.T.flatten())
    def enable(self):
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glMultMatrixf(self.matrix_raw)
    def disable(self):
        glPopMatrix()

def _rot2d(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])

def rotz(theta):
    r = np.eye(3)
    r[:2,:2] = _rot2d(theta)
    return r

class CheckerTexture(Attr):
    def __init__(self):
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
        print("max anisotropy:", anisotropy)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy)

    def enable(self):
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (GLfloat * 4)(1,1,1,1))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(1,1,1,1))
        glEnable(self.tex.target)
        glBindTexture(self.tex.target, self.tex.id)

    def disable(self):
        glDisable(self.tex.target)

class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4
    def enable(self):
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (GLfloat * 4)(*self.vec4))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(*self.vec4))

class LineStyle(Attr):
    def __init__(self, style):
        self.style = style
    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)
    def disable(self):
        glDisable(GL_LINE_STIPPLE)

class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke
    def enable(self):
        glLineWidth(self.stroke)

class Point(Geom):
    def __init__(self):
        Geom.__init__(self)
    def render1(self):
        glBegin(GL_POINTS) # draw point
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()

class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v
    def render1(self):
        if   len(self.v) == 4 : glBegin(GL_QUADS)
        elif len(self.v)  > 4 : glBegin(GL_POLYGON)
        else: glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()

def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2*math.pi*i / res
        points.append((math.cos(ang)*radius, math.sin(ang)*radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)

def make_polygon(v, filled=True):
    if filled: return FilledPolygon(v)
    else: return PolyLine(v, True)

def make_polyline(v):
    return PolyLine(v, False)

def make_capsule(length, width):
    l, r, t, b = 0, length, width/2, -width/2
    box = make_polygon([(l,b), (l,t), (r,t), (r,b)])
    circ0 = make_circle(width/2)
    circ1 = make_circle(width/2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom

class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
    def render1(self):
        for g in self.gs:
            g.render()
    def set_color(self, r, g, b):
        for gm in self.gs:
            gm.set_color(r, g, b)

class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)
    def render1(self):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()
    def set_linewidth(self, x):
        self.linewidth.stroke = x

class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()




class Mesh(Geom):
    def __init__(self, verts, uvs=None):
        if len(verts.shape) != 2 or verts.shape[1] != 3:
            raise ValueError('verts must be an N x 3 NumPy array')

        nverts = verts.shape[0]
        assert int(nverts) % 3 == 0

        if uvs is not None:
            assert uvs.shape == (nverts, 2)

        super().__init__()

        # implicit normals
        normals = deepcopy(verts)

        for i in range(0, nverts, 3):
            v0, v1, v2 = verts[i:(i+3),:]
            d0, d1 = (v1 - v0), (v2 - v1)
            n = normalize(np.cross(d0, d1))
            normals[i:(i+3),:] = n

        args = [
            ('v3f', list(verts.flatten())),
            ('n3f', list(normals.flatten())),
        ]
        if uvs is not None:
            args.append(('t2f', list(uvs.flatten())))

        self.vertex_list = pyglet.graphics.vertex_list(nverts, *args)

    def render1(self):
        self.vertex_list.draw(GL_TRIANGLES)


class Strip(Geom):
    def __init__(self, verts, normals):
        N, k, dim = verts.shape
        assert k == 2
        assert dim == 3
        assert normals.shape == verts.shape

        super().__init__()

        # TODO guess normals

        self.vertex_list = pyglet.graphics.vertex_list(N * 2,
            ('v3f', list(verts.flatten())),
            ('n3f', list(normals.flatten())))

    def render1(self):
        self.vertex_list.draw(GL_TRIANGLE_STRIP)


class Mesh_Indexed(Geom):
    def __init__(self, verts, tris):
        if len(verts.shape) != 2 or verts.shape[1] != 3:
            raise ValueError('verts must be an N x 3 NumPy array')
        if (len(tris.shape) != 2 or tris.shape[1] != 3 or np.any(np.mod(tris, 1) != 0)):
            raise ValueError('tris must be an N x 3 NumPy array of nonnegative integer values')
        nverts = verts.shape[0]
        ntris = tris.shape[0]
        if np.any(tris >= nverts):
            raise IndexError('tris contains an index >= the number of vertices')

        super().__init__()

        verts = verts.astype(np.float32)
        tris = tris.astype(np.uint32)

        self.vertex_list = pyglet.graphics.vertex_list_indexed(
            nverts,
            list(tris.flatten().astype(np.uint32)),
            ('v3f', list(verts.flatten())),
        )

    def render1(self):
        self.vertex_list.draw(GL_TRIANGLES)


# a box centered on the origin
class Box(Mesh):
    def __init__(self, x, y, z):
        vtop = np.array([[x, y, z], [x, -y, z], [-x, -y, z], [-x, y, z]])
        vbottom = deepcopy(vtop)
        vbottom[:,2] = -vbottom[:,2]
        v = 0.5 * np.concatenate([vtop, vbottom], axis=0)
        t = np.array([[1, 3, 2,], [1, 4, 3,], [1, 2, 5,], [2, 6, 5,], [2, 3, 6,], [3, 7, 6,], [3, 4, 8,], [3, 8, 7,], [4, 1, 8,], [1, 5, 8,], [5, 6, 7,], [5, 7, 8,]]) - 1
        t = t.flatten()
        assert len(t) == 2 * 3 * 6
        v = v[t,:]
        super().__init__(v)


# cylinder centered on the origin
class Cylinder(Mesh):
    def __init__(self, radius, height, sections):
        step = 2 * np.pi / sections;
        theta = step * np.arange(sections)
        theta = theta[:,None]
        vtop = np.concatenate([
            radius * np.cos(theta),
            radius * np.sin(theta),
            height / 2.0 * np.ones((sections,1))],
            axis=1)
        vbottom = deepcopy(vtop);
        vbottom[:,2] = -height / 2;
        verts = np.vstack([
            np.array([0, 0, height / 2.0]),
            vtop,
            np.array([0, 0, -height / 2.0]),
            vbottom])
        tris_top = np.hstack([
            np.ones((sections-1, 1)),
            np.arange(2,sections+1)[:,None],
            np.arange(3,sections+2)[:,None]])
        tris_top = np.vstack([tris_top, np.array([1, sections + 1, 2])])
        tris_bottom = sections + 1 + tris_top;
        v1 = np.arange(2, sections + 1)[:,None]
        v2 = v1 + 1;
        v3 = v1 + sections + 1;
        v4 = v3 + 1;
        tris_side = np.concatenate([
            np.concatenate([v1, v2, v3], axis=1),
            np.concatenate([v3, v2, v4], axis=1)],
            axis=0)
        tris_side = np.vstack([
            tris_side,
            np.array([sections + 1, 2, 2*(sections+1)]),
            np.array([2*(sections+1), 2, 2+sections+1])])
        tris = np.concatenate([tris_top, tris_bottom, tris_side]) - 1
        assert np.all(tris % 1 == 0)
        tris = tris.flatten().astype(np.uint32)
        v = verts[tris,:]
        super().__init__(v)

# a cone sitting on xy plane with radius r, height h, n facets
class Cone(Mesh):
    def __init__(self, radius, height, sections):
        n = sections
        r = radius
        h = height
        step = 2 * np.pi / n
        theta = step * np.arange(sections)[:,None]
        vbase = np.hstack([r * np.cos(theta), r * np.sin(theta), 0 * theta])
        v = np.vstack([[0, 0, 0], vbase, [0, 0, height]])
        tbase = np.hstack([
            np.ones((n-1,1)),
            np.arange(3,n+2)[:,None],
            np.arange(2,n+1)[:,None]])
        tbase = np.vstack([tbase, [1, 2, n+1]])
        tcone = deepcopy(tbase)
        tcone[:,0] = n+2;
        t = np.vstack([tbase, tcone])
        assert np.all(t % 1 == 0)
        t = t.flatten().astype(np.uint32) - 1
        v = v[t,:]
        super().__init__(v)


# make an arrow sitting on xy plane with tube radius r, total height h, n facets
class Arrow(Compound):
    def __init__(self, radius, height, facets):
        cyl_r = radius
        cyl_h = 0.7 * height
        cone_h = height - cyl_h
        cone_half_angle = np.radians(30)
        cone_r = cone_h * np.tan(cone_half_angle)
        cyl = Cylinder(cyl_r, cyl_h, facets).translate(0,0,cyl_h/2)
        cone = Cone(cone_r, cone_h, facets).translate(0,0,cyl_h)
        super().__init__([cyl, cone])

# square in xy plane centered on origin
class Rect(Mesh):
    # dim: (w, h)
    # urange, vrange: desired min/max (u, v) tex coords
    def __init__(self, dim, urange=(0,1), vrange=(0,1)):
        v = np.array([
            [1, 1, 0], [-1, 1, 0], [1, -1, 0],
            [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])
        vtx = np.matmul(v, np.diag([dim[0] / 2.0, dim[1] / 2.0, 0]))
        u0, u1 = urange
        v0, v1 = vrange
        uv = np.array([
            [u1, v1], [u0, v1], [u1, v0],
            [u0, v1], [u0, v0], [u1, v0]])
        super().__init__(vtx, uv)

class Sphere(Strip):
    def __init__(self, radius, resolution):
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

        strip = np.vstack(vtx).reshape(-1,2,3)
        super().__init__(radius * strip, strip)


class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False
    def render1(self):
        self.img.blit(-self.width/2, -self.height/2, width=self.width, height=self.height)

# ================================================================

class SimpleImageViewer(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display
    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0,0)
        self.window.flip()
    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False
    def __del__(self):
        self.close()
