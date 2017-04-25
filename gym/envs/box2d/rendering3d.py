"""
3D rendering framework
"""
from __future__ import division
import os
import six
import sys

if "Apple" in sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

from gym.utils import reraise
from gym import error

try:
    import pyglet
except ImportError as e:
    reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
    from pyglet.gl import *
except ImportError as e:
    reraise(prefix="Error occured while running `from pyglet.gl import *`",suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")

import pyglet.gl.glu as glu

import math
import numpy as np

RAD2DEG = 57.29577951308232

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
        self.transform = Transform()

        #glEnable(GL_BLEND)
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        self.tristrips = []

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        self.transform = Transform(
            translation=(-left*scalex, -bottom*scaley),
            scale=(scalex, scaley))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render_old(self, return_rgb_array=False):
        glClearColor(1,1,1,1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
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

    def lookat(self, pos, look, up, fov):
        self.campos = pos
        self.camlook = look
        self.camup = up
        self.camfov = fov

    def render(self, return_rgb_array=False):
        glClearColor(0.2,0.2,0.5,1.0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()


        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        #glFrustum(-1.0, 1.0, -1.0, 1.0, 1.5, 20.0)
        gluPerspective(self.camfov, 8.0 / 6.0, 0.005, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glu.gluLookAt(
            self.campos[0],  self.campos[1],  self.campos[2],
            self.camlook[0], self.camlook[1], self.camlook[2],
            self.camup[0],   self.camup[1],   self.camup[2])

        r = 0.7
        g = 0.4
        b = 0.2

        #glColor3f(r, g, b)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        light_x = self.xrange[0]
        light_z = light_x
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(100*light_x,0,100*np.abs(light_z)/3.0,1))


        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, (GLfloat * 4)(r, g, b, 1.0))
        glMaterialfv(GL_BACK, GL_DIFFUSE, (GLfloat * 4)(0, 0.3, 1.0, 1.0))

        self.batch.draw()

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
        return arr


    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1,:,0:3]

    def reset_terrain(self, z, xrange, yrange):

        for s in self.tristrips:
            s.delete()
        self.tristrips = []

        def mesh2v(x, y, z):
            assert(x.shape == y.shape)
            assert(x.shape == z.shape)
            N = x.size
            vN3 = np.empty((N, 3))
            vN3[:,0] = np.ravel(x)
            vN3[:,1] = np.ravel(y)
            vN3[:,2] = np.ravel(z)
            return vN3

        # TEMP DEBUG
        assert(xrange == yrange)

        nx, ny = z.shape
        N = nx * ny
        vx = np.linspace(*xrange, num=nx)
        vy = np.linspace(*yrange, num=ny)
        meshx, meshy = np.meshgrid(vx, vy)

        self.xrange = xrange

        vN3 = mesh2v(meshx, meshy, z)

        normx = np.diff(z, n=2, axis=0)
        normx = np.pad(normx, ((1,1),(0,0)), 'edge')
        normy = np.diff(z, n=2, axis=1)
        normy = np.pad(normy, ((0,0),(1,1)), 'edge')
        assert(normx.shape == normy.shape)
        normz = abs(vx[2] - vx[0]) + 0 * normx
        invmag = 1.0 / np.sqrt(normx**2 + normy**2 + normz**2)
        normx *= invmag
        normy *= invmag
        normz *= invmag

        normN3 = mesh2v(normx, normy, normz)

        def ith_strip(n3, i):
            v0 = n3[(i*ny):((i+1)*ny),:]
            v1 = n3[((i+1)*ny):((i+2)*ny),:]
            strip = np.empty((ny*2+2,3))
            strip[0,:] = v0[0,:]
            strip[1:-1:2,:] = v0
            strip[2::2,:] = v1
            strip[-1,:] = v1[-1,:]
            strip = strip.flatten()
            return strip

        n_strips = nx - 1
        self.batch = pyglet.graphics.Batch()
        for i in range(n_strips):
            v_strip = ith_strip(vN3, i)
            nv = v_strip.size // 3
            n_strip = ith_strip(normN3, i)
            assert(n_strip.size == v_strip.size)
            gl_strip = self.batch.add(nv, pyglet.gl.GL_TRIANGLE_STRIP, None, 
                ('v3f', v_strip), ('n3f', n_strip))
            self.tristrips.append(gl_strip)




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

class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4
    def enable(self):
        glColor4f(*self.vec4)

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
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]
    def render1(self):
        for g in self.gs:
            g.render()

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
