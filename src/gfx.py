from PIL import Image
import math
import numpy as np

palette = {
    'colors': [(255, 0, 0)] * 256, # bright red to spot color errors
    'size':   0,
}
def add_color(r, g, b):
    """ add color to palette (should use max 256 colors) """
    idx = palette['size']
    assert(idx < 256)
    palette['size'] += 1
    palette['colors'][idx] = (r, g, b)
    return idx

def get_color(idx):
    return palette['colors'][idx]

black  = add_color(  0,   0,   0)
red    = add_color(255,   0,   0)
green  = add_color(  0, 255,   0)
blue   = add_color(  0,   0, 255)
yellow = add_color(255, 255,   0)
pink   = add_color(255,   0, 255)
cyan   = add_color(  0, 255, 255)
white  = add_color(255, 255, 255)

def blend_max(c1, c2):
    return max(c1, c2)  # kinda dumb cause c1, c2 are indexes into palette...
"""
def blend_add(c1, c2):
    c1, c2 = get_color(c1), get_color(c2)
    return tuple(min(255, a + b) for a, b in zip(c1, c2))
def blend_or(c1, c2):
    c1, c2 = get_color(c1), get_color(c2)
    return tuple(a | b for a, b in zip(c1, c2))
def blend_xor(c1, c2):
    c1, c2 = get_color(c1), get_color(c2)
    return tuple(a ^ b for a, b in zip(c1, c2))
def blend_max(c1, c2):
    c1, c2 = get_color(c1), get_color(c2)
    return tuple(max(a, b) for a, b in zip(c1, c2))
def blend_mult(C1, c2):
    c1, c2 = get_color(c1), get_color(c2)
    return tuple(int(round(a * b / 255)) for a, v in zip(c1, c2))
"""

class Screen:
    def __init__(self, W, H, fps, blend_fn=blend_max,
                 clear_color=black, border_color=white):
        self.W            = W
        self.H            = H
        self.fps          = fps
        self.blend_fn     = blend_fn
        self.clear_color  = clear_color

        self.buf  = np.empty((H+2, W+2), dtype=np.uint8)
        self.imgs = []

        # draw 1px border
        for x in range(W+2):
            self.buf[ 0, x] = border_color
            self.buf[-1, x] = border_color
        for y in range(H+2):
            self.buf[y,  0] = border_color
            self.buf[y, -1] = border_color

    def clear(self):
        for y in range(1, self.H+1):
            for x in range(1, self.W+1):
                self.buf[y, x] = self.clear_color

    def put_pixel(self, p, color):
        x, y = p
        if x < 0 or y < 0 or x >= self.W or y >= self.H:
            print(f"WARNING - out of bounds draw: {p}")
            return

        self.buf[y+1, x+1] = self.blend_fn(self.buf[y+1, x+1], color)

    def save_frame(self, scale=1):
        # [(r, g, b), (r, g, b), ...] -> [r, g, b, r, g, b, ...]
        plt = [x for c in palette['colors'] for x in c]

        h, w = self.buf.shape
        img = Image.fromarray(self.buf, 'P')
        img = img.resize((w * scale, h * scale), Image.NEAREST)
        img.putpalette(plt)
        self.imgs.append(img)

    def to_gif(self, fname, time_scale=1, loop=0):
        if fname[-4:] != ".gif":
            fname += ".gif"
        self.imgs[0].save(fname, save_all=True, append_images=self.imgs[1:],
                          loop=loop, duration=time_scale * 1/self.fps * 1000)

    def show(self, frame=-1):
        self.imgs[frame].show()

def rasterize_line(p0, p1):
    """
    Bresenham line algorithm
    look at wikipedia for an explanation this code is garbage
    """
    x0, y0 = p0
    x1, y1 = p1

    small_slope = abs(y1-y0) < abs(x1-x0)
    if small_slope:
        # for slopes w/ small mag, y = f(x)
        a0, a1 = x0, x1
        b0, b1 = y0, y1
    else:
        # otherwise x = f(y)
        a0, a1 = y0, y1
        b0, b1 = x0, x1

    if a0 > a1:
        # swap start/end so dep var goes from small->big
        a0, a1 = a1, a0
        b0, b1 = b1, b0

    da = a1 - a0
    db = b1 - b0
        
    i = 1
    if db < 0:
        i = -1
        db = -db
    D = 2*db - da  
    b = b0

    for a in range(a0, a1+1):
        if small_slope:  # bleh
            yield (a, b)
        else:
            yield (b, a)
        if D > 0:
            b = b + i
            D = D - 2*da
        D = D + 2*db    

"""
Homogeneous coordinates:
- http://deltaorange.com/2012/03/08/the-truth-behind-homogenous-coordinates/

Coordinate systems:
 coord-space:      object ----> world ---> camera ---------> clip --------> screen
 transform matrix:        model       view        projection      viewport?

object-space:
- aka local/model-space
- each object can use whatever units it wants, usually centered around origin
- use cartesian coords when defining object then convert to homog coords
  before applying transforms

- model matrix "places" object in world

world-space:                                               y            
- location/size relative to other models                +? ^  z
- units arbitrary, but i guess theres precision            | / +?
  loss w/ very big/small numbers                           |/           
                                                     <-----+-----> x                
- view matrix moves world in front of camera        -?     |    +?
                                                           |
camera-space:                                           -? v
- aka view/eye-space                                 
- camera is at (0:0:0)                                               

- projection matrix adds perspective and
  normalizes coords to +/-1 (NDC)

                                                             
clip-space:                                                y
- left handed coord system, half cube                   +1 ^  z
- anything outside +/-1 doesn't get rendered               | / +1
- viewport transform shifts/scales coords                  |/
  to the dimensions of the actual screen             <-----+-----> x
                                                    -1     |    +1
                                                           |
                                                        -1 v

screen-space:                                           z
- all coords should be integers now                    / +???
                                                      /
                                                     +-----------> x
                                                     |          +W
                                                     |
                                                  +H v
                                                     y
"""
def proj(screen, zfar, znear, fov):
    """ projection matrix to go from screenspace (3D) -> framebuffer (2D) """
    asp = screen.W / screen.H
    t   = tan(rad(fov/2))
    z   = zfar/(zfar-znear)
    return (
        (1/(asp*t),   0,       0, 0),
        (        0, 1/t,       0, 0),
        (        0,   0,       z, 1),
        (        0,   0, z*znear, 0),
    )
    """
    zd = zfar-znear
    return (
        (1/(asp*t), 0,                 0,                0),
        (        0, t,                 0,                0),
        (        0, 0,  -(zfar+znear)/zd, -2*zfar*znear/zd),
        (        0, 0,                -1,                0),
    )
    """
"""
framebuffer:                                         +-----------> x
- fully 2D                                           |          +W
- rasterize from here                                |
- by now all points should be integral            +H v
                                                     y         
"""

"""
Transformations

- cos(0) == 1, cos(pi/2) == 0       (x)
- sin(0) == 0, sin(pi/2) == 1       (y)
Dot product:
- a . b       == cos(theta)|a|b|
              == 1/2(ab + ba)
  if a |_ b:  == 0
  if a || b:  == |a||b|
  if a == b:  == |a|^2
Outer product:
- a /\ b      == B_xy(x^y) + B_xz(x^z) + B_yz(x^z)
              == 1/2(ab - ba)
              == sin(theta)|a||b|(x^y)
- if a |_ b   == |a||b|(a^b)

Rotors:
- http://marctenbosch.com/quaternions/

Cross product:
"""

"""
transformation matrices
"""
from math import sin, cos, tan
def sin(r):
    stats['sin_calls'] += 1
    return math.sin(r)
def cos(r):
    stats['cos_calls'] += 1
    return math.cos(r)
def tan(r):
    stats['tan_calls'] += 1
    return math.tan(r)

def ident(n):
    """ identity matrix """
    assert(n == 4) # :)
    return (
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    )
""" all angles are in radians """
def rotX(r):
    """ rotate around X axis """
    return (
        (1,       0,      0, 0),
        (0,  cos(r), sin(r), 0),
        (0, -sin(r), cos(r), 0),
        (0,       0,      0, 1),
    )
def rotY(r):
    """ rotate around Y axis """
    return (
        (cos(r), 0, -sin(r), 0),
        (     0, 1,       0, 0),
        (sin(r), 0,  cos(r), 0),
        (     0, 0,       0, 1),
    )
def rotZ(r):
    """ rotate around Z axis """
    return (
        ( cos(r), sin(r), 0, 0),
        (-sin(r), cos(r), 0, 0),
        (      0,      0, 1, 0),
        (      0,      0, 0, 1),
    )
def scale(X, Y, Z):
    """ scale """
    return (
        (X, 0, 0, 0),
        (0, Y, 0, 0),
        (0, 0, Z, 0),
        (0, 0, 0, 1),
    )
def scaleU(S):
    """ scale uniformly """
    return scale(S, S, S)

def trans(X, Y, Z):
    """ translate """
    return (
        (1, 0, 0, X),
        (0, 1, 0, Y),
        (0, 0, 1, Z),
        (0, 0, 0, 1),
    )

def dot_prod(vec1, vec2):
    stats['dot_prods'] += 1
    assert(len(vec1) == 4)
    assert(len(vec2) == 4)
    return sum(v1 * v2 for (v1, v2) in zip(vec1, vec2))



stats = {
    'frames':        0,
    'mat_mat_mults': 0,
    'mat_vec_mults': 0,
    'dot_prods':     0,
    'sin_calls':     0,
    'cos_calls':     0,
    'tan_calls':     0,
}
def print_stats():
    frames = stats['frames']
    keys = tuple(stats.keys())
    for key in keys:
        stats['avg_' + key] = stats[key] / frames
    print("\nstats:\n    " + "\n    ".join(f"{k}: {v}" for k, v in stats.items()))

def mult_mv(mat, vec):
    """ (matNN x vecN) -> vecN """
    stats['mat_vec_mults'] += 1
    # we should only be doing N=4 mults
    assert(len(vec) == 4)
    assert(len(mat) == 4) # num rows
    return [dot_prod(vec, mat[y]) for y in range(len(mat))]

def mmult(*matrices):
    """
    multiply some matrices together, right to left
    A*B*C*D <-> A*(B*(C*D))
    """
    ms = list(matrices)
    M = ms.pop()
    for m in reversed(ms):
        M = _mult_mm(m, M)
    return M
def _mult_mm(m1, m2):
    """
    matrix-matrix multiplication
    - matrix should be row major
    - (M_nm x M_mp) -> M_np
    """
    stats['mat_mat_mults'] += 1
    # print(f"  {m1}")
    # print(f"x {m2}")

    n = len(m1[0])
    m = len(m1)
    assert(m == len(m2[0]))
    p = len(m2)
    assert(n == 4)
    assert(m == 4)
    assert(p == 4)

    rows = []
    for y in range(p):
        row = []
        for x in range(n):
            col = [r[x] for r in m2]
            row.append(dot_prod(m1[y], col))
        rows.append(row)
    # print(f"= {rows}")
    
    return rows

PI = math.pi
def rad(deg):
    return deg / 180 * PI

def to_homog(vec3):
    """ cartesian coord to homogenous coord """
    x, y, z = vec3
    return (x, y, z, 1)

def to_screen(vec4, screen):
    """ homog coord -> 2D screen-space """
    x, y, z, w = vec4                                       # throw away z coord
    # assert(w != 0)
    if w == 0:  # TODO: idk why this is 0 sometimes
        w = 1
    x, y = x/w, y/w                                         # homog -> cart
    x, y = x/z, y/z                                         # perspective division
    x, y = int((x+1)/2*screen.W), int((-y+1)/2*screen.H)    # scale to screen
    return x, y

def render(screen, objects, PV):
    """ object order rendering "pipeline" """
    stats['frames'] += 1
    for obj in objects:
        PVM = mmult(PV, obj.M)
        for line in obj.lines:
            # "primitive assembly"
            c, (v0, v1) = line
            points = [obj.points[v0], obj.points[v1]]

            # "vertex shader"
            points = [to_homog(p) for p in points]          # convert to homog before transforms
            points = [mult_mv(PVM, p) for p in points]      # model -> world -> camera -> clip
            points = [to_screen(p, screen) for p in points] # clip  -> screen

            # rasterization
            for p in rasterize_line(*points):
                screen.put_pixel(p, c)
            # fragment shader?
            # color blending?
