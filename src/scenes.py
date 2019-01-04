from src.gfx import PI
import src.gfx as g
import src.obj as o

def fast(screen):
    """ Fast rendering test scene """
    time     = 0.5
    frames   = int(screen.fps * time)
    objects  = [
        o.cube(M=g.ident(4)),
        o.square(g.white, M=g.rotX(PI/2))
    ]
    P = g.proj(screen, fov=90, znear=0, zfar=1)
    def draw(i):
        PV = g.mmult(
            P,
            g.trans(0, 0, 1.5),
            g.rotX(PI/4), g.rotY(i/frames * PI/2)
        )
        g.render(screen, objects, PV)
    return draw, frames

def proj_test(screen):
    """ Test perspective projection params """
    time     = 9
    frames   = int(screen.fps * time)
    objects  = [o.cube(M=g.ident(4))]

    fov   = 90
    znear = -1
    zfar  = +1

    fns = [
        lambda f: (90 + 35*g.sin(2*PI*f),               znear,               zfar), # vary fov
        lambda f: (                  fov, znear + g.sin(PI*f),                  2), # vary znear
        lambda f: (                  fov,                  -2, zfar - g.sin(PI*f)), # vary zfar
    ]

    def draw(i):
        sections = len(fns)
        sec_frames = frames / sections
        section = int(i / sec_frames)
        f = (i % sec_frames) / sec_frames

        P = g.proj(screen, *fns[section](f))
        PV = g.mmult(
            P,
            g.trans(0, 0, 1.5),
            g.rotX(g.rad(30)),
            g.rotY(g.rad(10))
        )
        g.render(screen, objects, PV)
    return draw, frames

def rot_test(screen):
    """ Rotate cube around Y, X, then Z axis """
    time    = 9.0
    frames  = int(screen.fps * time)
    objects = [o.cube(M=g.ident(4))]
    P = g.proj(screen, fov=90, znear=0, zfar=1)

    def draw(i):
        if i < frames/3:
            rot = g.rotY
        elif i < frames*2/3:
            rot = g.rotX
        else:
            rot = g.rotZ

        theta = (2*PI * 3*i/frames) % (2*PI)
        PV = g.mmult(P, g.trans(0, 0, 1.5), rot(theta))

        g.render(screen, objects, PV)
    return draw, frames

def cube_spin(screen):
    """ Multiple rotations/translatons on a cube """
    time     = 9.0
    frames   = int(screen.fps * time)
    P = g.proj(screen, fov=90, znear=0, zfar=1)

    # static square thats flat on the ground
    square   = o.square(g.white)
    square.M = g.mmult(g.scaleU(1.5), g.rotX(PI/2))

    # cube that's on the square and moves
    cube     = o.cube()

    objects  = [cube, square]

    def draw(i):
        theta  = i/frames * 2*PI
        theta2 = (2*theta) % (2*PI)
        s = g.sin(theta) / 4
        c = g.cos(theta) / 4

        cube.M = g.mmult(*reversed((
            g.scaleU(0.5),          # scale down
            g.rotZ(theta2),         # do a barrel roll
            g.rotY(theta),          # turn around
            g.trans(0, 0.25, 0),    # move up onto the square
            g.trans(s, 0, c),       # move in a circle
        )))

        # move camera `d` away from origin, looking down at angle `a`
        d, a = 1.5, g.rad(15)
        V  = g.mmult(g.trans(0, 0, d), g.rotX(-a))
        PV = g.mmult(P, V)

        g.render(screen, objects, PV)
    return draw, frames
