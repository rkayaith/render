import src.gfx as gfx

class Object:
    def __init__(self, vert_buf, elem_buf, M=None):
        self.vert_buf = vert_buf
        self.elem_buf = elem_buf
        self.M = M

def cube(**args):
    return Object(
        vert_buf = [
            # base
            gfx.cyan, -0.5, -0.5, -0.5,
            gfx.cyan,  0.5, -0.5, -0.5,
            gfx.cyan,  0.5, -0.5, +0.5,
            gfx.cyan, -0.5, -0.5, +0.5,
            # top
            gfx.pink, -0.5,  0.5, -0.5,
            gfx.pink,  0.5,  0.5, -0.5,
            gfx.pink,  0.5,  0.5, +0.5,
            gfx.pink, -0.5,  0.5, +0.5,
        ],
        # TODO: there's prob a more a compact way to do a cube
        elem_buf = [
            gfx.LINE_STRIP,
            # base
            0, 1, 2, 3, 0,
            # top
            gfx.PRIM_RESTART,
            4, 5, 6, 7, 4,
            # middle
            gfx.PRIM_RESTART, 0, 4,
            gfx.PRIM_RESTART, 2, 6,
            gfx.PRIM_RESTART, 5, 1,
            gfx.PRIM_RESTART, 7, 3,
        ],
        **args,
    )

def square(color, **args):
    return Object(
        vert_buf = [
            color, -0.5, -0.5, 0.0,
            color, -0.5,  0.5, 0.0,
            color,  0.5,  0.5, 0.0,
            color,  0.5, -0.5, 0.0,
        ],
        elem_buf = [
            gfx.LINE_LIST,
            0, 1,
            1, 2,
            2, 3,
            3, 0,
        ],
        **args,
    )
