import src.gfx as gfx

class Object:
    def __init__(self, points, lines, M=None):
        self.points = points
        self.lines = lines
        self.M = M

def cube(**args):
    return Object(
        points = [
            # base
            (-0.5, -0.5, -0.5), ( 0.5, -0.5, -0.5),
            ( 0.5, -0.5, +0.5), (-0.5, -0.5, +0.5),
            # top
            (-0.5,  0.5, -0.5), ( 0.5,  0.5, -0.5),
            ( 0.5,  0.5, +0.5), (-0.5,  0.5, +0.5),
        ],
        lines = [
            # base
            (gfx.cyan, (0, 1)), (gfx.cyan,   (1, 2)),
            (gfx.cyan, (2, 3)), (gfx.cyan,   (3, 0)),
            # top
            (gfx.cyan, (4, 5)), (gfx.cyan,   (5, 6)),
            (gfx.cyan, (6, 7)), (gfx.cyan,   (7, 4)),
            # middle
            (gfx.red,  (0, 4)), (gfx.green,  (1, 5)),
            (gfx.pink, (2, 6)), (gfx.yellow, (3, 7)),
        ],
        **args,
    )
"""
elem_buf = [
    # TODO: there's prob a more a compact way to do a cube
    gfx.LINE_STRIP,
    # base
    0, 1, 2, 3, 0,
    # top
    gfx.PRIM_RESTART,
    4, 5, 6, 7, 4,
    # middle
    gfx.PRIM_RESTART, 0, 4
    gfx.PRIM_RESTART, 1, 5
    gfx.PRIM_RESTART, 2, 6
    gfx.PRIM_RESTART, 3, 7
]
"""

def square(color, **args):
    return Object(
        points = [
            (-0.5, -0.5, 0.0), (-0.5,  0.5, 0.0),
            ( 0.5,  0.5, 0.0), ( 0.5, -0.5, 0.0),
        ],
        lines = [
            (color, (0, 1)), (color, (1, 2)),
            (color, (2, 3)), (color, (3, 0)),
        ],
        **args,
    )

"""
def square(color, **args):
    return Object(
        vert_buf = [
            (color, (-0.5, -0.5, 0.0)),
            (color, (-0.5,  0.5, 0.0)),
            (color, ( 0.5,  0.5, 0.0)),
            (color, ( 0.5, -0.5, 0.0)),
        ],
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
        ]
        elem_buf = [
            gfx.LINE_STRIP,
            0, 1, 2, 3, 0
        ]
        lines = [
            (color, (0, 1)), (color, (1, 2)),
            (color, (2, 3)), (color, (3, 0)),
        ],
        **args,
    )
"""
