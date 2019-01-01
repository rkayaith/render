import argparse
import inspect
import os
import traceback

import src.gfx as gfx
import src.scenes as scenes

scene_fns = dict(inspect.getmembers(scenes, inspect.isfunction))

def app_open(path):
    os.system(f"xdg-open {path}") # TODO: should use `open` on mac?

def run(args, fname, screen):
    try:
        draw, iters = scene_fns[args.scene](screen)
        for i in range(iters):
            print(f"frame: {i+1}/{iters}", end='\r')

            screen.clear()
            draw(i)
            screen.save_frame(args.scale)

            if args.preview:
                if i % int(screen.fps) == 0:
                    screen.imgs[-1].save(fname)
                if i == 0:
                    app_open(fname)
        return True
    except Exception as exc:
        traceback.print_exc()
        screen.save_frame()
        return False


# program arguments
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    '-p', '--preview', action='store_true',
    help='show some frames while rendering'
)
parser.add_argument(
    'scene', choices=scene_fns.keys(),
    help='scene function to call\n' +
         '\n'.join(f"  {s}:{f.__doc__.rstrip()}" for s, f in scene_fns.items())
)

render_args = parser.add_argument_group('render arguments')
render_args.add_argument(
    '-W', '--width', type=int, default=256,
    help='frame buffer width  (default: %(default)s)'
)
render_args.add_argument(
    '-H', '--height', type=int, default=256,
    help='frame buffer height (default: %(default)s)'
)

output_args = parser.add_argument_group('output arguments', 'options for the output gif')
output_args.add_argument(
    '-s', '--scale', type=int, default=1,
    help='factor to scale image size (default: %(default)s)'
)
output_args.add_argument(
    '-t', '--time-scale', type=int, default=1,
    help='factor to scale gif time length (default: %(default)s)'
)

args = parser.parse_args()


fname = f"_out/{args.scene}.gif"
screen = gfx.Screen(W=args.width, H=args.height, fps=50) # higher than 50 breaks the gifs
error = not run(args, fname, screen)
if error:
    fname = "_out/error.gif"
    screen.to_gif(fname, args.time_scale, loop=1)
else:
    screen.to_gif(fname, args.time_scale)
    gfx.print_stats()
print(f"saved to {fname}")
if not args.preview:
    app_open(fname)
