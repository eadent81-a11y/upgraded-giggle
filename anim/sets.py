"""Environment sets: Glimmer Station depot interior."""
import numpy as np
from core.renderer import Scene, trs4, mat
from anim.characters import (Mm, M_FLOOR, M_WALL, M_WALL_D, M_WOOD, M_DARK, M_BELT,
                             M_STEEL, M_STEEL_D, M_NEON_P, M_NEON_C, M_NEON_G, M_ORANGE,
                             M_GOLD, M_GLASS, M_LEATHER, M_WHITEBOX, M_PINKBOX, M_TEALBOX, M_CLOTH,
                             M_BLACK, M_SCREEN, _T, caps_scl, cyl_scl)

M_RAINGLASS = Mm(albedo=(0.35, 0.5, 0.7), rough=0.06, refl=0.5, spec=1.0, sspec=0.15)
M_CLOUDY    = Mm(albedo=(0.10, 0.12, 0.2), rough=0.8)
M_TOOLBOX   = Mm(albedo=(0.75, 0.3, 0.15), rough=0.45)
M_UMBRELLA  = Mm(albedo=(0.7, 0.2, 0.25), rough=0.6)
M_CUP       = Mm(albedo=(0.8, 0.8, 0.8), rough=0.3, refl=0.1)
M_TICKET    = Mm(albedo=(0.92, 0.88, 0.7), rough=0.7)


def build_depot(scene, w=4.4, d=6.0, h=3.4, group='room'):
    """Main depot interior. Origin center floor. Window on -x wall, conveyor +z-> 0, dome hatch in ceiling."""
    # floor & walls
    scene.static('box', (0, -0.4, 0), (0,0,0), (w+2, 0.4, d+2), M_FLOOR, group)
    scene.static('box', (0, h/2-0.4, -d/2), (0,0,0), (w, h, 0.2), M_WALL, group)          # back wall
    scene.static('box', (-w/2-0.1, h/2-0.4, 0), (0,0,0), (0.2, h, d), M_WALL_D, group)    # left wall
    scene.static('box', ( w/2+0.1, h/2-0.4, 0), (0,0,0), (0.2, h, d), M_WALL_D, group)    # right wall
    # ceiling with skylight gap (|x|<0.32, full depth) covered by sliding hatch panels
    scene.static('box', (-1.31, h-0.26, 0), (0,0,0), (0.99, 0.12, d), M_WALL_D, group)    # ceiling left
    scene.static('box', ( 1.31, h-0.26, 0), (0,0,0), (0.99, 0.12, d), M_WALL_D, group)    # ceiling right
    scene.static('box', (0, h-0.26, -2.5), (0,0,0), (0.64, 0.12, 1.0), M_WALL_D, group)   # ceiling trim back
    scene.static('box', (0, h-0.26, 2.5), (0,0,0), (0.64, 0.12, 1.0), M_WALL_D, group)    # ceiling trim front
    # floor tiles (grid seams via thin dark boxes on top)
    for i in range(-2, 3):
        scene.static('box', (i*1.0, 0.003, 0), (0,0,0), (0.02, 0.004, d*0.9), M_BLACK, group)
    for j in range(-2, 4):
        scene.static('box', (0, 0.003, j*1.0), (0,0,0), (w*0.9, 0.004, 0.02), M_BLACK, group)
    # door frame on back wall (props office door)
    scene.static('box', (1.55, 1.1, -d/2+0.12), (0,0,0), (0.9, 2.2, 0.08), M_WOOD, group)
    scene.static('box', (1.55, 1.1, -d/2+0.09), (0,0,0), (0.7, 1.9, 0.06), M_WALL_D, group)
    scene.static('sphere', (1.29, 1.05, -d/2+0.045), (0,0,0), (0.03,)*3, M_GOLD, group)
    # rain window on left wall (big glass + city glow panel behind)
    scene.static('box', (-w/2-0.12, 1.7, -0.4), (0, 0.0, 0), (0.06, 1.7, 2.2), M_CLOUDY, group)
    scene.static('box', (-w/2-0.06, 1.7, -0.4), (0, 0, 0), (0.05, 1.55, 2.0), M_RAINGLASS, group)
    mull = Mm(albedo=(0.35, 0.38, 0.44))
    for k in range(3):
        scene.static('box', (-w/2-0.05, 1.7, -1.35+k*0.95), (0,0,0), (0.07, 1.62, 0.045), mull, group)
    scene.static('box', (-w/2-0.05, 0.88, -0.4), (0,0,0), (0.07, 0.06, 2.0), mull, group)
    scene.static('box', (-w/2-0.05, 2.52, -0.4), (0,0,0), (0.07, 0.06, 2.0), mull, group)
    # rain streaks (thin emissive-ish shiny boxes at slight offsets)
    mstreak = Mm(albedo=(0.5, 0.6, 0.8), rough=0.05, refl=0.3, emis=(0.25, 0.4, 0.7))
    rng = np.random.default_rng(11)
    for k in range(26):
        y = rng.uniform(1.0, 2.4); z = rng.uniform(-1.35, 0.55); ln = rng.uniform(0.08, 0.30)
        scene.static('cyl', (-w/2+0.015, y, z), (0, rng.uniform(0,0.4), 0), cyl_scl(0.006, ln), mstreak, 'rain')
    # neon signs on back wall
    scene.static('box', (-0.7, 2.35, -d/2+0.22), (0,0,0), (1.3, 0.34, 0.05), M_NEON_P, group)   # big pink
    scene.static('box', (0.4, 2.7, -d/2+0.22), (0,0,0), (0.55, 0.14, 0.05), M_NEON_C, group)
    scene.static('box', (-0.7, 2.35, -d/2+0.13), (0,0,0), (1.22, 0.26, 0.04), Mm(albedo=(0.3,0.1,0.3)), group)
    # hanging lamps (2)
    for lx in (-1.3, 1.3):
        scene.static('cyl', (lx, h-0.45, -0.9), (0,0,0), cyl_scl(0.005, 0.7), M_STEEL_D, group)
        scene.static('cone', (lx, h-0.78, -0.9), (0,0,0), (0.30, 0.24, 0.30), M_STEEL, group)  # shade
        scene.static('sphere', (lx, h-0.92, -0.9), (0,0,0), (0.05,)*3, Mm(albedo=(1,0.9,0.7), emis=(16, 12, 7)), group)
    # shelf wall (left of door)
    for sy, wd in ((1.0, 0.6), (1.75, 0.6), (2.5, 0.6)):
        scene.static('box', (-w/2+0.45, sy, -d/2+1.05), (0, 0.0, 0), (0.55, 0.05, 1.7), M_WOOD, group)
    scene.static('box', (-w/2+0.45, 0.6, -d/2+1.05), (0,0,0), (0.52, 1.9, 0.05), M_WOOD, group)
    # shelf clutter
    scene.static('box', (-w/2+0.45, 0.34, -d/2+0.7), (0, 0.3, 0), (0.16, 0.16, 0.16), M_PINKBOX, group)
    scene.static('box', (-w/2+0.45, 0.34, -d/2+1.1), (0, -0.2, 0), (0.2, 0.2, 0.2), M_TEALBOX, group)
    scene.static('cone', (-w/2+0.42, 1.17, -d/2+0.8), (0,0,0), (0.10, 0.26, 0.10), M_UMBRELLA, group)
    scene.static('cyl', (-w/2+0.45, 0.88, -d/2+0.75), (0, 0, 1.51), cyl_scl(0.02, 0.34), M_UMBRELLA, group)
    # clocks & ticket sign
    scene.static('box', (0.4, 2.7, -d/2+0.13), (0,0,0), (0.48, 0.08, 0.04), Mm(albedo=(0.2,0.2,0.22)), group)
    # conveyor (belt along z from back wall to front)
    scene.static('box', (0.0, 0.34, -1.0), (0,0,0), (0.95, 0.14, 4.6), M_STEEL_D, group)
    scene.static('box', (0.0, 0.40, -1.0), (0,0,0), (0.80, 0.035, 4.6), M_BELT, group)
    scene.static('box', (0.0, 0.34, 1.35), (0,0,0), (0.95, 0.14, 0.2), M_STEEL_D, group)
    scene.static('cyl', (0.0, 0.34, -3.2), (np.pi/2, 0, 0), cyl_scl(0.17, 0.9), M_STEEL, 'conveyor')
    # side rails
    for rx in (-0.5, 0.5):
        scene.static('box', (rx, 0.50, -1.0), (0,0,0), (0.05, 0.10, 4.6), M_ORANGE, group)
    # lost props on conveyor & floor
    scene.static('box', (0.62, 0.15, 0.7), (0, 0.5, 0), (0.3, 0.3, 0.3), M_WHITEBOX, group)   # parcel floor
    scene.static('box', (0.4, 0.47+0.14, 0.2), (0, 0.2, 0), (0.2, 0.2, 0.2), M_PINKBOX, group)
    scene.static('sphere', (0.45, 0.47+0.345, 0.2), (0,0,0), (0.055,)*3, M_GOLD, group)
    scene.static('cone', (0.45, 0.47+0.40, 0.2), (0,0,0), (0.03, 0.12, 0.03), M_UMBRELLA, group)
    scene.static('cyl', (0.45, 0.47+0.40, 0.2), (0, 0, 1.51), cyl_scl(0.012, 0.2), M_UMBRELLA, group)
    scene.static('box', (0.45, 0.47+0.345, 0.34), (0, 0.6, 0), (0.1, 0.12, 0.1), M_CLOTH, group)
    # counter / desk with stamp
    scene.static('box', (1.45, 0.42, 0.9), (0, 0.2, 0), (0.8, 0.8, 0.8), M_WOOD, group)
    scene.static('box', (1.45, 0.855, 0.9), (0, 0.0, 0), (0.85, 0.05, 0.85), M_WOOD, group)
    scene.static('box', (1.30, 0.885, 0.75), (0, 0.15, 0), (0.2, 0.015, 0.12), M_TICKET, group)
    scene.static('box', (1.62, 0.895, 1.1), (0, -0.4, 0), (0.16, 0.015, 0.11), M_TICKET, group)
    scene.static('cyl', (1.45, 0.895, 1.1), (0.2, 0, 0.15), cyl_scl(0.055, 0.11), M_TOOLBOX, group)
    scene.static('cyl', (1.45, 0.895, 1.1), (0.2, 0, 0.15), cyl_scl(0.03, 0.015), M_BLACK, group)
    return None
