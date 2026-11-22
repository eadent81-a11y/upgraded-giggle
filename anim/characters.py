"""Character rigs. Proportions tuned by visual iteration."""
import numpy as np
from core.renderer import Node, pr, mat, trs4

def Mm(**kw):
    return dict(mat(**kw))

M_STEEL      = Mm(albedo=(0.80, 0.82, 0.86), rough=0.28, refl=0.10, spec=0.7)
M_STEEL_D    = Mm(albedo=(0.50, 0.52, 0.57), rough=0.42, refl=0.06)
M_DARK       = Mm(albedo=(0.13, 0.14, 0.17), rough=0.52, spec=0.6)
M_EYE        = Mm(albedo=(0.94, 0.95, 0.98), rough=0.06, refl=0.30, spec=1.0, sspec=0.12)
M_PUPIL      = Mm(albedo=(0.09, 0.10, 0.12))
M_SOCKET     = Mm(albedo=(0.20, 0.21, 0.24))
M_ROBOT_BLUE = Mm(albedo=(0.15, 0.33, 0.56), rough=0.36, refl=0.06)
M_RIPPLE     = Mm(albedo=(0.80, 0.83, 0.88), rough=0.22, refl=0.14, spec=0.8)
M_SCREEN     = Mm(albedo=(0.35, 0.6, 0.8), rough=0.5, emis=(5.0, 12.0, 16.0))
M_SCREEN_OFF = Mm(albedo=(0.20, 0.26, 0.34), rough=0.5)
M_RED_TIP    = Mm(albedo=(0.85, 0.25, 0.2), emis=(7.0, 1.1, 0.8))
M_ORANGE     = Mm(albedo=(0.88, 0.5, 0.15))
M_ORANGE_E   = Mm(albedo=(0.88, 0.5, 0.15), emis=(6.0, 2.2, 0.5))
M_WOOD       = Mm(albedo=(0.36, 0.27, 0.19), rough=0.6)
M_CLOTH      = Mm(albedo=(0.62, 0.34, 0.30), rough=0.8)
M_FLOOR      = Mm(albedo=(0.24, 0.26, 0.34), rough=0.28, refl=0.20, spec=0.8)
M_WALL       = Mm(albedo=(0.26, 0.30, 0.40), rough=0.5)
M_WALL_D     = Mm(albedo=(0.16, 0.19, 0.27), rough=0.5)
M_BELT       = Mm(albedo=(0.20, 0.21, 0.26), rough=0.42, refl=0.10)
M_NEON_P     = Mm(albedo=(0.35, 0.15, 0.4), emis=(8.0, 1.4, 7.5))
M_NEON_C     = Mm(albedo=(0.15, 0.3, 0.4), emis=(1.2, 6.5, 9.0))
M_NEON_G     = Mm(albedo=(0.15, 0.4, 0.25), emis=(1.0, 8.0, 3.5))
M_GOLD       = Mm(albedo=(0.85, 0.68, 0.30), rough=0.3, refl=0.25, spec=0.9)
M_GLASS      = Mm(albedo=(0.75, 0.85, 0.9), rough=0.06, refl=0.45, spec=1.0, sspec=0.15)
M_LEATHER    = Mm(albedo=(0.35, 0.22, 0.14), rough=0.5)
M_WHITEBOX   = Mm(albedo=(0.72, 0.70, 0.66), rough=0.5)
M_PINKBOX    = Mm(albedo=(0.62, 0.42, 0.44), rough=0.5)
M_TEALBOX    = Mm(albedo=(0.20, 0.44, 0.44), rough=0.5)
M_BLACK      = Mm(albedo=(0.06, 0.06, 0.08), rough=0.5)
M_WHITE      = Mm(albedo=(0.85, 0.85, 0.85))



def caps_scl(radius, total_len):
    """scale for 'caps' prim with cap centers at +-0.6 unit: total y-extent = 1.6*scl_y"""
    return (radius, total_len/1.6, radius)

def cyl_scl(radius, height):
    """scale for 'cyl' prim: unit half-height 0.5 -> total = scl_y"""
    return (radius, height, radius)

def _T(pos=(0,0,0), rot=(0,0,0), scl=(1,1,1)):
    return trs4(pos, rot, scl)


def build_pip(scene, parent=None, group='pip', screen_mat=None):
    """PIP: chibi keeper-bot. Proportions (root at floor): total height ~1.0."""
    parent = parent if parent is not None else scene.root
    R = scene.node('pip_root', (0, 0.0, 0), parent=parent)

    # ---- wheel (r=0.17, on floor)
    wheel = scene.node('wheel', (0, 0.17, 0), parent=R)
    scene.attach('cyl', wheel, _T((0,0,0),(np.pi/2,0,0), cyl_scl(0.17, 0.14)), M_DARK, group)
    scene.attach('cyl', wheel, _T((0,0,0),(np.pi/2,0,0), cyl_scl(0.175, 0.035)), M_ORANGE, group)
    scene.attach('cyl', wheel, _T((0,0,0),(np.pi/2,0,0), cyl_scl(0.055, 0.15)), M_STEEL_D, group)

    # ---- body: pill center y=0.42, spans 0.19..0.65, radius 0.21
    B = scene.node('body', (0, 0.0, 0), parent=R)
    scene.attach('caps', B, _T((0, 0.42-0.19, 0), (0,0,0), caps_scl(0.21, 0.46)), M_ROBOT_BLUE, group)
    scene.attach('sphere', B, _T((0, 0.23, 0.045), (0, 0.4, 0), (0.185, 0.17, 0.16)), M_RIPPLE, group)
    scr = scene.node('screen', (0, 0.23, 0.16), parent=B)
    scene.attach('box', scr, _T((0, 0, 0), (0, 0.4, 0), (0.19, 0.125, 0.01)), M_DARK, group)
    scr_box = scene.attach('box', scr, _T((0, 0, 0.004), (0, 0.4, 0), (0.15, 0.09, 0.004)), screen_mat or M_SCREEN, group)
    scene.attach('cyl', B, _T((0, 0.645+0.035, 0), (0,0,0), cyl_scl(0.05, 0.07)), M_STEEL_D, group)

    # ---- head: node pivot at y 0.66 (neck top); sphere center rel (0, 0.18), r~0.22
    H_ = scene.node('head', (0, 0.66, 0), parent=R)
    scene.attach('sphere', H_, _T((0, 0.18, 0), (0,0,0), (0.235, 0.215, 0.225)), M_ROBOT_BLUE, group)
    scene.attach('sphere', H_, _T((0, 0.175, 0.035), (0, 0.42, 0), (0.20, 0.175, 0.20)), M_RIPPLE, group)
    # antenna base + stems
    antL = scene.node('antL', (-0.20, 0.34, 0), parent=H_)
    antR = scene.node('antR', ( 0.20, 0.34, 0), parent=H_)
    scene.attach('caps', antL, _T((0, 0.0, 0), (0,0,0.5), caps_scl(0.016, 0.16)), M_STEEL_D, group)
    scene.attach('caps', antR, _T((0, 0.0, 0), (0,0,-0.5), caps_scl(0.016, 0.16)), M_STEEL_D, group)
    scene.attach('sphere', antL, _T((0, 0.09, 0), (0,0,0), (0.026,)*3), M_RED_TIP, group)
    scene.attach('sphere', antR, _T((0, 0.09, 0), (0,0,0), (0.026,)*3), M_RED_TIP, group)
    # eyes: dark socket, white ball, pupil, highlight
    eyes = []
    for sx in (-1, 1):
        sock = scene.node(f'socket{sx}', (sx*0.092, 0.225, 0.190), parent=H_)
        scene.attach('sphere', sock, _T((0,0,0),(0,0,0),(0.068,0.062,0.058)), M_SOCKET, group)
        ball = scene.node(f'ball{sx}', (0, 0, 0.022), parent=sock)
        scene.attach('sphere', ball, _T((0,0,0),(0,0,0),(0.047,0.047,0.042)), M_EYE, group)
        pup = scene.node(f'pup{sx}', (0, 0, 0.030), parent=ball)
        scene.attach('sphere', pup, _T((0,0,0),(0,0,0),(0.020,0.020,0.008)), M_PUPIL, group)
        hl = scene.node(f'hl{sx}', (0.015, 0.015, 0.024), parent=ball)
        scene.attach('sphere', hl, _T((0,0,0),(0,0,0),(0.008,0.008,0.005)), Mm(albedo=(1,1,1), emis=(14,14,13)), group)
        eyes.append(dict(sock=sock, ball=ball, pup=pup, hl=hl))
    # brows
    browL = scene.node('browL', (-0.092, 0.302, 0.212), parent=H_)
    browR = scene.node('browR', ( 0.092, 0.302, 0.212), parent=H_)
    scene.attach('box', browL, _T((0,0,0),(0,0,0.14),(0.10, 0.013, 0.010)), M_DARK, group)
    scene.attach('box', browR, _T((0,0,0),(0,0,-0.14),(0.10, 0.013, 0.010)), M_DARK, group)
    # mouth
    mouth = scene.node('mouth', (0, 0.113, 0.222), parent=H_)
    mouth_p = scene.attach('box', mouth, _T((0,0,0),(0,0,0),(0.11, 0.011, 0.008)), M_DARK, group)

    # ---- arms: shoulder pivots at (±0.225, 0.50); capsule points down when rot=0
    armL = scene.node('armL', (-0.225, 0.50, 0), parent=R)
    armR = scene.node('armR', ( 0.225, 0.50, 0), parent=R)
    scene.attach('sphere', armL, _T((0,0,0),(0,0,0),(0.062,)*3), M_STEEL_D, group)
    scene.attach('sphere', armR, _T((0,0,0),(0,0,0),(0.062,)*3), M_STEEL_D, group)
    upperL = scene.node('upperL', (0, -0.035, 0), parent=armL)
    upperR = scene.node('upperR', (0, -0.035, 0), parent=armR)
    scene.attach('caps', upperL, _T((0, 0, 0), (0,0,0), caps_scl(0.040, 0.21)), M_STEEL, group)
    scene.attach('caps', upperR, _T((0, 0, 0), (0,0,0), caps_scl(0.040, 0.21)), M_STEEL, group)
    handL = scene.node('handL', (0, -0.115, 0), parent=upperL)
    handR = scene.node('handR', (0, -0.115, 0), parent=upperR)
    scene.attach('sphere', handL, _T((0,0,0),(0,0,0),(0.048,)*3), M_STEEL_D, group)
    scene.attach('sphere', handR, _T((0,0,0),(0,0,0),(0.048,)*3), M_STEEL_D, group)

    return dict(root=R, body=B, head=H_, eyes=eyes, browL=browL, browR=browR,
                mouth=mouth, mouth_p=mouth_p, antL=antL, antR=antR,
                armL=armL, armR=armR, upperL=upperL, upperR=upperR,
                handL=handL, handR=handR, wheel=wheel, scr=scr, scr_box=scr_box)
