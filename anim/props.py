"""GLINT the orb + SCRAP the drone."""
import numpy as np
from core.renderer import Scene, trs4, mat
from anim.characters import (Mm, M_DARK, M_STEEL, M_STEEL_D, M_RED_TIP, M_GOLD, _T,
                             caps_scl, cyl_scl)


def build_glint(scene, parent=None, group='glint'):
    parent = parent if parent is not None else scene.root
    G = scene.node('glint_root', (0, 0.6, 0), parent=parent)
    shell = scene.attach('sphere', G, _T((0,0,0),(0,0,0),(0.115,0.115,0.115)),
                         Mm(albedo=(0.9, 0.75, 0.5), rough=0.06, refl=0.5, spec=1.0, sspec=0.15, nocast=True), group)
    core = scene.attach('sphere', G, _T((0,0,0.0),(0,0,0),(0.085,0.085,0.085)),
                        Mm(albedo=(1.0, 0.8, 0.5), emis=(30.0, 17.0, 6.0), nocast=True), group)
    halo = scene.attach('sphere', G, _T((0,0,0),(0,0,0),(0.145,0.145,0.145)),
                        Mm(albedo=(0.35, 0.25, 0.15), emis=(6.0, 2.6, 0.6), nocast=True), group)
    return dict(root=G, shell=shell, core=core, halo=halo)


def build_scrap(scene, parent=None, group='scrap'):
    parent = parent if parent is not None else scene.root
    R = scene.node('scrap_root', (0, 2.6, -1.5), parent=parent)
    B = scene.node('scrap_body', (0, 0, 0), parent=R)
    p_body = scene.attach('caps', B, _T((0, 0, 0), (0, np.pi/2, 0), (0.19, 0.34, 0.19)),
                          Mm(albedo=(0.30, 0.30, 0.33), rough=0.35, refl=0.12, spec=0.8), group)
    for i in (-1, 1):
        scene.attach('cyl', B, _T((i*0.42, 0.02, 0), (0, 0, np.pi/2), cyl_scl(0.17, 0.05)),
                     Mm(albedo=(0.24, 0.25, 0.28), rough=0.3, refl=0.2), group)
        scene.attach('box', B, _T((i*0.30, 0.0, 0), (0,0,0), (0.13, 0.035, 0.035)), M_STEEL_D, group)
    eye = scene.attach('sphere', B, _T((0, 0.0, 0.205), (0,0,0), (0.115, 0.11, 0.10)),
                       Mm(albedo=(0.35, 0.08, 0.08), emis=(7.0, 0.6, 0.6), rough=0.06, spec=1.0, sspec=0.1), group)
    pupil = scene.attach('sphere', B, _T((0, 0, 0.30), (0,0,0), (0.036, 0.036, 0.02)),
                         Mm(albedo=(0.02,0.02,0.02), emis=(16.0, 1.2, 1.2)), group)
    # rotor nubs
    for i in (-1, 1):
        scene.attach('cyl', B, _T((i*0.42, 0.16, 0), (0,0,0), cyl_scl(0.06, 0.05)), M_STEEL_D, group)
        scene.attach('cyl', B, _T((i*0.42, 0.19, 0), (0,0,0), cyl_scl(0.29, 0.004)),
                     Mm(albedo=(0.35, 0.36, 0.4), rough=0.2, refl=0.2), group)
    # spotlight cone (pseudo beam)
    beam = scene.attach('cone', B, _T((0, -0.28, 0.34), (np.pi*0.62, 0, 0), (0.05, 1.7, 0.05)),
                        Mm(albedo=(0.35, 0.3, 0.2), emis=(1.35, 1.1, 0.6), nocast=True), group)
    return dict(root=R, body=B, eye=eye, pupil=pupil, beam=beam)
