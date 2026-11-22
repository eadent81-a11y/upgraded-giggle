"""Facial performance: blinking, gaze, brows, mouth, head bounce."""
import numpy as np
from core.renderer import trs4


EYE_T = {'neutral': ((0.42, 0.0), 1.0, (0.11, -0.035)),
         'sad':     ((0.52, 0.0), 0.94, (0.20, -0.035)),
         'happy':   ((0.34, 0.0), 1.0, (0.11, -0.035)),
         'shock':   ((0.34, 0.0), 1.0, (0.11, -0.035))}


def blink_val(t, blinks, dur=0.13):
    """1 = open, 0 = closed."""
    for b in blinks:
        dt = t - b
        if -0.02 <= dt <= dur:
            f = np.clip(dt/dur, 0.0, 1.0)
            return 1.0 - np.sin(np.pi*f)**0.9
    return 1.0


def apply_eyes(pip, t, gaze=(0, 0), mood='neutral', blink_open=1.0, pupil_zoom=1.0):
    """gaze: (-1..1, -1..1) screen-space look direction (x, y)."""
    (gy_s, gz_s), scl, brow_off = EYE_T[mood]
    for i, e in enumerate(pip['eyes']):
        e['ball'].rot = np.array([-gaze[1]*0.30, -gaze[0]*0.35, 0.0])
        px, py = 0.0135*2.7*gaze[0], 0.0135*2.7*gaze[1]
        e['pup'].pos = np.array([px, py, 0.030])
        e['hl'].pos = np.array([0.015 - px*0.4, 0.015 - py*0.4, 0.024])
        sc = e['ball'].scl.copy()
        base = np.array([0.047, 0.047, 0.042])
        open_y = scl*blink_open
        e['ball'].scl = base*sc[0]/0.047*np.array([1.0, max(0.06, open_y), 1.0])
        e['pup'].scl = np.array([0.020, 0.020*max(0.15, open_y), 0.008])*pupil_zoom
        if open_y < 0.2:
            e['hl'].pos = np.array([0, -0.01, -0.02])
        else:
            e['hl'].pos = np.array([0.015 - px*0.4, 0.015 - py*0.4, 0.024])


def apply_brows(pip, mood='neutral', energy=0.0):
    bl, br = pip['browL'], pip['browR']
    if mood == 'neutral':
        bzl = 0.14; bzr = -0.14; dy = 0.0
    elif mood == 'sad':
        bzl = 0.02; bzr = -0.02; dy = 0.004
    elif mood == 'happy':
        bzl = 0.24; bzr = -0.24; dy = 0.006
    elif mood == 'shock':
        bzl = 0.05; bzr = -0.05; dy = 0.014
    elif mood == 'angry':
        bzl = 0.28; bzr = -0.28; dy = -0.006
    elif mood == 'sus':
        bzl = 0.14; bzr = -0.14; dy = 0.004
    else:
        bzl = 0.14; bzr = -0.14; dy = 0.0
    bl.rot = np.array([0, 0, bzl]); br.rot = np.array([0, 0, bzr])
    base_y = 0.302
    bl.pos = np.array([-0.092, base_y + dy, 0.212])
    br.pos = np.array([ 0.092, base_y + dy, 0.212])


def apply_mouth(pip, openness=0.0, smile=-0.5, width=1.0):
    """openness 0..1 (talk); smile -1 frown .. +1 smile; width 0.6..1.4."""
    p = pip['mouth_p']
    oy = 0.011*(0.55+1.5*openness)
    rz = -(1.0-smile)*0.30
    p.local = trs4((0, 0, 0), (0, 0, rz), (0.11*width, max(0.001, oy), 0.004))
    m = pip['mouth']
    y = 0.113 - 0.010*(1.0-smile)*0.5 - 0.010*openness*0.35
    z = 0.222 + 0.004*(1-smile)
    m.pos = np.array([0, y, z])


def head_bounce(pip, t, energy_env=None, base=(0, 0.66, 0)):
    """subtle head bob while talking"""
    b = 0.0
    if energy_env is not None and np.isscalar(energy_env):
        e = float(energy_env)
        b = 0.006*e*np.sin(t*26.0)
    elif energy_env is not None and len(energy_env):
        idx = min(int(t*60), len(energy_env)-1)
        e = energy_env[idx]
        b = 0.006*e*np.sin(t*26.0)
    pip['head'].pos = np.array([base[0], base[1]+b, base[2]])
    return b
