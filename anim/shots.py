"""GLIMMER STATION — full 48s choreography.

Global-time animation: every rig is driven by absolute time tracks, so any
worker process can render any shot range and continuity is guaranteed.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.renderer import Scene, trs4, mat
from core import anim as A

IO = A.EASINGS['io']
from anim.characters import (build_pip, Mm, M_STEEL, M_STEEL_D, M_DARK,
                             M_TEALBOX, caps_scl, cyl_scl, _T)
from anim.sets import build_depot, M_TICKET
from anim.props import build_glint, build_scrap
from anim import faces

FPS = 30
TOTAL = 48.0
W, H = 486, 864
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(ROOT, 'output', 'voice')
SFX_DIR = os.path.join(ROOT, 'output', 'sfx')

BELT_TOP = 0.4175
SHELF_X, SHELF_Z = -1.78, -1.9
SHELF_TOPS = (1.05, 1.80)
DOOR = (1.55, -2.6)
COUNTER = (1.45, 0.9)
HATCH_HX = 0.32
GLINT_START = np.array([0.12, BELT_TOP + 0.115, 0.30])
NEWITEM_POS = np.array([0.0, BELT_TOP + 0.145, 0.5])

DIALOGUE = [
    (3.60,  'pip',   'pip_item',   1.00),
    (7.40,  'pip',   'pip_egg',    1.00),
    (10.30, 'pip',   'pip_notegg', 1.00),
    (14.00, 'pip',   'pip_whoa',   1.12),
    (19.60, 'scrap', 'scrap_unfiled', 1.18),
    (22.30, 'pip',   'pip_uhoh',   1.12),
    (24.95, 'scrap', 'scrap_file', 1.18),
    (27.50, 'pip',   'pip_return', 1.05),
    (31.30, 'pip',   'pip_stamps', 1.10),
    (32.55, 'scrap', 'scrap_error', 1.20),
    (37.05, 'pip',   'pip_bye',    1.00),
    (44.60, 'pip',   'pip_comeon', 1.05),
]

SFX_CUES = [
    (1.05, 'boot_up', 0.45), (1.20, 'blink', 0.55), (2.55, 'clang_conveyor', 0.95),
    (2.95, 'servo1', 0.70), (3.65, 'boop', 0.65), (5.05, 'boop2', 0.45),
    (5.30, 'servo2', 0.65), (6.90, 'boop_dn', 0.45), (7.25, 'stamp', 0.95),
    (9.15, 'boop2', 0.5), (9.30, 'servo1', 0.55), (12.20, 'paper', 0.45),
    (13.20, 'sparkle_rise', 0.80), (13.85, 'glint_pulse', 0.55),
    (14.60, 'sparkle_rise2', 0.75), (15.40, 'glint_pulse2', 0.5),
    (18.05, 'dome_open', 0.95), (18.15, 'hatch_clunk', 0.85), (18.62, 'servo2', 0.5),
    (19.65, 'scan', 0.80), (21.30, 'whoosh', 0.65), (22.80, 'whoosh_dn', 0.55),
    (24.20, 'servo1', 0.45), (25.30, 'scan', 0.55),
    (26.05, 'whoosh', 0.75), (26.75, 'servo2', 0.65), (26.92, 'beltpack', 0.75),
    (27.55, 'whoosh', 0.65), (28.32, 'impact_soft', 0.80), (28.90, 'whoosh', 0.6),
    (29.52, 'impact_soft', 0.70), (30.95, 'stamp', 0.95), (31.05, 'impact_soft', 0.85),
    (32.55, 'error', 0.90), (33.10, 'spin_up', 0.75), (33.25, 'whoosh', 0.7),
    (34.30, 'servo1', 0.55), (35.08, 'impact_soft', 0.6),
    (36.30, 'star_rise', 0.85), (37.10, 'glint_pulse', 0.55), (40.40, 'sky_ignite', 0.95),
    (41.15, 'boop', 0.5), (43.35, 'servo2', 0.6), (44.62, 'boop2', 0.7),
    (45.60, 'boop', 0.45), (47.38, 'twitch', 0.95), (47.72, 'blink', 0.6),
]

MOTION_BLUR_WINDOWS = [(2.55, 3.15), (26.0, 26.95), (27.5, 28.45), (33.1, 33.65),
                       (35.9, 36.9), (40.35, 40.9), (47.35, 47.95)]

def exposure_at(t):
    if t < 2.88:
        return 1.18
    if t < 13.4:
        return 1.32
    if t < 18.0:
        return 1.22
    if t < 34.0:
        return 1.28
    if t < 40.4:
        return 1.18
    return 1.12


INPACTS = (2.55, 7.25, 26.95, 28.32, 29.52, 31.00, 40.40, 47.38)

BLINK_SKIP = [(13.35, 13.95), (21.55, 22.95), (47.30, 48.0)]


# ================================================================ helpers
def glint_pulse_v(t, speed=1.9):
    ph = (t * speed) % 1.0
    a = np.exp(-((ph - 0.22) / 0.10) ** 2)
    b = 0.55 * np.exp(-((ph - 0.55) / 0.13) ** 2)
    return 0.45 + 0.85 * a + 0.42 * b


def jump(t, t0, t1, x0, y0, z0, x1, y1, z1, hgt, squash=None):
    """Parabolic hop. Returns (pos, squash_scale)."""
    t = np.clip(t, t0, t1)
    f = A.ease_in_out((t - t0) / max(t1 - t0, 1e-9))
    x = x0 + (x1 - x0) * f
    z = z0 + (z1 - z0) * f
    arch = 4.0 * f * (1.0 - f)
    y = y0 + (y1 - y0) * f + hgt * arch
    sq = (1.0, 1.0, 1.0)
    if squash is not None:
        t01 = (t - t0) / max(t1 - t0, 1e-9)
        if t01 < 0.09:
            k = 1.0 - (t01 / 0.09)
            sq = (1.0 + 0.18 * k, 1.0 - 0.22 * k, 1.0 + 0.18 * k)
        elif t01 > 0.91:
            k = 1.0 - ((1.0 - t01) / 0.09)
            sq = (1.0 + 0.22 * k, 1.0 - 0.26 * k, 1.0 + 0.22 * k)
    return np.array([x, y, z]), sq


def rise_move(t, t0, t1, p0, p1, ease='io'):
    f = np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
    f = A.EASINGS[ease](f)
    return np.array(p0) + (np.array(p1) - np.array(p0)) * f


def starfield_factory(scene):
    def sky_fn(dirs):
        amp = float(getattr(scene, 'star_amp', 0.0))
        k = 42.0
        g = np.floor(dirs * k)
        h = np.sin(g[:, 0] * 127.1 + g[:, 1] * 311.7 + g[:, 2] * 74.7) * 43758.5453
        frac = h - np.floor(h)
        h2 = np.sin(g[:, 0] * 269.5 + g[:, 1] * 183.3 + g[:, 2] * 246.1) * 28001.8
        frac2 = h2 - np.floor(h2)
        mask = (frac > 0.986).astype(np.float64)
        tw = 0.6 + 0.4 * np.sin(getattr(scene, 'star_time', 0.0) * 2.1 + frac2 * 40.0)
        bright = mask * (0.35 + 1.9 * frac2) * tw * amp
        col = np.zeros_like(dirs)
        col += bright[:, None] * np.array([0.85, 0.9, 1.0])[None]
        t_up = np.clip(-dirs[:, 1] * 1.6, 0, 1)
        col += (np.array([0.030, 0.045, 0.10])[None] * t_up[:, None]) * amp
        horizon = np.clip(0.5 - dirs[:, 1] * 0.5, 0, 1)
        col += np.array([0.045, 0.035, 0.05])[None] * horizon[:, None] * 2.0
        flash = float(getattr(scene, 'sky_flash', 0.0))
        if flash > 0:
            col += np.array([0.9, 0.85, 1.0])[None] * flash
        return col * 0.55
    return sky_fn


# ================================================================ scene build
def build_scene():
    scene = Scene()
    scene.fog = {'color': (0.035, 0.045, 0.085), 'density': 0.026}
    build_depot(scene)

    st = {'scene': scene}
    screen_mat = Mm(albedo=(0.3, 0.5, 0.7), rough=0.5, emis=(4.0, 9.5, 12.0))
    st['pip'] = build_pip(scene, screen_mat=screen_mat)
    st['glint'] = build_glint(scene)
    st['scrap'] = build_scrap(scene)

    gl = st['glint']
    gl['root'].pos = GLINT_START.copy()

    # sliding hatch panels (open at 18.05 with SCRAP's entrance, stay open)
    hl = scene.node('hatchL', (HATCH_HX / 2 - 0.16, 3.26, 0), parent=scene.root)
    hr = scene.node('hatchR', (-HATCH_HX / 2 + 0.16, 3.26, 0), parent=scene.root)
    for nd, sx in ((hl, HATCH_HX / 2), (hr, -HATCH_HX / 2)):
        scene.attach('box', nd, _T((0, 0, 0), (0, 0, 0), (0.16, 0.05, d_hatchz())),
                     M_DARK, 'hatch')
    st['hatchL'], st['hatchR'] = hl, hr
    st['hatch0'] = np.array([HATCH_HX / 2 - 0.165, 3.26, 0])

    # ticket prop carried in PIP's left hand (scale-animated in)
    tick = scene.node('ticket', (0, -0.115, 0.02), parent=st['pip']['handL'])
    scene.attach('box', tick, _T((0, 0, 0), (0, 0.3, 0), (0.075, 0.006, 0.052)),
                 M_TICKET, 'props')
    st['ticket'] = tick

    # new mystery item for the closing beat
    ni = scene.node('newitem', NEWITEM_POS + np.array([0, 0, -2.5]), parent=scene.root)
    st['m_glow_new'] = Mm(albedo=(0.35, 0.8, 0.8), rough=0.1, refl=0.4, spec=1.0,
                          emis=(0.6, 3.2, 3.0), nocast=True)
    scene.attach('sphere', ni, _T((0, 0, 0), (0, 0, 0), (0.145,)*3), st['m_glow_new'], 'props')
    st['newitem'] = ni

    # projected ceiling stars (galaxy moment in shot 3)
    rng = np.random.default_rng(7)
    star_node = scene.node('ceilstars', (0, 3.02, 0), parent=scene.root)
    st['m_star'] = Mm(albedo=(0.7, 0.8, 1.0), emis=(0.0, 0.0, 0.0), nocast=True)
    for i in range(46):
        x = rng.uniform(-2.0, 2.0); z = rng.uniform(-2.8, 2.2)
        s = rng.uniform(0.012, 0.030)
        scene.attach('sphere', star_node, _T((x, 0, z), (0, 0, 0), (s,)*3), st['m_star'], 'stars')
    st['stars'] = star_node

    # galaxy projector beam from glint to ceiling (shot 3)
    beam = scene.node('gbeam', GLINT_START + np.array([0, 0.6, 0]), parent=scene.root)
    st['m_gbeam'] = Mm(albedo=(0.3, 0.4, 0.7), emis=(0.0, 0.0, 0.0), nocast=True)
    scene.attach('cone', beam, _T((0, 0, 0), (0, 0, 0), (0.55, 2.2, 0.55)),
                 st['m_gbeam'], 'stars')
    st['gbeam'] = beam

    # dust motes
    dust = scene.node('dust', (0, 1.7, -0.4), parent=scene.root)
    st['m_dust'] = Mm(albedo=(0.8, 0.85, 1.0), emis=(0.14, 0.16, 0.2), nocast=True)
    for i in range(22):
        x = rng.uniform(-1.9, 1.9); y = rng.uniform(-1.2, 1.3); z = rng.uniform(-2.4, 2.2)
        s = rng.uniform(0.006, 0.013)
        scene.attach('sphere', dust, _T((x, y, z), (0, 0, 0), (s,)*3), st['m_dust'], 'dust')
    st['dust'] = dust

    # ------------------------------------------------------------ lights
    L = []
    def add(pos, color, inten, radius, shadow=True):
        L.append({'pos': list(pos), 'color': list(color), 'int': inten,
                  'radius': radius, 'shadow': shadow})
        return L[-1]
    st['lamp1'] = add((-1.3, 2.50, -0.9), (1.0, 0.82, 0.58), 1.55, 0.36, shadow=True)
    st['lamp2'] = add((1.3, 2.50, -0.9), (1.0, 0.82, 0.58), 1.55, 0.36, shadow=True)
    st['city'] = add((-2.05, 1.7, -0.4), (0.22, 0.36, 0.62), 0.55, 1.3, shadow=False)
    st['fill'] = add((1.8, 2.2, 2.2), (0.55, 0.62, 0.85), 0.42, 2.2, shadow=False)
    st['glintLight'] = add(GLINT_START + (0, 0.12, 0), (1.0, 0.76, 0.42), 1.0, 0.75, shadow=False)
    st['scrapLight'] = add((0.9, 1.5, -0.7), (1.0, 0.12, 0.08), 0.0, 0.8, shadow=False)
    st['skyLight'] = add((0, 3.0, 0.3), (0.55, 0.66, 1.0), 0.0, 1.5, shadow=False)
    scene.lights = L

    scene.sky_fn = starfield_factory(scene)
    scene.star_amp = 0.0
    scene.star_time = 0.0
    scene.sky_flash = 0.0
    scene.amb_top = np.array([0.10, 0.11, 0.17])
    scene.amb_bot = np.array([0.045, 0.045, 0.065])

    st['mouth_env'] = build_mouth_env()
    st['blinks'] = build_blinks()
    st['glint_pos'] = GLINT_START.copy()
    st['scrap_pos'] = np.array([1.8, 3.9, -2.5])
    st['pip_pos'] = np.array([1.5, 0.0, -2.5])
    scene.refresh()
    return st


def d_hatchz():
    return 3.0


def build_mouth_env():
    """Global 60Hz lip-sync envelope over the full timeline."""
    try:
        sys.path.insert(0, ROOT)
        from audio.voice import load_wav, envelope_60hz
    except Exception:
        return np.zeros(2880)
    env = np.zeros(int(TOTAL * 60) + 4)
    try:
        with open(os.path.join(VOICE_DIR, 'durations.json')) as f:
            durs = json.load(f)
    except Exception:
        durs = {}
    for t0, ch, wav, gain in DIALOGUE:
        p = os.path.join(VOICE_DIR, wav + '.wav')
        if not os.path.exists(p):
            continue
        x = load_wav(p)
        e = envelope_60hz(x)
        i0 = int(round(t0 * 60))
        n = min(len(e), len(env) - i0)
        env[i0:i0 + n] = np.maximum(env[i0:i0 + n], e[:n] * 1.15)
    return env


def build_blinks(seed=3):
    rng = np.random.default_rng(seed)
    ts = []
    t = 0.7
    while t < TOTAL - 0.3:
        skip = any(a - 0.12 < t < b + 0.12 for a, b in BLINK_SKIP)
        if not skip:
            ts.append(round(t, 3))
        t += rng.uniform(0.75, 1.7)
    return ts


# ================================================================ performance
def gaze_of(st, t, target):
    """Convert world target to PIP screen-space gaze (-1..1)."""
    pip = st['pip']
    head = (pip['head'].world[:3, 3] + pip['root'].world[:3, :3] @ np.array([0, 0.18, 0])) \
        if hasattr(pip['head'], 'world') else np.array([0, 0.85, 0])
    base = pip['root'].world[:3, 3]
    ry = float(pip['root'].rot[1])
    fwd = np.array([np.sin(ry), 0.0, np.cos(ry)])
    right = np.array([np.cos(ry), 0.0, -np.sin(ry)])
    v = np.asarray(target, np.float64) - (base + np.array([0, 0.845, 0]))
    v = v / (np.linalg.norm(v) + 1e-9)
    gx = np.clip(np.dot(v, right[:2] * [1, 1]) if False else v @ right, -1, 1)
    gy = np.clip(v[1] * 2.6, -1, 1)
    return (float(np.clip(gx * 2.1, -1, 1)), float(gy))


def mood_at(t):
    """(eye_mood, brow_mood, smile, pupil_zoom) schedule."""
    S = [(0.0,      'neutral', 'neutral', -0.4, 1.0),
         (3.40,     'neutral', 'neutral', -0.2, 1.0),
         (3.60,     'happy',   'happy',   0.85, 1.0),
         (5.90,     'neutral', 'neutral', -0.35, 1.0),
         (7.40,     'happy',   'happy',   0.85, 1.0),
         (8.90,     'neutral', 'neutral', -0.1, 1.0),
         (10.30,    'shock',   'sus',     -0.55, 1.14),
         (12.85,    'shock',   'shock',   -0.7, 0.6),
         (13.75,    'happy',   'happy',   0.55, 0.8),
         (16.20,    'happy',   'happy',   0.65, 0.9),
         (17.90,    'neutral', 'neutral', -0.1, 1.0),
         (18.05,    'shock',   'shock',   -0.6, 0.75),
         (19.60,    'sad',     'angry',   -0.7, 1.1),
         (21.60,    'shock',   'shock',   -0.7, 0.6),
         (23.60,    'sad',     'sad',     -0.85, 1.1),
         (25.95,    'shock',   'angry',   -0.7, 0.7),
         (27.40,    'shock',   'angry',   -0.55, 0.85),
         (29.70,    'shock',   'sus',     -0.45, 1.1),
         (31.25,    'shock',   'happy',   0.65, 0.95),
         (32.30,    'shock',   'sus',     0.35, 1.12),
         (33.60,    'happy',   'happy',   0.85, 1.0),
         (34.80,    'happy',   'happy',   0.7, 0.95),
         (36.60,    'sad',     'sad',     -0.1, 1.0),
         (40.90,    'happy',   'sad',     0.35, 1.0),
         (42.60,    'sad',     'sad',     -0.85, 1.1),
         (43.90,    'shock',   'sus',     -0.45, 0.8),
         (44.70,    'shock',   'sus',     -0.45, 0.9),
         (46.30,    'shock',   'angry',   -0.6, 1.1),
         (47.38,    'shock',   'shock',   -0.7, 0.6),
         (48.0,     'shock',   'shock',   -0.7, 0.6)]
    for i in range(len(S) - 1):
        if S[i][0] <= t < S[i + 1][0]:
            return S[i][1], S[i][2], S[i][3], S[i][4]
    return S[-1][1], S[-1][2], S[-1][3], S[-1][4]


GAZE = [
    (0.0,   3.4,  GLINT_START),
    (3.4,   5.9,  GLINT_START),
    (5.9,   7.45, (1.45, 0.88, 1.1)),
    (7.45,  8.6,  GLINT_START),
    (8.6,   10.0, GLINT_START),
    (10.0,  13.4, 'HANDGLINT'),
    (13.4,  16.2, (0.2, 2.7, 0.0)),
    (16.2,  18.0, (0.0, 2.95, -0.2)),
    (18.0,  19.6, (0.6, 2.6, -0.4)),
    (19.6,  25.9, 'SCRAP'),
    (25.95, 27.5, (0.15, 0.7, 0.3)),
    (27.5,  30.9, (-2.0, 1.6, -2.5)),
    (30.9,  31.3, 'TICKET'),
    (31.3,  33.3, 'SCRAP'),
    (33.3,  34.3, (0.0, 2.9, -0.4)),
    (34.3,  36.4, (0.0, 2.9, -0.4)),
    (36.4,  40.3, 'HANDGLINT'),
    (40.3,  42.2, 'HANDGLINT'),
    (42.2,  44.3, (0.0, 0.6, -0.9)),
    (44.3,  48.0, (0.0, 0.6, 0.5)),
]

BLINK_RELEASE = {
    30.40: None,
}


def gaze_now(st, t):
    for t0, t1, tg in GAZE:
        if t0 <= t < t1:
            if isinstance(tg, str):
                if tg == 'HANDGLINT':
                    return st['glint_pos']
                if tg == 'SCRAP':
                    return st['scrap_pos']
                if tg == 'TICKET':
                    hp = st['pip']['handL'].world[:3, 3]
                    return hp
            return np.array(tg, np.float64)
    return np.array([0, 1.0, 0])


def open_env(st, t):
    i = int(np.clip(t * 60, 0, len(st['mouth_env']) - 1))
    return float(st['mouth_env'][i])


# ================================================================ PIP pose
def pip_track(st, t):
    """PIP root position/facing over absolute time. Returns wheel spin."""
    pos = np.array([1.5, 0.0, -2.5])
    ry, lean = 0.0, 0.0
    spin = 0.0
    if t < 2.88:
        pos = np.array([1.55, 0.0, -2.5])
        ry = 0.0
    elif t < 3.55:
        f = A.ease_in_out((t - 2.88) / 0.67)
        pos = np.array([1.55, 0.0, -2.5 + 4.2 * f])
        ry = 0.0
        spin = 4.2 * f / 0.17
    elif t < 5.30:
        pos = np.array([1.5, 0.0, 1.7])
        ry = 0.28 + 0.22 * A.smooth(np.clip((t - 3.55) / 1.0, 0, 1))
    elif t < 6.15:
        f = A.ease_in_out((t - 5.30) / 0.85)
        pos = np.array([1.5 - 0.55 * f, 0.0, 1.7 - 0.25 * f])
        ry = 0.92 + (2.18 - 0.92) * f
        spin = 0.8 * f / 0.17
    elif t < 7.75:
        pos = np.array([0.95, 0.0, 1.45])
        ry = 2.18
    elif t < 8.25:
        f = A.ease_in(np.clip((t - 7.75) / 0.5, 0, 1))
        ry = 2.18 + (-2.40 + 2.18 + 2 * np.pi) * f
        ry = ((ry + np.pi) % (2 * np.pi)) - np.pi
        pos = np.array([0.95, 0.0, 1.45])
    elif t < 8.95:
        f = A.ease_in_out((t - 8.25) / 0.7)
        pos = np.array([0.95 - 0.45 * f, 0.0, 1.45 - 0.6 * f])
        ry = -2.40
        spin = 0.75 * f / 0.17
    elif t < 10.40:
        f = A.ease_in_out((t - 8.95) / 1.45)
        pos = np.array([0.5 + 0.05 * f, 0.0, 0.85 + 0.25 * f])
        ry = -2.40 + (2 * np.pi - 2.40 - -0.35) * f
        ry = ((ry + np.pi) % (2 * np.pi)) - np.pi
    elif t < 12.90:
        pos = np.array([0.55, 0.0, 1.1])
        ry = -0.35
    elif t < 18.00:
        pos = np.array([0.55, 0.0, 1.1])
        ry = 0.10
    st['_last_pip'] = [pos.copy(), float(ry)]
    P = st['pip']['root']
    P.pos = pos
    P.rot = np.array([lean, ry, 0.0])
    return spin, (pos.copy(), float(ry))


def pip_face_and_body(st, t):
    """arms, head, squash, eye/face params. Returns (mode, squash)."""
    pip = st['pip']
    env = open_env(st, t)
    blinks = st['blinks']
    for a, b in BLINK_SKIP:
        if a <= t <= b:
            blinks = [x for x in blinks if not (a - 0.05 <= x <= b + 0.05)]

    eye_mood, brow_mood, smile, pz = mood_at(t)

    # ---------- arms / body acting
    aL, aR = pip['armL'], pip['armR']
    uL, uR = pip['upperL'], pip['upperR']
    mode = 'idle'
    lean_x, ry = 0.0, float(pip['root'].rot[1])
    squash = (1.0, 1.0, 1.0)

    ar_rx, ar_rz = 0.10, -0.28
    al_rx, al_rz = 0.10, 0.28
    uR_rx, uL_rx = 0.0, 0.0
    idle_b = 0.03 * np.sin(t * 2.1 + 1.0)

    if t < 3.55:                # rolling in
        mode = 'rollin'
        ar_rx, al_rx = 0.35, 0.35
        bob = 0.02 * np.sin(t * 16.0)
        uR_rx, uL_rx = 0.5, 0.5
    elif t < 5.30:              # proud announcement
        mode = 'announce'
        ar_rx, al_rx = 0.55, 0.30
        pip['head'].rot[0] = -0.08
    elif t < 7.75:              # walk to counter + stamp
        mode = 'work'
        if t < 6.15:
            ar_rx = 0.5 + 0.25 * np.sin(t * 14.0)
            bob = 0.015 * np.sin(t * 16.0)
        if 6.60 <= t < 6.90:
            uR_rx = -2.35 * A.ease_in((t - 6.60) / 0.30)
            ar_rx = -0.2
        elif 6.90 <= t < 7.05:
            uR_rx = -2.35
        elif 7.05 <= t < 7.30:
            uR_rx = -2.35 + (-1.05 + 2.35) * A.EASINGS['in']((t - 7.05) / 0.25)
        elif 7.30 <= t < 7.75:
            uR_rx = -1.05 + 0.10 * A.ease_out((t - 7.30) / 0.45)
            lean_x = -0.06
    elif t < 8.95:              # notice the egg
        mode = 'notice'
        uL_rx = -0.2
        pip['head'].rot[0] = 0.05
    elif t < 10.40:             # reach / grab / lift
        mode = 'grab'
        if t < 9.25:
            uR_rx = -1.35 * A.ease_in_out((t - 8.95) / 0.30)
            uL_rx = 0.5
            lean_x = 0.10
            ar_rx = 0.15
        elif t < 9.60:
            uR_rx = -1.35
            uL_rx = 0.5
            lean_x = 0.10
        else:
            f = A.ease_in_out((t - 9.60) / 0.35)
            uR_rx = -1.35 - 0.25 * f
            lean_x = 0.10 * (1 - f)
            ar_rz = -0.2
    elif t < 12.90:             # suspicious inspection
        mode = 'insp'
        uR_rx = -1.62
        uL_rx = -0.15
        pip['head'].rot[0] = 0.10 * np.sin(t * 1.4)
        ar_rz = -0.2
        lean_x = 0.14 * (0.5 + 0.5 * np.sin((t - 10.3) * 3.2)) * 0.6
    elif t < 13.65:             # shock hop back
        mode = 'shock'
        uR_rx = -2.0
        uL_rx = -2.0
        ar_rz = -0.6
        al_rz = 0.6
        lean_x = -0.18
    elif t < 18.00:             # wonder
        mode = 'wonder'
        uR_rx = -2.3 + 0.08 * np.sin(t * 2.2)
        al_rx = -0.2
        pip['head'].rot[0] = -0.25
        lean_x = -0.05
    elif t < 19.60:             # hatch snap
        mode = 'alarm'
        uR_rx = -2.3
        uL_rx = -0.2
    elif t < 21.55:             # fear, watching scrap
        mode = 'fear'
        uR_rx = -1.9
        uL_rx = -1.9
        ar_rz = -0.5
        al_rz = 0.5
    elif t < 23.70:             # startle + flee to hide
        mode = 'flee'
        uR_rx = -2.2
        uL_rx = -2.2
        ar_rz = -0.6
        al_rz = 0.6
    elif t < 25.95:             # hiding / peeking
        mode = 'hide'
        uR_rx = -2.35
        uL_rx = -2.35
        ar_rz = -0.55
        al_rz = 0.55
        pip['head'].rot[0] = 0.14
    elif t < 26.95:             # sprint to belt
        mode = 'run'
        uL_rx, uR_rx = 0.9, -0.4
        al_rx, ar_rx = 0.5 + 0.4 * np.sin(t * 18.0), 0.5 - 0.4 * np.sin(t * 18.0)
        ry = float(pip['root'].rot[1])
    elif t < 31.30:             # belt ride + shelf hops + ticket raise
        mode = 'climb'
        uL_rx = -0.2 + (-2.6 + 0.2) * A.ease_in_out(np.clip((t - 30.35) / 0.35, 0, 1))
        uR_rx = -1.55 if t > 29.0 else 0.35
        ar_rz = -0.2
        al_rz = 0.5 * A.ease_in_out(np.clip((t - 30.35) / 0.35, 0, 1))
    elif t < 31.95:             # the slam
        mode = 'slam'
        if t < 31.02:
            uL_rx = -2.6 + (-0.75 + 2.6) * A.EASINGS['in']((t - 30.95) / 0.35)
            al_rz = 0.5 * (1 - (t - 30.95) / 0.6)
        else:
            uL_rx = -0.75 + 0.12 * A.ease_out((t - 31.0) / 0.35)
            al_rz = 0.5
        uR_rx = -1.35
        ar_rz = -0.2
        lean_x = 0.14 * (1.0 if t < 31.1 else 0.3)
    elif t < 33.60:             # sass at scrap
        mode = 'sass'
        uL_rx = -0.9
        al_rz = 0.5
        uR_rx = 0.5 * np.sin(t * 6.0)
    elif t < 34.90:             # hop down + settle
        mode = 'descend'
        uR_rx = 0.5
        uL_rx = 0.5
    elif t < 36.55:             # gather glint, lift
        mode = 'lift'
        uR_rx = -1.35 + (-1.75 + 1.35) * A.ease_in_out(np.clip((t - 36.2) / 0.45, 0, 1))
    elif t < 40.60:             # farewell, glint rising away
        mode = 'farewell'
        uR_rx = -2.75
        pip['head'].rot[0] = -0.32
    elif t < 42.30:             # salute
        mode = 'salute'
        uR_rx = -2.85
        ar_rz = -0.45
        pip['head'].rot[0] = -0.2
    elif t < 44.30:             # alone
        mode = 'alone'
        uR_rx = 0.15
        uL_rx = 0.15
        pip['head'].rot[0] = 0.16
    elif t < 46.40:             # head-shake at new item
        mode = 'shake'
        uR_rx = 0.5
        uL_rx = 0.5
        shake = np.sin((t - 44.3) * 9.5) * np.exp(-(t - 44.3) * 1.1)
        pip['head'].rot[1] = shake * 0.30
        pip['head'].rot[0] = 0.05
    else:                       # frozen twitch
        mode = 'twitch'
        uR_rx = 0.5
        uL_rx = 0.5

    # arms
    aL.rot = np.array([al_rx, 0.0, al_rz + idle_b])
    aR.rot = np.array([ar_rx, 0.0, ar_rz - idle_b])
    uL.rot = np.array([uL_rx, 0.0, 0.0])
    uR.rot = np.array([uR_rx, 0.0, 0.0])

    # head + look bounce
    faces.head_bounce(pip, t, env, base=(0, 0.66, 0))

    # eyes
    gtarget = gaze_now(st, t)
    gx, gy = gaze_of(st, t, gtarget)
    open_y = faces.blink_val(t, blinks)
    faces.apply_eyes(pip, t, gaze=(gx, gy), mood=eye_mood, blink_open=open_y,
                     pupil_zoom=pz)
    faces.apply_brows(pip, mood=brow_mood)
    faces.apply_mouth(pip, openness=np.clip(env, 0, 1), smile=smile, width=1.0)

    st['_pip_mode'] = mode
    return squash


# keep _T imported-name free of PIP for ticket raise
def pip_pos_track(st, t):
    """PIP root position/facing beyond 18.0s (hide, run, hops, farewell)."""
    pos = np.array([0.55, 0.0, 1.1])
    ry = 0.10
    squash = (1.0, 1.0, 1.0)
    if t < 12.90:
        if t >= 10.40:
            pos = np.array([0.55, 0.0, 1.1])
            ry = -0.35
    elif t < 13.60:
        f = A.back_out((t - 12.90) / 0.35, s=2.2) if t < 13.25 else 1.0
        pos = np.array([0.55 - 0.34 * f, 0.0, 1.1 + 0.42 * f])
        ry = -0.35
        if t < 13.12:
            k = 1.0 - abs(f - 1.0)
            squash = (1.0 + 0.08 * k, 1.0 - 0.10 * k, 1.0 + 0.08 * k)
    elif t < 23.70:
        if t < 21.55:
            pos = np.array([0.55, 0.0, 1.1])
            ry = 0.10
        elif t < 23.05:
            pos, squash = jump(t, 22.80, 23.05, 0.55, 0.0, 1.1, 2.0, 0.0, 1.72, 0.34)
            ry = -0.35 + (1.45 + 0.35) * A.ease_in(np.clip((t - 22.45) / 0.35, 0, 1))
            ry = ((ry + np.pi) % (2 * np.pi)) - np.pi
        else:
            pos = np.array([2.0, 0.0, 1.72])
            ry = -0.15
    elif t < 26.90:
        if t < 26.00:
            pos = np.array([2.0, 0.0, 1.72])
            ry = -0.15
        else:
            f = A.ease_in_out((t - 26.0) / 0.9)
            pos = np.array([2.0 - 1.7 * f, 0.0, 1.72 - 1.12 * f])
            ry = -0.15 + (-1.9 + 0.15) * f
    elif t < 26.98:
        pos = np.array([0.3, BELT_TOP, 0.6])
        ry = -1.9
    elif t < 27.55:
        f = IO((t - 26.98) / 0.57)
        pos = np.array([0.15, BELT_TOP, 0.62 - 0.5 * f])
        ry = -1.9 + 0.7 * f
    elif t < 28.35:
        pos, squash = jump(t, 27.55, 28.35, 0.15, BELT_TOP, 0.12,
                           SHELF_X, SHELF_TOPS[0], SHELF_Z, 0.75)
        ry = -1.2
    elif t < 28.85:
        pos = np.array([SHELF_X, SHELF_TOPS[0], SHELF_Z])
        ry = -1.2
    elif t < 29.55:
        pos, squash = jump(t, 28.85, 29.55, SHELF_X, SHELF_TOPS[0], SHELF_Z,
                           SHELF_X, SHELF_TOPS[1], SHELF_Z - 0.05, 0.34)
        ry = -1.2
    elif t < 34.30:
        pos = np.array([SHELF_X, SHELF_TOPS[1], SHELF_Z - 0.05])
        f = (2.8) * IO(np.clip((t - 32.6) / 0.7, 0, 1))
        ry = (((-1.2 + f) + np.pi) % (2 * np.pi)) - np.pi
    elif t < 35.15:
        pos, squash = jump(t, 34.30, 35.10, SHELF_X, SHELF_TOPS[1], SHELF_Z - 0.05,
                           0.0, 0.0, 0.6, 0.30)
        f = (-1.75) * IO(np.clip((t - 34.3) / 0.35, 0, 1))
        ry = (((1.6 + f) + np.pi) % (2 * np.pi)) - np.pi
    else:
        pos = np.array([0.0, 0.0, 0.6])
        ry = -0.15
    return pos, ry, squash


# ================================================================ glint
def update_glint(st, t):
    gl = st['glint']
    carry_f = IO(np.clip((t - 9.20) / 0.35, 0, 1))
    if t < 9.20:
        gpos = GLINT_START.copy()
    elif t < 26.90:
        hw = st['pip']['handR'].world[:3, 3]
        if t < 9.60:
            f = A.ease_out((t - 9.20) / 0.35)
            belt = GLINT_START + np.array([0, 0.30 * f, 0])
            gpos = belt + (hw + np.array([0, 0.01, 0]) - belt) * f
        else:
            gpos = hw + np.array([0, 0.01, 0])
    elif t < 36.25:
        hw = st['pip']['handR'].world[:3, 3]
        gpos = hw + np.array([0, 0.01, 0])
    else:
        hw = st['pip']['handR'].world[:3, 3]
        if t < 36.60:
            f = A.ease_in_out((t - 36.25) / 0.35)
            p0 = hw + np.array([0, 0.01, 0])
            p1 = np.array([0.0, 1.35, 0.6])
            gpos = p0 + (p1 - p0) * f
        elif t < 39.85:
            f = A.ease_in_out((t - 36.60) / (39.85 - 36.60))
            gpos = np.array([0.0, 1.35, 0.6]) + np.array([0.0, 2.23, 0.0]) * f
        else:
            f = np.clip((t - 39.85) / 2.4, 0, 1)
            gpos = np.array([0.0, 3.58 + 2.2 * f, 0.6])
    st['glint_pos'] = np.array(gpos, np.float64)

    gp = glint_pulse_v(t)
    boost = 1.0
    if 13.0 <= t <= 15.7:
        boost = 1.0 + 1.35 * A.smooth(np.clip((t - 13.0) / 0.5, 0, 1))
    if t > 40.0:
        boost = 1.0 + 0.8 * A.smooth(np.clip((t - 40.4) / 0.35, 0, 1))
    if 18.0 < t < 34.0:
        boost = 0.85
    core_m, halo_m, shell_m = gl['core'].mat, gl['halo'].mat, gl['shell'].mat
    core_m['emis'] = np.array([30.0, 17.0, 6.0]) * (0.35 + 0.42 * gp) * boost
    shell_m['emis'] = np.array([2.2, 1.1, 0.4]) * min(gp, 1.4) * boost
    halo_m['emis'] = np.array([6.0, 2.6, 0.6]) * (0.25 + 0.42 * gp) * boost
    halo_s = 0.145 * (1.0 + 0.10 * gp) * (1.0 + 0.25 * (boost - 1.0))
    gl['halo'].local = trs4((0, 0, 0), (0, 0, 0), (halo_s,) * 3)
    core_s = 0.085 * (1.0 + 0.05 * gp)
    gl['core'].local = trs4((0, 0, 0), (0, 0, 0), (core_s,) * 3)
    if t < 9.20:
        st['glint_pos'] = GLINT_START + np.array([0.0, 0.012 * np.sin(t * 2.4), 0.0])
    if 13.0 <= t <= 15.8:
        st['glint_pos'] = st['glint_pos'] + np.array([0.0, 0.10 * np.sin((t - 13.0) * 3.4), 0.0])
    gl['root'].pos = st['glint_pos']


def update_scrap(st, t):
    R = st['scrap']['root']
    B = st['scrap']['body']
    eye_m = st['scrap']['eye'].mat
    beam_m = st['scrap']['beam'].mat
    beam = st['scrap']['beam']

    beam_on = 0.0
    if t < 18.30:
        R.pos = np.array([1.8, 3.9, -2.5])
        R.rot = np.array([0.0, 0.7, 0.0])
        eye_m['emis'] = np.array([7.0, 0.6, 0.6]) * 0.25
        beam_m['emis'] = np.array([0.0, 0.0, 0.0])
        st['scrap_pos'] = R.pos.copy()
        return
    elif t < 19.30:
        f = A.ease_in_out((t - 18.30) / 1.0)
        R.pos = np.array([0.75 + 0.15 * f, 3.0 - 1.45 * f, -0.4 - 0.3 * f])
        R.rot = np.array([0.0, 0.7 - 0.7 * f, 0.0])
        B.rot = np.array([0.0, 0.0, 0.12 * np.sin((t - 18.3) * 9.0)])
        beam_on = 0.6
    elif t < 26.00:
        base_p = np.array([0.9, 1.55, -0.7])
        if t < 21.55:
            R.pos = base_p + np.array([0.0, 0.05 * np.sin(t * 2.3), 0.05 * np.sin(t * 1.7)])
            beam_on = 0.5 + 0.3 * np.sin((t - 19.4) * 3.2)
            sway = 0.35 * np.sin((t - 19.5) * 2.6)
            R.rot = np.array([0.0, sway, 0.0])
            B.rot = np.array([0.0, 0.0, 0.15 * np.sin(t * 2.9)])
        elif t < 23.70:
            f = IO(np.clip((t - 21.55) / 0.5, 0, 1))
            R.pos = base_p + np.array([0.0, 0.12 * f, 0.0])
            R.rot = np.array([0.0, 0.0, 0.0])
            beam_on = 1.4
            B.rot = np.array([0.0, 0.0, 0.2 * np.sin(t * 7.0)])
        else:
            f = IO(np.clip((t - 23.70) / 2.2, 0, 1))
            R.pos = np.array([0.9, 1.55, -0.7]) + np.array([0.55, -0.2, 1.3]) * f \
                + np.array([0.0, 0.04 * np.sin(t * 2.6), 0.0])
            tgt = st['pip_pos']
            dyaw = np.arctan2(tgt[0] - R.pos[0], tgt[2] - R.pos[2])
            R.rot = np.array([0.0, np.clip(dyaw, -0.9, 1.1), 0.0])
            beam_on = 1.5 + 0.3 * np.sin(t * 5.0)
            B.rot = np.array([0.0, 0.0, 0.2 * np.sin(t * 7.0)])
    elif t < 26.90:
        f = A.ease_in((t - 26.0) / 0.9)
        R.pos = np.array([1.45, 1.35, 0.6]) + np.array([-0.9, 0.15, -1.4]) * f
        R.rot = np.array([0.0, 0.5 * np.sin(t * 3.0), 0.0])
        beam_on = 1.6
        B.rot = np.array([0.1, 0.0, 0.3 * np.sin(t * 8.0)])
    elif t < 31.90:
        f = IO(np.clip((t - 26.9) / 1.2, 0, 1))
        R.pos = np.array([0.55, 1.5, -0.8]) + np.array([0.9, -0.7, 1.4]) * f \
            + np.array([0.0, 0.05 * np.sin(t * 2.6), 0.0])
        tgt = st['pip_pos']
        R.rot = np.array([0.0, np.arctan2(tgt[0] - R.pos[0], tgt[2] - R.pos[2]), 0.0])
        beam_on = 1.3
        B.rot = np.array([0.0, 0.0, 0.15 * np.sin(t * 6.0)])
    elif t < 33.10:
        wob = (t - 31.9) * 6.0
        R.pos = np.array([1.45, 1.35, 0.6]) + np.array([0.1 * np.sin(wob * 2.0),
                                                        0.15, 0.12 * np.cos(wob * 1.6)])
        R.rot = np.array([0.25 * np.sin(wob * 1.3), 0.8 * np.sin(wob * 0.7), 0.35 * np.sin(wob * 1.1)])
        beam_on = 0.5 * (0.5 + 0.5 * np.sin(wob * 4.0))
        B.rot = np.array([0.0, 0.0, 0.15 * np.sin(t * 6.0)])
        if t > 32.55:
            R.pos[1] += 0.02 * np.sin((t - 32.55) * 30.0)
    else:
        f = np.clip((t - 33.10) / 0.85, 0, 1)
        fe = A.ease_in(f)
        spf = 2.0 * np.pi * f
        cx = 0.6; czH = 0.0
        R.pos = np.array([cx + 0.42 * (1 - f) * np.sin(spf) - 0.45 * fe,
                          1.5 + 2.1 * fe,
                          czH + 0.42 * (1 - f) * np.cos(spf) - 0.7 * fe])
        R.rot = np.array([0.0, 4.5 * f, 2.2 * f])
        beam_on = 0.4 * (1 - f)
        if f >= 1.0:
            R.pos = np.array([1.2, 4.0, -0.7])
            beam_on = 0.0
    eye_m['emis'] = np.array([7.0, 0.6, 0.6]) * (0.6 + 0.7 * min(beam_on, 1.5))
    beam_k = min(beam_on, 1.7)
    beam_m['emis'] = np.array([1.35, 1.1, 0.6]) * beam_k if beam_on > 0.02 else np.array([0.0, 0.0, 0.0])
    bs = min(beam_on, 1.7)
    beam.local = trs4((0, -0.28, 0.34), (np.pi * 0.62, 0, 0), (0.05, 1.7 * max(0.05, bs / 1.5), 0.05))
    st['scrap_pos'] = R.pos.copy()


# ================================================================ world bits
def update_hatch_panels(st, t):
    hl, hr = st['hatchL'], st['hatchR']
    h0 = st['hatch0']
    f = A.ease_in_out(np.clip((t - 18.02) / 0.55, 0, 1))
    slide = 2.45 * f
    hl.pos = h0 + np.array([-0.16 - slide, 0, 0])
    hr.pos = h0 + np.array([0.16 + slide, 0, 0])


def update_stars(st, t):
    scene = st['scene']
    amp = 0.0
    if 13.4 <= t:
        amp = 0.85 * A.smooth(np.clip((t - 13.4) / 1.4, 0, 1))
        if t > 17.8:
            amp = 0.85 * max(0.0, 1.0 - (t - 17.8) / 1.2)
    sky_amp = 0.0
    if t > 35.9:
        sky_amp = IO(np.clip((t - 35.9) / 1.0, 0, 1))
    if 40.4 <= t <= 41.1:
        sky_amp = 1.0 + 1.0 * A.ease_out((t - 40.4) / 0.7)
    scene.star_amp = max(amp, sky_amp)
    scene.star_time = t
    flash = 0.0
    if 40.40 <= t <= 41.6:
        flash = np.exp(-((t - 40.42) / 0.35) ** 2) * 0.55
    scene.sky_flash = flash
    m = st['m_star']
    m['emis'] = np.array([0.65, 0.8, 1.1]) * (2.2 * amp) * (0.8 + 0.4 * np.sin(t * 1.3))
    gb = st['m_gbeam']
    beam_amp = amp * (1.0 if 13.4 <= t <= 17.6 else 0.0)
    gb['emis'] = np.array([0.5, 0.8, 1.4]) * beam_amp * 0.25
    gbeam = st['gbeam']
    if beam_amp > 0.004:
        gbeam.pos = st['glint_pos'] + np.array([0, 0.9, 0])
        gbeam.rot = np.array([0, 0.2 * np.sin(t * 0.7), 0])
        gs = 1.0 + 0.08 * np.sin(t * 2.4)
        gbeam.scl = np.array([0.55 * gs, 2.2, 0.55 * gs])
    else:
        gbeam.pos = np.array([0.0, -50.0, 0.0])
        gbeam.scl = np.array([0.001, 0.001, 0.001])


def update_dust(st, t):
    d = st['dust']
    d.rot = np.array([0.0, t * 0.05, 0.0])
    d.pos = np.array([0.0, 1.7 + 0.08 * np.sin(t * 0.5), -0.4])
    scene = st['scene']
    glow = 0.5 if t < 2.88 else (0.8 if t < 18.0 else (0.55 if t < 34 else 1.0))
    st['m_dust']['emis'] = np.array([0.14, 0.16, 0.2]) * glow


def update_lights(st, t):
    scene = st['scene']
    dark = np.clip((0.4 - t) / 0.4, 0, 1) if t < 2.88 else 0.0
    lam = 1.55
    if t < 2.88:
        lam = 0.30 + 1.25 * A.smooth(np.clip((t - 2.55) / 0.6, 0, 1))
    st['lamp1']['int'] = lam
    st['lamp2']['int'] = lam
    st['fill']['int'] = 0.12 + 0.30 * (lam / 1.55)
    scene.amb_top = np.array([0.10, 0.11, 0.17]) + (np.array([0.30, 0.32, 0.42]) - np.array([0.10, 0.11, 0.17])) \
        * A.smooth(np.clip((t - 2.55) / 0.6, 0, 1)) if t < 3.2 else np.array([0.36, 0.38, 0.48])
    if t < 2.55:
        scene.amb_top = np.array([0.10, 0.11, 0.17])
    scene.amb_bot = scene.amb_top * 0.30

    gp = glint_pulse_v(t)
    boost = 1.0
    if 13.0 <= t <= 15.7:
        boost = 1.0 + 1.35 * A.smooth(np.clip((t - 13.0) / 0.5, 0, 1))
    if t > 40.0:
        boost = 1.0 + 0.8 * A.smooth(np.clip((t - 40.4) / 0.35, 0, 1))
    if 18.0 < t < 34.0:
        boost = 0.8
    glint_off = 1.0 if t < 42.0 else max(0.0, 1.0 - (t - 42.0) / 1.5)
    st['glintLight']['pos'] = list(st['glint_pos'] + np.array([0, 0.10, 0]))
    st['glintLight']['int'] = (0.85 + 0.85 * gp) * boost * glint_off

    sc_i = 0.0
    if 18.6 <= t < 19.4:
        sc_i = 0.8 * (t - 18.6)
    elif 19.4 <= t < 26.0:
        sc_i = 0.8 + 0.9 * np.clip((t - 19.5) / 1.0, 0, 1) + 0.25 * np.sin(t * 5.0)
    elif 26.0 <= t < 31.9:
        sc_i = 1.7 + 0.3 * np.sin(t * 5.0)
    elif 31.9 <= t < 33.3:
        sc_i = (1.2 + 0.9 * np.sin((t - 31.9) * 22.0)) * max(0.0, 1.0 - (t - 32.9) / 0.4)
    elif t >= 33.3:
        sc_i = max(0.0, 0.6 * (1.0 - (t - 33.3) / 0.7))
    st['scrapLight']['pos'] = list(st['scrap_pos'] + np.array([0, 0.0, 0]))
    st['scrapLight']['int'] = sc_i

    sk_i = 0.0
    if t > 35.9:
        sk_i = 1.15 * IO(np.clip((t - 35.9) / 1.0, 0, 1))
    if 40.4 <= t <= 41.5:
        sk_i += 1.4 * np.exp(-((t - 40.45) / 0.35) ** 2)
    st['skyLight']['int'] = sk_i

    # screen face glow
    if st['pip']['scr_box'].mat['emis'] is not None:
        base = np.array([4.0, 9.5, 12.0])
        dim = 0.35 if t < 2.88 else (1.0 if t < 42 else 0.6)
        st['pip']['scr_box'].mat['emis'] = base * dim


def update_ticket(st, t):
    tk = st['ticket']
    vis = 0.0
    if 29.60 <= t < 31.95:
        vis = A.ease_out(np.clip((t - 29.6) / 0.35, 0, 1))
        if t > 31.75:
            vis = max(0.0, 1.0 - (t - 31.75) / 0.35)
    tk.scl = np.array([vis, vis, vis]) if vis > 0.01 else np.array([0.001, 0.001, 0.001])


def update_newitem(st, t):
    ni = st['newitem']
    if t < 43.30:
        ni.pos = NEWITEM_POS + np.array([0, 0, -2.5])
        st['m_glow_new']['emis'] = np.array([0.0, 0.0, 0.0])
    elif t < 44.60:
        f = A.ease_out(np.clip((t - 43.30) / 1.3, 0, 1))
        ni.pos = NEWITEM_POS + np.array([0, 0, -2.5 + 2.5 * f])
        st['m_glow_new']['emis'] = np.array([0.6, 3.2, 3.0]) * (0.4 + 0.8 * f)
    else:
        ni.pos = NEWITEM_POS + np.array([0, 0.01 * np.sin(t * 2.6), 0])
        st['m_glow_new']['emis'] = np.array([0.6, 3.2, 3.0]) * (1.1 + 0.25 * np.sin(t * 5.0))


# ================================================================ master update
def update(st, t):
    t = float(t)
    scene = st['scene']
    scene.star_time = t
    spin, (pos0, ry0) = pip_track(st, t)
    pos, ry, squash = pip_pos_track(st, t)
    if t < 12.90:
        pos, ry = pos0, ry0

    st['pip']['root'].pos = pos
    lean = 0.0
    if 26.0 <= t < 26.90:
        lean = 0.14
    if 22.45 <= t < 23.1:
        lean = 0.10
    st['pip']['root'].rot = np.array([lean, ry, 0.0])

    B = st['pip']['body']
    B.scl = np.array(squash, np.float64)

    sq2 = pip_face_and_body(st, t)

    # wheel spin: rotate wheel node about z
    st['pip']['wheel'].rot = np.array([0.0, 0.0, -spin])

    scene.refresh()

    update_glint(st, t)
    update_scrap(st, t)
    update_ticket(st, t)
    update_newitem(st, t)
    update_stars(st, t)
    update_lights(st, t)
    update_dust(st, t)
    update_hatch_panels(st, t)
    scene.refresh()
    return st


# ================================================================ camera
def shake_at(t):
    s = 0.0
    for ti in INPACTS:
        dt = t - ti
        if 0.0 < dt < 0.55:
            amp = 0.055 if ti in (31.0, 2.55) else 0.035
            s += amp * np.exp(-dt / 0.09) * np.sin(52.0 * dt)
    return s


CUTS = []


def cut(t0, t1):
    def deco(fn):
        CUTS.append((t0, t1, fn))
        return fn
    return deco


def _pick(t):
    for t0, t1, fn in CUTS:
        if t0 <= t < t1:
            return fn(t)
    return CUTS[-1][2](min(t, CUTS[-1][1] - 1e-3))


def camera(st, t):
    c = _pick(t)
    s = shake_at(t)
    eye = np.array(c['eye'], np.float64)
    tgt = np.array(c['tgt'], np.float64)
    if s:
        eye = eye + np.array([s * 0.7, s, 0.0])
    out = {'eye': eye, 'tgt': tgt, 'fov': c.get('fov', 40.0),
           'dof': c.get('dof'), 'roll': c.get('roll', 0.0)}
    return out


@cut(0.0, 3.2)
def c1(t):
    g = GLINT_START
    f1 = A.ease_in_out(min(t / 2.4, 1.0))
    eye = [0.42 + 0.01 * np.sin(t * 0.8), 0.66 - 0.02 * f1, 1.85 - 0.13 * f1]
    tgt = [g[0], g[1], g[2]]
    if t < 2.60:
        return {'eye': eye, 'tgt': tgt, 'fov': 30.0, 'dof': (1.45, 0.012)}
    f = A.ease_out(np.clip((t - 2.60) / 0.60, 0, 1))
    e0 = np.array(eye)
    e1 = np.array([1.75, 1.15, 2.70])
    t0 = np.array(tgt)
    t1 = np.array([0.60, 0.60, 0.90])
    return {'eye': list(e0 + (e1 - e0) * f),
            'tgt': list(t0 + (t1 - t0) * f),
            'fov': 30.0 + 14.0 * f}


@cut(3.2, 5.6)
def c2a(t):
    f = A.ease_in_out(min((t - 3.2) / 2.4, 1.0))
    return {'eye': [2.05 - 0.10 * f, 1.00, 2.85 - 0.10 * f],
            'tgt': [1.35, 0.55, 1.60], 'fov': 42.0}


@cut(5.6, 7.5)
def c2b(t):
    f = A.ease_in_out(min((t - 5.6) / 1.9, 1.0))
    return {'eye': [2.00 - 0.08 * f, 1.05, 2.30 - 0.08 * f],
            'tgt': [1.20, 0.70, 1.25], 'fov': 40.0}


@cut(7.5, 10.0)
def c2c(t):
    f = A.ease_in_out(min((t - 7.5) / 2.5, 1.0))
    return {'eye': [1.90 - 0.15 * f, 0.80 + 0.08 * f, 2.50 - 0.20 * f],
            'tgt': [0.85, 0.60, 1.10], 'fov': 42.0}


@cut(10.0, 12.9)
def c3a(t):
    f = A.ease_in_out((t - 10.0) / 2.9)
    return {'eye': [-0.15 - 0.08 * f, 0.90, 2.35 - 0.12 * f],
            'tgt': 'GLINT', 'fov': 38.0, 'dof': (1.25, 0.014)}


@cut(12.9, 13.5)
def c3b(t):
    return {'eye': [-0.23, 0.92, 2.23], 'tgt': 'GLINT', 'fov': 38.0, 'dof': (1.25, 0.014)}


@cut(13.5, 16.0)
def c3c(t):
    f = A.ease_in_out(np.clip((t - 13.5) / 2.5, 0, 1))
    f2 = A.ease_in(np.clip((t - 14.6) / 1.4, 0, 1))
    eye = [0.10 * f, 0.85 + 0.30 * f, 2.50 + 0.10 * f]
    t0 = np.array([0.60, 0.95, 0.90])
    t1 = np.array([0.30, 2.70, 0.0])
    return {'eye': eye, 'tgt': list(t0 + (t1 - t0) * f2), 'fov': 38.0 + 4.0 * f2}


@cut(16.0, 18.0)
def c3d(t):
    f = A.ease_in_out(min((t - 16.0) / 2.0, 1.0))
    return {'eye': [0.10 + 0.10 * f, 1.15 + 0.03 * np.sin(t * 0.7), 2.60],
            'tgt': [0.20, 2.80 - 0.1 * f, -0.10], 'fov': 42.0}


@cut(18.0, 19.4)
def c4a(t):
    return {'eye': [-0.55, 0.55, 1.90],
            'tgt': [0.50, 2.60 - 1.05 * np.clip((t - 18.4) / 0.9, 0, 1), -0.50],
            'fov': 47.0}


@cut(19.4, 21.6)
def c4b(t):
    return {'eye': [0.95, 0.95, 2.00], 'tgt': 'SCRAP', 'fov': 36.0}


@cut(21.6, 23.6)
def c4c(t):
    return {'eye': [1.30, 1.10, 2.55], 'tgt': [1.30, 0.70, 1.30], 'fov': 46.0}


@cut(23.6, 26.0)
def c4d(t):
    f = A.ease_in(np.clip((t - 23.6) / 2.4, 0, 1))
    return {'eye': [1.30 + 0.15 * f, 1.50 + 0.05 * np.sin(t * 2.6), 0.25 - 0.05 * f],
            'tgt': [1.7 + 0.25 * f, 0.70, 1.72], 'fov': 44.0 - 6.0 * f}


@cut(26.0, 27.55)
def c5a(t):
    return {'eye': [1.60, 0.80, 2.60], 'tgt': 'PIP', 'fov': 44.0}


@cut(27.55, 29.70)
def c5b(t):
    f = A.ease_in_out(np.clip((t - 27.55) / 2.15, 0, 1))
    return {'eye': [0.75 - 0.20 * f, 1.45 + 0.15 * f, 2.35],
            'tgt': [-1.40 - 0.3 * f, 1.50 + 0.3 * f, -1.90], 'fov': 40.0}


@cut(29.70, 31.90)
def c5c(t):
    f = A.ease_in_out(np.clip((t - 29.7) / 1.2, 0, 1))
    return {'eye': [-0.55 + 0.06 * f, 1.60, -0.55],
            'tgt': [-1.78, 2.40, -1.90], 'fov': 36.0}


@cut(31.90, 33.20)
def c5e(t):
    return {'eye': [0.85, 1.30, -0.50], 'tgt': 'SCRAP', 'fov': 34.0}


@cut(33.20, 34.0)
def c5f(t):
    return {'eye': [-0.60, 1.60, 0.70], 'tgt': 'SCRAP', 'fov': 44.0}


@cut(34.0, 36.1)
def c6a(t):
    f = A.ease_in_out(np.clip((t - 34.0) / 2.1, 0, 1))
    return {'eye': [1.40 - 0.15 * f, 0.90, 2.70],
            'tgt': [0.10, 1.40 - 0.3 * f, 0.20], 'fov': 46.0}


@cut(36.1, 40.5)
def c6b(t):
    f = A.ease_in_out(np.clip((t - 36.1) / 4.4, 0, 1))
    return {'eye': [0.80 - 0.10 * f, 0.85 + 0.65 * f, 2.00 - 0.20 * f],
            'tgt': 'GLINT', 'fov': 34.0}


@cut(40.5, 42.0)
def c6c(t):
    f = A.ease_in_out(np.clip((t - 40.5) / 1.5, 0, 1))
    return {'eye': [-1.60 + 0.12 * f, 0.80, 2.35 - 0.2 * f],
            'tgt': [0.10, 1.70 + 0.4 * f, 0.50], 'fov': 40.0}


@cut(42.0, 44.55)
def c7a(t):
    f = A.ease_in_out(np.clip((t - 42.0) / 2.5, 0, 1))
    return {'eye': [0.95 - 0.1 * f, 0.75, 1.95],
            'tgt': [0.10, 0.60, 0.60], 'fov': 36.0}


@cut(44.55, 47.35)
def c7b(t):
    f = A.ease_in_out(np.clip((t - 44.55) / 2.8, 0, 1))
    return {'eye': [0.50, 0.70, 1.40 - 0.08 * f],
            'tgt': [0.10, 0.68, 0.60], 'fov': 32.0, 'dof': (0.95, 0.012)}


@cut(47.35, 48.0)
def c7c(t):
    f = A.ease_out(np.clip((t - 47.35) / 0.35, 0, 1))
    return {'eye': [0.46, 0.68, 1.36 - 0.10 * f],
            'tgt': [0.08, 0.66, 0.60], 'fov': 32.0 - 8.0 * f}


def _resolve_dynamic(st, tgt, t):
    if isinstance(tgt, str):
        if tgt == 'GLINT':
            return st['glint_pos']
        if tgt == 'SCRAP':
            return st['scrap_pos']
        if tgt == 'PIP':
            return st['pip_pos'] + np.array([0, 0.6, 0])
    return tgt


def camera_camera(st, t):
    c = _pick(t)
    tgt = _resolve_dynamic(st, c['tgt'], t)
    return camera(st, t) | {'tgt': np.array(tgt, np.float64)}


def get_shots():
    return [
        {'name': 's1_hook',    't0': 0.0,  't1': 3.2},
        {'name': 's2_rounds',  't0': 3.2,  't1': 10.0},
        {'name': 's3_notegg',  't0': 10.0, 't1': 18.0},
        {'name': 's4_scrap',   't0': 18.0, 't1': 26.0},
        {'name': 's5_chase',   't0': 26.0, 't1': 34.0},
        {'name': 's6_sky',     't0': 34.0, 't1': 42.0},
        {'name': 's7_loop',    't0': 42.0, 't1': 48.0},
    ]



def _resolve_dynamic(st, tgt, t):
    if isinstance(tgt, str):
        if tgt == 'GLINT':
            return st['glint_pos']
        if tgt == 'SCRAP':
            return st['scrap_pos']
        if tgt == 'PIP':
            return st['pip_pos'] + np.array([0, 0.6, 0])
    return tgt


def camera(st, t):
    c = _pick(t)
    tgt = np.array(_resolve_dynamic(st, c['tgt'], t), np.float64)
    s = shake_at(t)
    eye = np.array(c['eye'], np.float64)
    if s:
        eye = eye + np.array([s * 0.7, s, 0.0])
    return {'eye': eye, 'tgt': tgt, 'fov': c.get('fov', 40.0),
            'dof': c.get('dof'), 'roll': c.get('roll', 0.0)}
