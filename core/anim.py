"""Keyframe animation system: channels, easings, tracks."""
import numpy as np


# ---------------------------------------------------------------- easings
def linear(t): return t
def smooth(t): return t*t*(3-2*t)
def smoother(t): return t*t*t*(t*(t*6-15)+10)
def ease_in(t): return t*t
def ease_out(t): return 1-(1-t)**2
def ease_in_out(t): return 3*t*t - 2*t*t*t
def ease_in_cubic(t): return t**3
def ease_out_cubic(t): return 1-(1-t)**3
def ease_out_quart(t): return 1-(1-t)**4
def ease_in_out_quart(t):
    return 2*t*t*t*t if t < 0.5 else 1-((-2*t+2)**4)/2
def back_out(t, s=1.70158):
    t = t-1
    return 1 + t*t*((s+1)*t + s)
def back_in(t, s=1.70158):
    return t*t*((s+1)*t - s)
def back_in_out(t, s=1.70158):
    t0, t1 = 0.5, 0.5
    if t < t0: return back_in(t*2, s)/2
    return 1 + back_out((t-0.5)*2, s)/2
def elastic_out(t, s=4.0):
    if t >= 1: return 1.0
    if t <= 0: return 0.0
    return 1 + np.exp(-s*t)*np.cos(9*t)*(-1)
def bounce_out(t):
    if t >= 1: return 1.0
    n1, d1 = 7.5625, 2.75
    if t < 1/d1: return n1*t*t
    elif t < 2/d1: t -= 1.5/d1; return n1*t*t + 0.75
    elif t < 2.5/d1: t -= 2.25/d1; return n1*t*t + 0.9375
    else: t -= 2.625/d1; return n1*t*t + 0.984375

EASINGS = {'lin':linear,'smooth':smooth,'smoother':smoother,'in':ease_in,'out':ease_out,
           'io':ease_in_out,'ic':ease_in_cubic,'oc':ease_out_cubic,'oq':ease_out_quart,
           'ioq':ease_in_out_quart,'back':back_out,'backin':back_in,'bio':back_in_out,
           'elastic':elastic_out,'bounce':bounce_out}


class Track:
    """Scalar keyframe track: keys=[(time, value, ease_out_of_segment)]"""
    def __init__(self, keys, default=0.0):
        self.keys = sorted(keys, key=lambda k: k[0])
        self.default = default

    def at(self, t):
        ks = self.keys
        if not ks: return self.default
        if t <= ks[0][0]: return ks[0][1]
        if t >= ks[-1][0]: return ks[-1][1]
        for i in range(len(ks)-1):
            t0, v0, e0 = ks[i]
            t1, v1, _ = ks[i+1]
            if t0 <= t <= t1:
                dt = t1 - t0
                f = 0.0 if dt <= 1e-9 else (t-t0)/dt
                e = EASINGS[e0] if isinstance(e0, str) else e0
                return v0 + (v1-v0)*e(f)
        return self.default


def track(t, *seg):
    """Convenience: track(t, (t0,v0,'ease',(t1,v1)) ...) Frames list of (time, value, ease)
    e.g. K(t, (0,0),(0.5,1,'back'), (1,0)) -> sorted; between key i and i+1 uses ease of i-th key's 3rd elem (or 'io')
    """
    ks = []
    for s in seg:
        t_k, v = s[0], s[1]
        e = s[2] if len(s) > 2 else 'io'
        ks.append((t_k, v, e))
    return Track(ks).at(t)


def K(*seg):
    """Build track given duration normalized 0..1 with keys."""
    ks = []
    for s in seg:
        t_k, v = s[0], s[1]
        e = s[2] if len(s) > 2 else 'io'
        ks.append((t_k, v, e))
    return Track(ks)


class Vec3Track:
    def __init__(self, tracks):
        self.t = tracks
    def __call__(self, t):
        return np.array([self.t[0].at(t), self.t[1].at(t), self.t[2].at(t)])


def VT(t0, t1, t2):
    return Vec3Track((t0, t1, t2))


# ---------------------------------------------------------------- helpers
def pulse(t, period, duty=0.5, amp=1.0):
    ph = (t % period)/period
    return amp * (1.0 if ph < duty else 0.0)

def sin01(t, period=1.0, phase=0.0):
    return 0.5+0.5*np.sin(2*np.pi*(t/period + phase))

def osc(t, period, amp, base=0.0, phase=0.0, ease_kind=None):
    return base + amp*np.sin(2*np.pi*(t/period + phase))

def clamp(x, a, b): return float(np.clip(x, a, b))

def lerp(a, b, f): return a + (b-a)*f


def mouth_curve(t, syllables, base=0.0):
    """Simple syllable-based mouth openness envelope: syllables = list of (start, dur, open)."""
    v = base
    for (s, d, o) in syllables:
        if s <= t <= s+d:
            ph = (t-s)/max(d, 1e-6)
            v = o*np.sin(np.pi*ph)**0.8
    return v
