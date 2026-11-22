"""SFX synthesizer + cue sheet. All synthesized from scratch."""
import numpy as np, wave, os

SR = 48000
OUT = '/workspace/studio/output/sfx'


def _wav_write(path, x):
    m = np.abs(x).max() + 1e-9
    x = x/np.maximum(m, 1e-9)*0.85
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((np.clip(x, -1, 1)*32767).astype(np.int16).tobytes())




def mixlen(*sigs):
    n = max(len(s) for s in sigs)
    out = np.zeros(n)
    for s in sigs:
        out[:len(s)] += s
    return out

def rand(seed):
    return np.random.default_rng(seed)


def noise_burst(dur, lpf=3000, decay=18, amp=1.0, seed=0, brightness=1.0):
    from scipy.signal import butter, lfilter
    n = int(dur*SR)
    x = rand(seed).standard_normal(n)
    b, a = butter(2, max(lpf*brightness, 80)/(SR/2), 'low')
    x = lfilter(b, a, x)
    t = np.arange(n)/SR
    x *= np.exp(-t*decay)
    return x*amp


def thud(f0=70, f1=40, dur=0.5, amp=1.0):
    n = int(dur*SR); t = np.arange(n)/SR
    f = f0 + (f1-f0)*t/dur
    x = np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*9)
    return x*amp


def clang(dur=1.0, seed=3, amp=1.0):
    n = int(dur*SR); t = np.arange(n)/SR
    x = np.zeros(n)
    rng = rand(seed)
    for k, det in enumerate((1.0, 1.51, 2.26, 2.98, 3.71, 4.83)):
        x += np.sin(2*np.pi*260*det*t + rng.random())*np.exp(-t*(6+k*2.6))*(1.0-0.1*k)
    x += noise_burst(dur, 6000, 40, 0.4, seed)
    return x*amp


def servo(dur=0.4, f0=300, f1=700, amp=0.5):
    n = int(dur*SR); t = np.arange(n)/SR
    f = f0+(f1-f0)*t/dur
    x = np.sign(np.sin(2*np.pi*np.cumsum(f)/SR))*0.4 + np.sin(2*np.pi*np.cumsum(f*1.02)/SR)*0.6
    e = np.minimum(1, t*30)*np.exp(-t*2.2)
    return x*e*amp


def boop(f0=700, up=True, dur=0.14, amp=0.5):
    n = int(dur*SR); t = np.arange(n)/SR
    f = f0*(1.5 if up else 0.6)*t/dur + f0
    x = np.sin(2*np.pi*np.cumsum(f)/SR)
    e = np.sin(np.pi*t/dur)**0.7
    return x*e*amp


def chime(freqs=(1046, 1318, 1568), dur=1.6, amp=1.0, spread=0.09, seed=5):
    n = int(dur*SR); t = np.arange(n)/SR
    x = np.zeros(n)
    for i, f in enumerate(freqs):
        st = int(i*spread*SR)
        xi = np.sin(2*np.pi*f*t)+0.3*np.sin(2*np.pi*f*4*t)
        xi *= np.exp(-t*3.2)
        x[st:] += xi[:len(x)-st]*1.0/len(freqs)
    return x*amp


def sparkle_rise(dur=1.2, n_ch=9, amp=0.5, seed=9):
    n = int(dur*SR); t = np.arange(n)/SR
    x = np.zeros(n)
    rng = rand(seed)
    for i in range(n_ch):
        st = int(i/(n_ch-1)*dur*0.7*SR)
        f = 700*np.exp(0.16*i) + rng.uniform(-80, 80)
        xi = np.sin(2*np.pi*f*(np.arange(0, n-st))/SR)*np.exp(-np.arange(0, n-st)/SR*5)
        x[st:] += xi*0.7/n_ch
    return x*amp*2.2


def whoosh(dur=0.6, amp=0.5, seed=7, rise=True):
    n = int(dur*SR); t = np.arange(n)/SR
    from scipy.signal import butter, lfilter
    x = rand(seed).standard_normal(n)
    fc = (300+1200*(t/dur)) if rise else (1200-900*(t/dur))
    ph = 2*np.pi*np.cumsum(fc)/SR
    # fake band sweep: modulate noise
    x = x*(0.4+0.6*np.sin(ph/30)**2)
    e = np.sin(np.pi*t/dur)**1.5
    b, a = butter(2, 2600/(SR/2), 'low')
    return lfilter(b, a, x)*e*amp


def hiss_rise(dur=1.4, amp=0.5, seed=13):
    n = int(dur*SR); t = np.arange(n)/SR
    from scipy.signal import butter, lfilter
    x = rand(seed).standard_normal(n)
    b, a = butter(2, 1000/(SR/2), 'high')
    x = lfilter(b, a, x)*np.exp(-t*1.2)
    e = np.clip(t/dur*2.2, 0, 1)**0.5*np.where(t > dur*0.8, np.exp(-(t-dur*0.8)*16), 1)
    return x*e*amp


def drone_scan(dur=1.1, amp=0.5):
    n = int(dur*SR); t = np.arange(n)/SR
    f = 1500*np.exp(-t*1.1) + 60
    x = np.sin(2*np.pi*np.cumsum(f)/SR)*(0.5+0.5*np.sin(2*np.pi*16*t))
    e = np.clip(t*40, 0, 1)*np.exp(-t*2.0)
    return x*e*amp


def error_buzz(dur=0.8, amp=0.5):
    n = int(dur*SR); t = np.arange(n)/SR
    x = np.sign(np.sin(2*np.pi*110*t))*(0.6+0.4*np.sign(np.sin(2*np.pi*55*t)))
    e = (np.sin(2*np.pi*3.5*t)*0.5+0.5)*np.exp(-t*2.0)
    return x*e*amp


def shimmer_swell(dur=2.6, amp=1.0):
    n = int(dur*SR); t = np.arange(n)/SR
    x = np.zeros(n)
    for k, f in enumerate((523, 659, 784, 1046)):
        x += np.sin(2*np.pi*f*t*(1+0.001*k))*np.sin(np.pi*t/dur)**2
    x += chime((1568, 2093), dur, 0.4)[...]
    return x*amp*0.5


def conveyor_jerk(dur=0.55, amp=1.0, seed=17):
    c = clang(0.7, seed, 0.8) + thud(80, 45, 0.5, 0.9)
    n = int(dur*SR); c = c[:n] if len(c) > n else np.pad(c, (0, n-len(c)))
    return c*amp


def blink_tick(amp=0.3):
    return boop(1400, False, 0.05, amp)


def rain_ambience(dur=48.0, amp=0.16, seed=23):
    """steady rain hiss with gentle modulation"""
    from scipy.signal import butter, lfilter
    n = int(dur*SR)
    x = rand(seed).standard_normal(n)
    b, a = butter(2, 1600/(SR/2), 'low')
    x = lfilter(b, a, x)
    t = np.arange(n)/SR
    x *= (0.8+0.2*np.sin(2*np.pi*0.07*t)+0.1*np.sin(2*np.pi*0.013*t))
    # sparse drop ticks
    rng = rand(seed+1)
    for i in range(60):
        st = int(rng.uniform(0, dur-0.2)*SR)
        f = rng.uniform(900, 2000); dd = rng.uniform(0.03, 0.08)
        m = int(dd*SR); tt = np.arange(m)/SR
        x[st:st+m] += np.sin(2*np.pi*f*tt)*np.exp(-tt*70)*0.10
    return x*amp*8


def neon_hum(dur=48.0, amp=0.05, seed=31):
    n = int(dur*SR); t = np.arange(n)/SR
    x = np.sin(2*np.pi*58*t)*0.5 + np.sin(2*np.pi*58*2*t)*0.25 + np.sin(2*np.pi*58*3.01*t)*0.1
    x *= (0.8+0.2*np.sin(2*np.pi*0.24*t+1.0))
    return x*amp*2


def room_tone(dur=48.0, amp=0.1, seed=41):
    from scipy.signal import butter, lfilter
    n = int(dur*SR)
    x = rand(seed).standard_normal(n)
    b, a = butter(2, 260/(SR/2), 'low')
    x = lfilter(b, a, x)
    return x*amp*6


def render_all():
    os.makedirs(OUT, exist_ok=True)
    specs = {
        'clang_conveyor': lambda: mixlen(clang(1.0, 3, 0.9), thud(75, 42, 0.6, 0.9)),
        'boot_up':        lambda: np.concatenate([servo(0.5, 120, 600, 0.6), servo(0.3, 400, 900, 0.5)]),
        'servo1':         lambda: servo(0.35, 250, 620, 0.5),
        'servo2':         lambda: servo(0.35, 620, 280, 0.5),
        'boop':           lambda: boop(760, True, 0.15, 0.55),
        'boop2':          lambda: boop(980, True, 0.12, 0.45),
        'boop_dn':        lambda: boop(800, False, 0.14, 0.5),
        'blink':          lambda: boop(1500, False, 0.045, 0.22),
        'stamp':          lambda: mixlen(thud(120, 60, 0.30, 1.1), noise_burst(0.12, 4200, 60, 0.6, seed=23)),
        'paper':          lambda: noise_burst(0.45, 5000, 12, 0.30, seed=29, brightness=1.2),
        'glint_pulse':    lambda: chime((784, 1175), 1.0, 0.6, spread=0.2),
        'glint_pulse2':   lambda: chime((659, 988), 1.0, 0.5, spread=0.2),
        'sparkle_rise':   lambda: sparkle_rise(1.6, 12, 0.6),
        'sparkle_rise2':  lambda: sparkle_rise(1.2, 10, 0.5, seed=15),
        'whoosh':         lambda: whoosh(0.7, 0.6, rise=True),
        'whoosh_dn':      lambda: whoosh(0.7, 0.6, seed=8, rise=False),
        'hatch_open':     lambda: mixlen(hiss_rise(1.6, 0.7), thud(60, 38, 0.7, 0.8)),
        'hatch_clunk':    lambda: mixlen(clang(0.6, 19, 0.6), thud(90, 50, 0.4, 1.0)),
        'scan':           lambda: drone_scan(1.2, 0.6),
        'alarm':          lambda: np.concatenate([(np.array([0]*int(0.0*SR))), build_alarm_beeps()]),
        'error':          lambda: error_buzz(0.9, 0.6),
        'spin_up':        lambda: servo(1.1, 200, 1400, 0.7),
        'impact_soft':    lambda: mixlen(thud(90, 50, 0.4, 0.8), noise_burst(0.16, 900, 28, 0.4, seed=35)),
        'dome_open':      lambda: mixlen(hiss_rise(2.0, 0.8), chime((523, 784), 2.2, 0.30)),
        'star_rise':      lambda: sparkle_rise(2.4, 14, 0.7, seed=51),
        'sky_ignite':     lambda: shimmer_swell(2.8, 1.1),
        'twitch':         lambda: np.concatenate([boop(2000, False, 0.04, 0.3), np.zeros(int(0.07*SR)), boop(2000, False, 0.04, 0.3)]),
        'fall_bonk':      lambda: mixlen(thud(140, 70, 0.30, 1.0), noise_burst(0.1, 2200, 40, 0.5, seed=39)),
        'beltpack':       lambda: np.concatenate([noise_burst(0.3, 1500, 10, 0.4, seed=45), thud(60, 40, 0.4, 0.6)]),
    }
    for name, fn in specs.items():
        _wav_write(f'{OUT}/{name}.wav', fn())
    print('sfx written:', len(specs))


def build_alarm_beeps():
    n = int(0.11*SR); t = np.arange(n)/SR
    a1 = np.sign(np.sin(2*np.pi*880*t))*0.4*np.sin(np.pi*t/0.11)**0.4
    a2 = np.sign(np.sin(2*np.pi*660*t))*0.4*np.sin(np.pi*t/0.11)**0.4
    return np.concatenate([a1, np.zeros(int(0.07*SR)), a2, np.zeros(int(0.09*SR)),
                           a1, np.zeros(int(0.07*SR)), a2])


def synchronize():
    return np.zeros(int(0.001*SR))


if __name__ == '__main__':
    render_all()
