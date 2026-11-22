"""Original score for Glimmer Station. 100% synthesized, no external assets.
Composition timeline matches picture: 7 sections over ~48s."""
import numpy as np

SR = 48000
DUR = 48.0
_F = {  # note freqs
}


def f_of(midi):
    return 440.0 * 2**((midi-69)/12)


def env_ar(n, a, r, sr):
    e = np.ones(n)
    an = max(1, int(a*sr)); rn = max(1, int(r*sr))
    an = min(an, n); rn = min(rn, n)
    e[:an] = np.linspace(0, 1, an)**2
    e[n-rn:] *= np.linspace(1, 0, rn)**1.5
    return e


def osc(kind, freq, n, sr, detune_cents=0.0, vib_hz=0.0, vib_amt=0.0):
    t = np.arange(n)/sr
    f = freq * 2**(detune_cents/1200)
    ph = 2*np.pi*f*t
    if vib_hz > 0:
        ph += vib_amt*np.sin(2*np.pi*vib_hz*t)
    if kind == 'sine':
        return np.sin(ph)
    if kind == 'saw':
        return 2*((f*t + vib_amt/(2*np.pi)*np.sin(2*np.pi*vib_hz*t)) % 1.0) - 1.0
    if kind == 'square':
        return np.sign(np.sin(ph))
    if kind == 'tri':
        return 2/np.pi*np.arcsin(np.sin(ph))


def pluck(freq, dur, amp=0.5, kind='karplus', seed=0):
    n = int(dur*SR)
    if kind == 'bell':
        t = np.arange(n)/SR
        x = (np.sin(2*np.pi*freq*t) + 0.4*np.sin(2*np.pi*freq*2.0*t + 0.4)
             + 0.18*np.sin(2*np.pi*freq*4.01*t))
        x *= np.exp(-t*4.2)
    elif kind == 'celesta':
        t = np.arange(n)/SR
        x = (np.sin(2*np.pi*freq*t) + 0.3*np.sin(2*np.pi*freq*4*t) + 0.12*np.sin(2*np.pi*freq*7.06*t))
        x *= np.exp(-t*2.8)
    elif kind == 'pizz':
        rng = np.random.default_rng(seed)
        t = np.arange(n)/SR
        # filtered noise burst + sine body
        noise = rng.standard_normal(n)
        from scipy.signal import butter, lfilter
        b, a = butter(2, max(freq*2.5, 400)/(SR/2), 'low')
        noise = lfilter(b, a, noise)
        x = (np.sin(2*np.pi*freq*t)*0.9 + noise*0.5*np.exp(-t*30)) * np.exp(-t*7.0)
    else:
        raise ValueError(kind)
    e = env_ar(n, 0.004, min(0.09, dur*0.35), SR)
    return x*e*amp


def pad_chord(freqs, dur, amp=0.30, detune=7.0, bright=0.5):
    n = int(dur*SR)
    x = np.zeros(n)
    for f in freqs:
        for dt in (-detune, detune):
            x += osc('saw', f, n, SR, detune_cents=dt, vib_hz=0.15, vib_amt=0.02)*0.5
            x += osc('sine', f*0.5, n, SR)*0.2
    # soften
    from scipy.signal import butter, lfilter
    b, a = butter(2, max(600, 1200*bright)/(SR/2), 'low')
    x = lfilter(b, a, x)
    e = env_ar(n, min(0.8, dur*0.35), min(1.2, dur*0.5), SR)
    return x*e*amp/ (1+len(freqs))


def bass(freq, dur, amp=0.5):
    n = int(dur*SR); t = np.arange(n)/SR
    x = np.sin(2*np.pi*freq*t) + 0.3*np.sin(2*np.pi*freq*2*t) + 0.08*np.sin(2*np.pi*freq*3*t)
    x *= env_ar(n, 0.005, min(0.10, dur*0.3), SR)
    return x*amp


def drone(freq, dur, amp=0.4, trem=0.25):
    n = int(dur*SR); t = np.arange(n)/SR
    x = (np.sin(2*np.pi*freq*t) + 0.5*np.sin(2*np.pi*freq*1.5*t) + 0.3*np.sin(2*np.pi*freq*0.5*t))
    x *= (1 + trem*np.sin(2*np.pi*0.24*t))*0.5+0.5
    e = env_ar(n, 1.2, 1.5, SR)
    return x*e*amp


def kick(dur=0.3, amp=0.9):
    n = int(dur*SR); t = np.arange(n)/SR
    f = 120*np.exp(-t*22)+38
    x = np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*16)
    return x*amp


def snare(dur=0.16, amp=0.5, seed=1):
    n = int(dur*SR); t = np.arange(n)/SR
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)*np.exp(-t*30) + 0.4*np.sin(2*np.pi*190*t)*np.exp(-t*24)
    return x*amp


def hat(dur=0.05, amp=0.2, seed=2):
    n = int(dur*SR); t = np.arange(n)/SR
    rng = np.random.default_rng(seed)
    from scipy.signal import butter, lfilter
    x = rng.standard_normal(n)
    b, a = butter(2, 6500/(SR/2), 'high')
    return lfilter(b, a, x)*np.exp(-t*60)*amp


def tick(dur=0.06, amp=0.35):
    n = int(dur*SR); t = np.arange(n)/SR
    return np.sin(2*np.pi*2100*t)*np.exp(-t*90)*amp


def alarm(freq=880, dur=0.32, amp=0.5):
    n = int(dur*SR); t = np.arange(n)/SR
    x = np.sign(np.sin(2*np.pi*freq*t))*(0.6+0.4*np.cos(2*np.pi*freq*0.5*t))
    x += 0.3*np.sin(2*np.pi*freq*0.75*t)
    return x*np.exp(-t*4)*amp


def sweep(f0, f1, dur, amp=0.4, kind='sine'):
    n = int(dur*SR); t = np.arange(n)/SR
    ph = 2*np.pi*(f0*t + (f1-f0)*t*t/(2*dur))
    x = np.sin(ph)
    if kind == 'noise':
        rng = np.random.default_rng(5)
        x = x*0.3 + rng.standard_normal(n)*np.exp(-(t-0)/dur*2.2)
    return x*amp*np.exp(-t/dur*1.7)


def add(buf, x, at):
    i = int(at*SR)
    j = min(len(buf), i+len(x))
    if j > i:
        buf[i:j] += x[:j-i]


# ---------------------------------------------------------------- score
def build_score():
    total = int((DUR+2.0)*SR)
    mixL = np.zeros(total); mixR = np.zeros(total)
    rng = np.random.default_rng(42)

    def addS(x, at, width=0.6):
        add(mixL, x*(1-width*0.5), at)
        add(mixR, x*(1+width*0.5), at)

    # ---- S1 0-3s: dark mystery pulse (Am-ish, Dronish Dm)
    for i, tt in enumerate(np.arange(0.0, 3.0, 0.5)):
        addS(tick(amp=0.22 if tt < 2.5 else 0.3), tt)
    addS(drone(f_of(50), 3.4, 0.34), 0.0)
    addS(drone(f_of(57)*0.5, 3.4, 0.22), 0.0)
    addS(pluck(f_of(86), 2.0, 0.14, 'bell'), 0.6)     # heartbeat sparkle
    addS(pluck(f_of(79), 2.0, 0.11, 'bell'), 1.1)

    # ---- S2 3-10s: playful pizzicato C major (bpm 96 -> beat 0.625)
    bpm = 96; b = 60/bpm
    prog = [(3, [57, 60, 64]), (8, [59, 62, 67]), (5, [57, 60, 64]), (10, [53, 57, 60])]
    # bars: C  G/B  Am  F each 1.75s (4 beats)
    chord_t = 3.0
    chords = [([48, 55], [60, 64, 67]), ([47, 55], [59, 62, 67]),
              ([45, 52], [57, 60, 64]), ([41, 48], [53, 57, 60])]
    bar = 2.5   # 4 beats at bpm 96
    for ci, (bassnotes, trinotes) in enumerate(chords):
        t0 = 3.0 + ci*bar
        if t0 >= 10.0: break
        for k in range(4):
            tb = t0 + k*b
            if tb >= 10.0: break
            if k % 2 == 0:
                addS(bass(f_of(bassnotes[k//2 % 2]), 0.5, 0.5), tb)
            addS(pluck(f_of(trinotes[k % 3] + 12), 0.4, 0.16, 'pizz', seed=ci*7+k), tb + 0.03)
            if k == 2:
                addS(pluck(f_of(trinotes[1]+12), 0.35, 0.11, 'pizz', seed=ci+30), tb + b/2)
        addS(pad_chord([f_of(x) for x in trinotes], bar*0.9, 0.10, bright=0.3), t0+0.1)

    # ---- S3 10-18s: wonder, celesta arps over F Lydian-ish pad
    pad_ch = [[65, 69, 72, 76], [64, 67, 71, 74], [69, 72, 76, 79], [67, 71, 74, 77]]
    for ci, ch in enumerate(pad_ch):
        addS(pad_chord([f_of(x) for x in ch], 2.2, 0.13, bright=0.55), 10.0 + ci*2.0)
    arp_notes = [81, 84, 88, 84, 86, 81, 76, 81]
    for k in range(26):
        tb = 10.2 + k*0.3
        if tb > 17.8: break
        f = f_of(arp_notes[k % len(arp_notes)] + (12 if k % 4 == 0 else 0))
        addS(pluck(f, 1.2, 0.11 + 0.03*(k % 2), 'celesta'), tb, width=0.5)

    # ---- S4 18-26s: tension. low tritone drone + ticking + rising pulse + alarm
    addS(drone(55.0, 8.4, 0.34, trem=0.4), 17.8)
    addS(drone(58.3, 8.4, 0.22, trem=0.5), 18.0)
    for k in range(32):
        addS(tick(amp=0.3), 18.0 + k*0.25)
    for k in range(8):
        f0 = 200 + k*40
        addS(sweep(f0, f0*1.3, 0.5, 0.16), 18.5 + k*0.9)
    # alarm stabs at 21.5 (SCRAP entry), 24.5
    for ta in (21.6, 22.0, 24.4, 24.8):
        addS(alarm(784, 0.34, 0.34), ta)
        addS(alarm(622, 0.34, 0.22), ta+0.08)

    # ---- S5 26-34s: determined march, D minor, bpm 128
    bpm = 128; b = 60/bpm
    dmin = [50]  # D
    addS(pad_chord([f_of(x) for x in (50, 53, 57)], 7.4, 0.14, bright=0.45), 26.0)
    riff = [50, 50, 53, 50, 57, 55, 53, 50]
    for k in range(24):
        tb = 26.0 + k*0.46875
        if tb > 33.4: break
        addS(bass(f_of(riff[k % 8] - 12), 0.42, 0.55), tb)
        if k % 2 == 1:
            addS(snare(0.16, 0.34), tb)
        if k % 4 == 0:
            addS(kick(0.3, 0.8), tb)
        if k % 2 == 0:
            addS(hat(0.05, 0.16), tb + 0.234)
    for k in range(4):
        addS(pluck(f_of(74 + (k % 2)*3), 0.6, 0.2, 'bell'), 27.0 + k*0.94)

    # ---- S6 34-42s: emotional swell Gm Eb Bb F
    prog = [([43, 46, 50], [62, 65, 67, 70]), ([39, 43, 46], [62, 65, 68, 72]),
            ([46, 50, 58], [62, 65, 70, 74]), ([41, 45, 48], [64, 67, 69, 72])]
    for ci, (low, hi) in enumerate(prog):
        t0 = 34.0 + ci*2.0
        addS(pad_chord([f_of(x) for x in low], 2.4, 0.30, bright=0.42), t0)
        addS(pad_chord([f_of(x) for x in hi], 2.4, 0.16, bright=0.7), t0)
    mel = [(34.4, 74), (35.0, 76), (36.2, 77), (37.2, 74), (38.2, 72), (39.2, 70), (40.2, 74)]
    for tt, mm in mel:
        addS(pluck(f_of(mm), 2.0, 0.22, 'bell'), tt)
    addS(sweep(600, 1200, 2.2, 0.10), 36.0)
    # final shimmer at 40.4 when sky ignites
    for k in range(7):
        addS(pluck(f_of(86 + k), 2.2, 0.14, 'celesta'), 40.4 + k*0.09)

    # ---- S7 42-48: playful button + comedic sting
    for ci, (bassnotes, trinotes) in enumerate(chords[:3]):
        t0 = 42.0 + ci*bar
        for k in range(4):
            tb = t0 + k*b
            if tb >= 47.4: break
            if k % 2 == 0:
                addS(bass(f_of(bassnotes[k//2 % 2]), 0.5, 0.42), tb)
            addS(pluck(f_of(trinotes[k % 3]+12), 0.4, 0.15, 'pizz', seed=ci*9+k), tb)
            if k == 1:
                addS(pluck(f_of(trinotes[2]+12), 0.35, 0.10, 'pizz', seed=70+ci), tb+b/2)
        addS(pad_chord([f_of(x) for x in trinotes], bar*0.7, 0.09, bright=0.3), t0+0.1)
    # comedic eye-twitch sting 46.6 & final button 47.3
    addS(sweep(900, 500, 0.5, 0.2), 46.6)
    addS(pluck(f_of(79+12), 1.5, 0.2, 'bell'), 47.35)

    # ---- global: gentle reverb
    from scipy.signal import fftconvolve
    rng2 = np.random.default_rng(7)
    ir_len = int(1.1*SR)
    tI = np.arange(ir_len)/SR
    ir = rng2.standard_normal(ir_len)*np.exp(-tI*4.5)*np.exp(-tI*1.5)
    ir[:int(0.012*SR)] = 0
    ir /= np.abs(ir).sum()*40
    wetL = fftconvolve(mixL, ir)[:total]
    wetR = fftconvolve(mixR, ir[:ir_len-1][::-1])[:total]
    out = np.stack([mixL + wetL*0.8, mixR + wetR*0.8], 1)
    return out


def write_score(path='/workspace/studio/output/music.wav'):
    import wave
    st = build_score()
    st = st[:int(DUR*SR)]
    m = np.abs(st).max()
    st = np.clip(st/m*0.82, -1, 1)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((st*32767).astype(np.int16).tobytes())
    print('score written', st.shape)


if __name__ == '__main__':
    write_score()
