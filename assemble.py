"""Assemble final MP4: concat shot videos, end card, soundtrack mix (music+SFX+dialogue, ducked)."""
import os, sys, json, wave, math, subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'output')
SR = 48000
FPS = 30
TOTAL = 48.0

sys.path.insert(0, ROOT)
import anim.shots as S
from audio.voice import load_wav  # SR-asserted loader (48k)


def load_any(path):
    with wave.open(path, 'rb') as wf:
        ch = wf.getnchannels()
        assert wf.getframerate() == SR, (path, wf.getframerate())
        x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float64) / 32768
        if ch == 2:
            x = x.reshape(-1, 2).mean(1)
    return x


def build_soundtrack(dst):
    N = int(TOTAL * SR)
    mix = np.zeros((N, 2))
    # ---------- music bed (stereo 48k 48s)
    mpath = os.path.join(OUT, 'music.wav')
    with wave.open(mpath, 'rb') as wf:
        mch = wf.getnchannels()
        mx = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float64) / 32768
        if mch == 2:
            mx = mx.reshape(-1, 2)
        else:
            mx = np.stack([mx, mx], 1)
    n = min(len(mx), N)
    mix[:n] += mx[:n] * 0.72

    # ---------- rain ambience bed
    rng = np.random.default_rng(20)
    w = rng.standard_normal(N)
    from scipy.signal import butter, lfilter
    b, a = butter(2, [900.0 / (SR / 2), 6000.0 / (SR / 2)], 'band')
    rain = lfilter(b, a, w) * 0.05
    slow = np.sin(2 * np.pi * 0.13 * np.arange(N) / SR) * 0.25 + 0.75
    rain *= slow[:, None] if rain.ndim > 1 else slow
    rain = np.stack([rain, np.roll(rain, 480)], 1) if rain.ndim == 1 else rain
    mix += rain * 0.25

    # ---------- SFX cues
    for t0, name, gain in S.SFX_CUES:
        p = os.path.join(S.SFX_DIR, name + '.wav')
        if not os.path.exists(p):
            print('  missing sfx', name); continue
        x = load_any(p)
        i0 = int(t0 * SR)
        i1 = min(i0 + len(x), N)
        if i1 <= i0:
            continue
        seg = x[:i1 - i0]
        fade = np.ones(len(seg))
        nf = min(int(0.012 * SR), len(seg))
        fade[:nf] = np.linspace(0, 1, nf); fade[-nf:] = np.linspace(1, 0, nf)
        seg = seg * gain * fade
        mix[i0:i1, 0] += seg
        mix[i0:i1, 1] += seg

    # ---------- dialogue (with tiny room reverb via comb)
    dlg_env = np.zeros(N)
    dlg_buf = np.zeros((N, 2))
    for t0, ch, wav, gain in S.DIALOGUE:
        p = os.path.join(S.VOICE_DIR, wav + '.wav')
        if not os.path.exists(p):
            print('  missing voice', wav); continue
        x = load_wav(p)
        i0 = int(t0 * SR)
        i1 = min(i0 + len(x), N)
        if i1 <= i0:
            continue
        seg = x[:i1 - i0] * gain
        # small room: feedback comb 31ms
        d = int(0.031 * SR)
        y = seg.copy()
        for rep in (1, 2):
            y[d * rep:] += seg[:max(0, len(seg) - d * rep)] * (0.32 ** rep)
        fade = np.ones(len(y))
        nf = min(int(0.008 * SR), len(y))
        fade[:nf] = np.linspace(0, 1, nf)
        fade[-int(0.05 * SR):] *= np.linspace(1, 0.2, int(0.05 * SR))
        y *= fade
        dlg_buf[i0:i1, 0] += y * 0.25
        dlg_buf[i0:i1, 1] += y * 0.25
        dlg_env[i0:i1] = np.maximum(dlg_env[i0:i1], np.abs(y))

    # ---------- dialogue ducking on music
    hop = SR // 60
    env = dlg_env.reshape(-1, hop).max(1)
    from scipy.ndimage import gaussian_filter1d
    env = gaussian_filter1d(env, 8)
    env = np.clip(env / max(env.max(), 1e-6), 0, 1)
    duck = 1.0 - 0.55 * env
    duck = np.repeat(duck, hop)[:N][:, None]
    # duck music/rain/sfx only; dialogue stays at full level
    mix *= (0.6 + 0.4 * duck)
    mix += dlg_buf

    # ---------- master
    mix = np.tanh(mix * 1.15) / 1.15
    peak = np.abs(mix).max()
    mix = mix / peak * 0.89
    data = (mix * 32767).astype(np.int16)
    with wave.open(dst, 'wb') as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(data.tobytes())
    print('soundtrack ->', dst, round(len(mix) / SR, 2), 's')


def make_endcard(dst):
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    fpath = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans-Bold.ttf')
    fpath2 = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans.ttf')
    W, H = 1080, 1920
    y, x = np.mgrid[0:H, 0:W]
    g = np.clip(1.0 - (np.sqrt(((x - 540) / 700) ** 2 + ((y - 700) / 900) ** 2)), 0, 1)
    img = (np.stack([0.045, 0.05, 0.09], 0)[None, None] + 0.05 * g[..., None]) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    im = Image.fromarray(img, 'RGB')
    d = ImageDraw.Draw(im)
    f1 = ImageFont.truetype(fpath, 118)
    f2 = ImageFont.truetype(fpath2, 44)
    f3 = ImageFont.truetype(fpath2, 34)
    def center(txt, y, font, fill):
        w = d.textlength(txt, font=font)
        d.text(((W - w) / 2, y), txt, font=font, fill=fill)
    center('GLIMMER', 660, f1, (255, 214, 120))
    center('STATION', 800, f1, (255, 214, 120))
    center('a tiny night shift among stars', 970, f2, (190, 205, 235))
    center('PIP  ·  GLINT  ·  SCRAP', 1560, f3, (120, 135, 175))
    center('original story, score & design', 1620, f3, (90, 100, 130))
    im.save(dst)
    print('endcard ->', dst)


def main():
    shots = S.get_shots()
    parts = []
    for nm in ['s1','s2a','s2b','s3a','s3b','s4a','s4b','s5a','s5b','s6a','s6b','s7']:
        p = os.path.join(OUT, 'shots', nm + '.mp4')
        if os.path.exists(p):
            parts.append(p)
        else:
            raise SystemExit(f'missing shot: {p}')
    lst = os.path.join(OUT, 'concat.txt')
    with open(lst, 'w') as f:
        for p in parts:
            f.write("file '%s'\n" % p)
    card = os.path.join(OUT, 'endcard.png')
    make_endcard(card)
    silent = os.path.join(OUT, 'video_silent.mp4')
    fade_in_cmd = ("[0:v]scale=1080:1920:flags=lanczos,setsar=1,fps=30,format=yuv420p[base];"
                   "[1:v]format=rgba,fade=t=in:st=46.25:d=0.55:alpha=1,fade=t=out:st=47.55:d=0.45:alpha=1[card];"
                   "[base][card]overlay=0:0:shortest=0[v]")
    cmd = [FF, '-y', '-f', 'concat', '-safe', '0', '-i', lst,
           '-loop', '1', '-t', '48', '-i', card,
           '-filter_complex', fade_in_cmd, '-map', '[v]',
           '-c:v', 'libx264', '-preset', 'medium', '-crf', '17',
           '-pix_fmt', 'yuv420p', silent]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print('video rc', r.returncode)
    if r.returncode != 0:
        print(r.stderr[-2000:]); raise SystemExit(1)

    wav = os.path.join(OUT, 'soundtrack.wav')
    build_soundtrack(wav)

    final = os.path.join(OUT, 'final_animation_1080x1920.mp4')
    cmd = [FF, '-y', '-i', silent, '-i', wav,
           '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
           '-movflags', '+faststart', '-shortest', final]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print('mux rc', r.returncode)
    if r.returncode != 0:
        print(r.stderr[-2000:]); raise SystemExit(1)
    print('FINAL ->', final)


if __name__ == '__main__':
    main()
