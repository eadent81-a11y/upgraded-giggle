"""Render one shot range to frames + MP4.

Usage: python3 render_shot.py <name> <t0> <t1> [start_index]
start_index: frame index to begin from (resume). Existing jpgs are skipped
anyway. If the part MP4 already exists, the job exits immediately.
"""
import os, sys, time, subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '1')

import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image
from core.renderer import Camera, render_frame, finalize, to_u8
import anim.shots as S

name, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
i_start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
FPS = S.FPS
W, H = S.W, S.H
fdir = os.path.join(S.ROOT, 'output', 'frames', name)
os.makedirs(fdir, exist_ok=True)

n_exp = int(round((t1 - t0) * FPS))
mp4 = os.path.join(S.ROOT, 'output', 'shots', name + '.mp4')
if os.path.exists(mp4):
    n_disk = len([f for f in os.listdir(fdir) if f.endswith('.jpg')])
    print(f'{name}: mp4 exists, {n_disk}/{n_exp} frames on disk, skipping', flush=True)
    sys.exit(0)

st = S.build_scene()
t_start = time.time()

frames = [t0 + i / FPS for i in range(n_exp)]

sub_frames = []
for ft in frames:
    subs = [ft]
    for a, b in S.MOTION_BLUR_WINDOWS:
        if a <= ft < b:
            subs = [ft, ft - 0.5 / FPS, ft + 0.5 / FPS]
            break
    sub_frames.append(subs)

for i in range(i_start, n_exp):
    ft = frames[i]
    fp = os.path.join(fdir, f'{i:05d}.jpg')
    if os.path.exists(fp):
        continue
    subs = sub_frames[i]
    acc = None
    for stt in subs:
        S.update(st, stt)
        c = S.camera(st, stt)
        dof = c.get('dof')
        cam = Camera(c['eye'], c['tgt'], fov_y=c['fov'], dof=dof, roll=c.get('roll', 0.0))
        r = np.random.default_rng(int(stt * 1000) + 7)
        col, _ = render_frame(st['scene'], cam, W, H, quality=1, rng=r)
        if acc is None:
            acc = col
        else:
            acc = acc + col
    acc /= len(subs)
    u8 = to_u8(finalize(acc.reshape(H, W, 3), exposure=S.exposure_at(ft)))
    Image.fromarray(u8).save(fp, quality=92)
    done = i - i_start + 1
    if done % 25 == 0 or done == n_exp - i_start:
        el = time.time() - t_start
        eta = el / done * (n_exp - i - 1)
        print(f'{name} {i+1}/{n_exp} el={el:.0f}s eta={eta:.0f}s', flush=True)

ff = imageio_ffmpeg.get_ffmpeg_exe()
os.makedirs(os.path.dirname(mp4), exist_ok=True)
cmd = [ff, '-y', '-framerate', str(FPS), '-i', os.path.join(fdir, '%05d.jpg'),
       '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p',
       '-vf', 'scale=1080:1920:flags=lanczos', mp4]
r = subprocess.run(cmd, capture_output=True, text=True)
print(name, 'mp4 rc', r.returncode, 'total', round(time.time() - t_start, 1), 's', flush=True)
