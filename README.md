# GLIMMER STATION

*A tiny night shift among stars* — an original 48-second 3D animated short.

**PIP**, a scrappy little night-shift keeper robot, works alone in a quiet
space-depot — until **GLINT**, a fallen star, crashes through the skylight
and lands in her hands. When **SCRAP**, a rigid old security drone, comes
scanning for the "unregistered luminous anomaly," PIP hides the star, sprints
the work-belt, and leaps shelf to shelf to keep it out of the scanner's beam.
In the end, the drone's own hatch becomes the star's launch window: GLINT
rises home through a beam of starlight, and PIP gets a tiny glowing ticket
stub as a keepsake.

## What's in the box

| Path | Contents |
| --- | --- |
| `output/final_animation_1080x1920.mp4` | **The finished film** — 1080×1920, 30 fps, H.264 + AAC, no watermarks |
| `web/player.html` | Preview + download page (play/pause, scrub, volume, fullscreen, time display, working **Download MP4** button) |
| `output/voice/*.wav` | 12 original dialogue lines (offline neural TTS, per-character voice processing) |
| `output/music.wav` | Original 48s orchestral-electronic score, generated procedurally |
| `output/sfx/*.wav` | 29 original synthesized sound effects (footsteps, whooshes, impacts, rain, scanner, skylight…) |
| `anim/`, `audio/`, `core/` | Full production pipeline source (renderer, rigs, choreography, audio) |
| `render_shot.py` | Per-shot render worker (frame resume + MP4 encode) |
| `assemble.py` | Shot concat, end-card overlay, full soundtrack mix, final mux |

Everything — story, characters, dialogue, music, sound, artwork — is original
and synthesized/ generated in-house. No copyrighted assets are used anywhere.

## Watch it right now (one command)

```bash
cd studio/web
cp ../studio/output/final_animation_1080x1920.mp4 .    # if not already there
python3 -m http.server 8000
```

Then open **http://localhost:8000/player.html** in any modern browser.

> **Why a local server?** Browsers restrict file access to a folder's files
> directly (`file://` works for the video on most browsers too, but some
> block the `<video>` source and the Download link from `file://`). A
> five-second local server is the reliable path.

Press play. The Download MP4 button grabs the exact same file the player
renders — full 1080×1920 H.264+AAC, playable in VLC, QuickTime, Premiere,
or uploadable straight to YouTube Shorts / TikTok / Reels.

## Rendering / rebuilding (optional)

The film is pre-rendered; nothing below is required to watch or download it.

```bash
# render all 12 shot segments in parallel (4 workers), resumable:
python3 render_shot.py s1 0.0 3.2      # ... one job per line range

# rebuild soundtrack + concat + end-card + final mux:
python3 assemble.py
```

Shot MP4s land in `output/shots/`; final film lands at
`output/final_animation_1080x1920.mp4`.

## Tech notes

- Renderer: custom analytic raytracer in pure Python + NumPy (soft shadows,
  Blinn-Phong + rim + 1-bounce reflections, fog, ACES tonemap, bloom, DOF).
- Characters: capsule/cylinder rigs with procedural facial rigs (blinking,
  gaze targets, 60 Hz lip-sync sampled from actually rendered voice WAVs).
- Audio: fully offline — Piper TTS voices, procedural score, additive SFX,
  comb-filter reverb plus sidechain ducking in the final mix.
- VFX: 3-tap temporal motion blur, depth of field on close-ups, dust motes,
  star sprites, galaxy beam, skylight flash, camera-shake on impacts, bloom.

© 2025 — original production, mid-space. Do whatever you like with it.
