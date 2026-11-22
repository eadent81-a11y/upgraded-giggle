"""Voice generation: piper TTS -> per-character processing -> lip-sync envelopes."""
import numpy as np, wave, os, subprocess, sys
sys.path.insert(0, '/workspace/studio')
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
SR = 48000
VOICES = {'pip': ('/workspace/en_US-amy-medium.onnx', dict(pitch=1.22, robot=0.10, bright=1.15)),
          'scrap': ('/workspace/en_US-ryan-high.onnx', dict(pitch=0.62, robot=0.35, bright=0.9)),
          'narr': ('/workspace/en_US-amy-medium.onnx', dict(pitch=0.92, robot=0.0, bright=1.0))}


def _resample(x, sr_from, sr_to):
    import scipy.signal as sp
    g = np.gcd(sr_from, sr_to)
    return sp.resample_poly(x, sr_to//g, sr_from//g)


def _shift_pitch(x, factor):
    import scipy.signal as sp
    n = len(x)
    X = np.fft.rfft(x)
    N = int(n*factor)
    freqs = np.fft.rfftfreq(n, 1/SR)
    newfreqs = np.fft.rfftfreq(N, 1/SR)
    Y = np.interp(newfreqs, freqs, np.abs(X)) * np.exp(1j*np.interp(newfreqs, freqs, np.unwrap(np.angle(X))))
    y = np.fft.irfft(Y, N)
    return y


def _robotize(x, amount):
    """subtle ring mod + slight comb for robotic timbre"""
    if amount <= 0:
        return x
    t = np.arange(len(x))/SR
    ring = 1.0 - amount + amount*0.6*np.sin(2*np.pi*95*t) + amount*0.4*np.sin(2*np.pi*61*t)
    comb = x.copy()
    d = int(0.0023*SR)
    comb[d:] += x[:-d]*0.35*amount
    return comb*ring


def _hi_shelf(x, gain):
    import scipy.signal as sp
    b, a = sp.butter(2, 2600/(SR/2), 'high')
    return x + gain*0.5*sp.lfilter(b, a, x)


def synthesize_line(text, char, out_wav, length_scale=1.0, sentence_sil=0.18):
    from piper import PiperVoice, SynthesisConfig
    path, proc = VOICES[char]
    v = PiperVoice.load(path)
    raw = out_wav.replace('.wav', '_raw.wav')
    syn = SynthesisConfig(length_scale=length_scale)
    with wave.open(raw, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(22050)
        v.synthesize_wav(text, wav_file=wf, syn_config=syn)
    with wave.open(raw, 'rb') as wf:
        x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float64)/32768
    x = _resample(x, 22050, SR)
    x = _shift_pitch(x, proc['pitch'])
    x = _robotize(x, proc['robot'])
    x = _hi_shelf(x, proc['bright'])
    # normalize to -3 dBFS
    m = np.max(np.abs(x)) + 1e-9
    x = x/m*0.68
    # pad edges
    pad = np.zeros(int(0.02*SR))
    x = np.concatenate([pad, x, pad, np.zeros(int(sentence_sil*SR))])
    x16 = (np.clip(x, -1, 1)*32767).astype(np.int16)
    with wave.open(out_wav, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(x16.tobytes())
    os.remove(raw)
    return len(x)/SR


def load_wav(path):
    with wave.open(path, 'rb') as wf:
        assert wf.getframerate() == SR, path
        x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float64)/32768
    return x


def envelope_60hz(x):
    """speech energy envelope sampled at 60 Hz for lip-sync."""
    import scipy.signal as sp
    b, a = sp.butter(3, 9.0/(SR/2))
    e = np.abs(sp.lfilter(b, a, x))
    e = e/ (e.max()+1e-9)
    hop = SR//60
    n = len(x)//hop
    return np.array([e[i*hop:(i+1)*hop].max() for i in range(n)])
