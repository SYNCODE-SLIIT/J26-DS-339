"""Simple additive rendering so a panel can HEAR what a note list or pitch track says."""
import numpy as np

from .evidence import SR

HARMONICS = (1.0, 0.5, 0.25, 0.12)


def midi_hz(m):
    return 440.0 * 2 ** ((np.asarray(m, float) - 69) / 12)


def _tone(freqs, n, sr, amp):
    t = np.arange(n) / sr
    return amp * sum(h * np.sin(2 * np.pi * f * (k + 1) * t) for k, h in enumerate(HARMONICS) for f in freqs) / 1.9


def render_notes(notes, duration_s, sr=SR, amp=0.25):
    out = np.zeros(int(duration_s * sr) + 1, dtype=np.float32)
    for n in notes:
        a = int(n["onset_s"] * sr)
        length = max(int(n["duration_s"] * sr), 1)
        seg = _tone([float(midi_hz(n["pitch_midi"]))], length, sr, amp)
        atk, rel = min(int(.01 * sr), length // 2), min(int(.04 * sr), length // 2)
        env = np.ones(length)
        if atk: env[:atk] = np.linspace(0, 1, atk)
        if rel: env[-rel:] *= np.linspace(1, 0, rel)
        b = min(a + length, len(out))
        out[a:b] += (seg * env)[: b - a].astype(np.float32)
    return out


def render_chords(chords, duration_s, sr=SR, amp=0.12):
    """chords: [{start_s, end_s, midi:[...]}]."""
    out = np.zeros(int(duration_s * sr) + 1, dtype=np.float32)
    for c in chords:
        a, b = int(c["start_s"] * sr), min(int(c["end_s"] * sr), len(out))
        if b <= a: continue
        seg = _tone([float(midi_hz(m)) for m in c["midi"]], b - a, sr, amp / max(len(c["midi"]), 1) * 2)
        env = np.ones(b - a); r = min(int(.05 * sr), (b - a) // 2)
        if r: env[:r] = np.linspace(0, 1, r); env[-r:] *= np.linspace(1, 0, r)
        out[a:b] += (seg * env).astype(np.float32)
    return out


def render_contour(trk, sr=SR, amp=0.2):
    """Continuous-phase tone following a pitch track (what the tracker 'heard')."""
    t, f0, v = trk["t"], trk["f0"], trk["voiced"]
    n = int(t[-1] * sr) + 1 if len(t) else 0
    grid = np.arange(n) / sr
    f = np.interp(grid, t, np.where(v, np.nan_to_num(f0, nan=0.0), 0.0))
    on = np.interp(grid, t, v.astype(float)) > 0.5
    phase = 2 * np.pi * np.cumsum(f) / sr
    return (amp * np.sin(phase) * on).astype(np.float32)


def mix(*parts, peak=0.9):
    n = max(len(p) for p in parts)
    y = np.zeros(n, dtype=np.float32)
    for p in parts: y[: len(p)] += p
    return (y * (peak / max(float(np.max(np.abs(y))), 1e-9))).astype(np.float32)
