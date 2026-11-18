"""Speech detection and the OPTIONAL speech-to-melody creative mode.

Scope label (shown in the UI): speech-to-melody COMPOSES a melody from the intonation and syllable
rhythm of spoken audio. It is not repair, does not recover an intended tune, and is outside the
BPDR research claim.
"""
import numpy as np
from scipy.signal import find_peaks

from .evidence import SR, frame_db

SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "pentatonic": [0, 2, 4, 7, 9]}   # chord models are fixed to major mode
SPEECH_PLATEAU_THRESHOLD = 0.22   # preliminary: 3 speech clips (<=0.16) vs 21 sung files (>=0.29)


def plateau_fraction(trk, tol_c=40.0, min_s=0.15):
    """Fraction of voiced frames inside runs that hold a steady pitch (sung/hummed notes do; speech glides)."""
    t, f0, v = trk["t"], trk["f0"], trk["voiced"]
    if v.sum() < 5 or len(t) < 3:
        return 0.0
    dt = float(np.median(np.diff(t)))
    cents = np.where(v, 1200 * np.log2(np.where(v, f0, 440.0) / 440.0), np.nan)
    held, i, n = 0, 0, len(cents)
    while i < n:
        if np.isfinite(cents[i]):
            j = i + 1
            while j < n and np.isfinite(cents[j]) and abs(cents[j] - np.median(cents[i:j + 1])) < tol_c:
                j += 1
            if (j - i) * dt >= min_s:
                held += j - i
            i = max(j, i + 1)
        else:
            i += 1
    return held / max(int(v.sum()), 1)


def speech_likeness(trk):
    p = plateau_fraction(trk)
    return {"plateau_fraction": p, "looks_like_speech": bool(p < SPEECH_PLATEAU_THRESHOLD and trk["voiced"].mean() > 0.15),
            "threshold": SPEECH_PLATEAU_THRESHOLD,
            "validation_note": "Preliminary threshold from 3 speech and 21 sung development files."}


def _syllables(x, trk, sr=SR):
    """Syllable-like spans: energy peaks inside voiced regions, bounded by the valleys between them."""
    t, v = trk["t"], trk["voiced"]
    db = frame_db(x, sr, 0.03)
    et = (np.arange(len(db)) + 0.5) * 0.03
    vo = np.interp(et, t, v.astype(float)) > 0.5
    db = np.where(vo, db, db.min())
    sm = np.convolve(db, np.ones(3) / 3, mode="same")
    peaks, _ = find_peaks(sm, prominence=3.0, distance=4)
    peaks = [p for p in peaks if vo[p]]
    spans = []
    for k, p in enumerate(peaks):
        lo = et[0] if k == 0 else et[int((peaks[k - 1] + p) // 2)]
        hi = et[-1] if k == len(peaks) - 1 else et[int((p + peaks[k + 1]) // 2)]
        m = (t >= lo) & (t < hi) & v
        if m.sum() >= 3:
            vt = t[m]
            spans.append({"onset": float(vt[0]), "end": float(vt[-1]), "f0": float(np.median(trk["f0"][m]))})
    return spans


def speech_to_melody(x, trk, tonic_pc=0, scale="major", bpm=100.0, max_bars=8, sr=SR):
    """Compose a short melody from speech intonation (relative pitch contour) and syllable rhythm."""
    spans = _syllables(x, trk, sr)
    if len(spans) < 3:
        raise ValueError("Too few syllable-like events to compose a melody; record a longer or clearer passage")
    eighth = 30.0 / bpm
    ioi = np.diff([s["onset"] for s in spans])
    stretch = eighth / float(np.clip(np.median(ioi), 0.12, 0.6))      # speech syllables -> musical eighths
    rel = 12 * np.log2(np.array([s["f0"] for s in spans]) / np.median([s["f0"] for s in spans]))
    rel = np.where(np.abs(rel) > 9, rel - 12 * np.round(rel / 12), rel)    # fold tracker octave errors
    rel = np.clip(rel, -9, 9)
    spread = max(float(np.percentile(rel, 95) - np.percentile(rel, 5)), 1.0)
    gain = float(np.clip(9.0 / spread, 0.6, 1.5))                     # keep the tune within roughly an octave
    allowed = SCALES[scale]
    centre = 60 + tonic_pc if tonic_pc < 6 else 48 + tonic_pc
    pool = [o * 12 + tonic_pc + d for o in range(3, 8) for d in allowed]
    pitches = [min(pool, key=lambda m, r=r: abs(m - (centre + gain * r))) for r in rel]
    t0 = spans[0]["onset"]
    grid = [int(round((s["onset"] - t0) * stretch / eighth)) for s in spans]
    notes, last_end = [], -1
    for k, (g, p) in enumerate(zip(grid, pitches)):
        g = max(g, last_end)
        nxt = grid[k + 1] if k + 1 < len(grid) else g + 2
        length = int(np.clip(max(nxt - g, 1), 1, 6))
        if g + length > max_bars * 8:
            break
        notes.append({"pitch_midi": int(p), "onset_s": g * eighth, "duration_s": length * eighth})
        last_end = g + length
    if len(notes) < 3:
        raise ValueError("Composed melody is too short")
    notes[-1]["pitch_midi"] = int(min(pool, key=lambda m: abs(m - (notes[-1]["pitch_midi"] // 12 * 12 + tonic_pc))))  # land on the tonic
    total_beats = int(np.ceil((notes[-1]["onset_s"] + notes[-1]["duration_s"]) / (60 / bpm) / 4) * 4)
    return {"notes": notes, "bpm": bpm, "tonic_pc": tonic_pc, "scale": scale, "meter": 4,
            "beats_s": [i * 60 / bpm for i in range(total_beats)], "syllables_found": len(spans),
            "label": "Composed from speech intonation and rhythm. Not repair; outside the BPDR research claim."}
