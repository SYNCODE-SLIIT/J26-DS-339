"""Pitch evidence and recording-quality measurements for the Vocal Rescue Studio.

Everything here is measurement. Notes produced by ``notes_from_track`` mirror the
nearest-semitone rule of the shared melody extractor so a before/after view shows
what that extractor would do; they are not BPDR's own note decisions.
"""
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 22050
HOP = 256            # same analysis grid as the shared pYIN baseline
FRAME = 4096
MAX_SECONDS = 20.0   # proposal: maximum supported length
MIN_SECONDS = 0.25
PESTO_MODEL = "mir-1k_g7"


def load_mono(path, sr=SR, max_s=MAX_SECONDS):
    """Decode, mix to mono without phase cancellation, resample. Never edits the source file."""
    data, original_sr = sf.read(Path(path), dtype="float32", always_2d=True)
    if not data.size or not np.isfinite(data).all():
        raise ValueError("Audio is empty or contains non-finite samples")
    channels = data.shape[1]
    if channels == 2:
        a, b = data[:, 0], data[:, 1]
        denom = float(np.sqrt((a * a).sum() * (b * b).sum())) or 1.0
        corr = float((a * b).sum()) / denom
        # Opposed channels would cancel when averaged; keep the louder one instead.
        mono = (a if (a * a).sum() >= (b * b).sum() else b) if corr < -0.2 else data.mean(axis=1)
    else:
        mono = data.mean(axis=1)
    peak = float(np.max(np.abs(mono)))
    if peak < 1e-5:
        raise ValueError("Audio is silent or too quiet")
    # Clipping is measured on the DECODED samples (every channel, original rate) before mixing/resampling smears flat tops.
    loudest = np.max(np.abs(data), axis=1)
    clip_spans = clipped_spans(loudest, original_sr)
    clip_fraction = float(np.mean(loudest >= 0.999))
    duration = len(mono) / original_sr
    if original_sr != sr:
        mono = librosa.resample(mono, orig_sr=original_sr, target_sr=sr)
    truncated = (max_s is not None) and (len(mono) > max_s * sr)
    if max_s is not None:
        mono = mono[: int(max_s * sr)]
    return mono.astype(np.float32), {"original_sample_rate": original_sr, "channels": channels,
                                    "source_duration_s": duration, "truncated_to_max": truncated,
                                    "peak": peak, "clipped_spans_s": clip_spans, "clipped_fraction": clip_fraction}


def normalize(x, peak=0.95):
    return (x * (peak / max(float(np.max(np.abs(x))), 1e-12))).astype(np.float32)


def normalization_gain(x, peak=0.95):
    return float(peak / max(float(np.max(np.abs(x))), 1e-12))


def pyin_track(x, sr=SR, frame_length=FRAME, fmin_note="C1"):
    """Defaults reproduce the shared extractor; other values exist only for the frame-size control."""
    f0, voiced, prob = librosa.pyin(x, fmin=librosa.note_to_hz(fmin_note), fmax=min(librosa.note_to_hz("C7"), sr / 2 - 1),
                                    sr=sr, frame_length=frame_length, hop_length=HOP, fill_na=np.nan)
    t = np.arange(len(f0)) * HOP / sr
    keep = t < len(x) / sr
    return {"t": t[keep], "f0": f0[keep], "voiced": voiced[keep] & np.isfinite(f0[keep]), "conf": prob[keep],
            "tracker": "pyin"}


def pesto_track(x, sr=SR, threshold=0.5):
    import torch
    import pesto
    ts, pitch, conf, _ = pesto.predict(torch.from_numpy(np.ascontiguousarray(x)), sr, model_name=PESTO_MODEL)
    t = ts.numpy().astype(float)
    t = t / 1000.0 if t.max() > 100 else t
    f0, c = pitch.numpy().astype(float), conf.numpy().astype(float)
    voiced = (c > threshold) & (f0 > 0)
    f0 = np.where(voiced, f0, np.nan)
    return {"t": t, "f0": f0, "voiced": voiced, "conf": c, "tracker": "pesto", "threshold": threshold}


def notes_from_track(trk, end_s, min_dur=0.06):
    """Nearest-semitone grouping of contiguous voiced frames (mirrors the shared extractor's rule)."""
    f0, voiced, t = trk["f0"], trk["voiced"], trk["t"]
    ok = voiced & np.isfinite(f0) & (f0 > 0)
    midi = np.full(len(f0), -1, dtype=int)
    midi[ok] = np.rint(librosa.hz_to_midi(f0[ok])).astype(int)
    notes, start = [], 0
    while start < len(midi):
        stop = start + 1
        while stop < len(midi) and midi[stop] == midi[start]:
            stop += 1
        end = min(float(t[stop]) if stop < len(t) else end_s, end_s)
        onset = float(t[start])
        if 0 <= midi[start] <= 127 and end - onset >= min_dur:
            notes.append({"pitch_midi": int(midi[start]), "onset_s": onset, "duration_s": end - onset})
        start = stop
    return notes


def clipped_spans(x, sr=SR, level=0.999, min_samples=3):
    """Time ranges (seconds) where the waveform sits at full scale."""
    hit = np.abs(x) >= level
    spans, i = [], 0
    while i < len(hit):
        if hit[i]:
            j = i
            while j < len(hit) and hit[j]:
                j += 1
            if j - i >= min_samples:
                spans.append((i / sr, j / sr))
            i = j
        else:
            i += 1
    return spans


def frame_db(x, sr=SR, frame_s=0.05):
    n = int(frame_s * sr)
    if len(x) < n:
        return np.array([20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9)])
    k = len(x) // n
    rms = np.sqrt((x[: k * n].reshape(k, n) ** 2).mean(axis=1))
    return 20 * np.log10(rms + 1e-9)


def quality_report(x, trk, sr=SR, raw_peak=None):
    """Technical measurements only. Quantities are proxies, not note-accuracy measurements."""
    db = frame_db(x, sr)
    floor, top = float(np.percentile(db, 10)), float(np.percentile(db, 90))
    voiced = trk["voiced"]
    jumps = 0
    midi = np.where(voiced, librosa.hz_to_midi(np.where(voiced, trk["f0"], 440.0)), np.nan)
    prev = None
    for m in midi:
        if np.isfinite(m):
            if prev is not None and abs(abs(m - prev) - 12) < 1.0:
                jumps += 1
            prev = m
        else:
            prev = None
    spans = clipped_spans(x if raw_peak is None else x * raw_peak / max(float(np.max(np.abs(x))), 1e-12), sr)
    return {"duration_s": len(x) / sr, "level_db": float(20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9)),
            "noise_floor_db": floor, "snr_estimate_db": top - floor,
            "voiced_ratio": float(voiced.mean()) if len(voiced) else 0.0,
            "mean_confidence_voiced": float(trk["conf"][voiced].mean()) if voiced.any() else 0.0,
            "octave_jumps": int(jumps), "clipped_spans_s": spans,
            "clipped_fraction": float(np.mean(np.abs(x) >= 0.999))}
