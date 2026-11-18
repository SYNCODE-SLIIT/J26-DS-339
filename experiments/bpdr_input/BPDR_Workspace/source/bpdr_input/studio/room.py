"""Room check: a few seconds of silence tell us how noisy the room is and give a voice-free noise reference.

Technical measurements only. Thresholds are preliminary and untested on real rooms (no user pilot yet).
"""
import librosa
import numpy as np

from .evidence import SR

MIN_S = 1.0
GOOD_DB, NOISY_DB = -55.0, -40.0           # RMS level of the silent clip; preliminary
LOUD_VOICE_DB = -20.0                      # a "silent" clip this loud probably contains the voice or music


def _hum(x, sr):
    """Strongest mains hum (50 or 60 Hz and harmonics) as dB above the surrounding spectrum, or None."""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))); f = np.fft.rfftfreq(len(x), 1 / sr)
    best = None
    for base in (50.0, 60.0):
        gains = []
        for k in (1, 2, 3, 4):
            c = base * k
            core = spec[(f > c - 1.5) & (f < c + 1.5)]
            side = spec[((f > c - 12) & (f < c - 4)) | ((f > c + 4) & (f < c + 12))]
            if len(core) and len(side) and side.mean() > 0:
                gains.append(20 * np.log10(core.max() / side.mean() + 1e-12))
        if gains and (best is None or np.mean(gains) > best[1]):
            best = (base, float(np.mean(gains)))
    return best


def room_check(x, sr=SR):
    """x: mono float audio of the silent clip. Returns measurements, a verdict and one plain-language tip."""
    if len(x) / sr < MIN_S:
        return dict(verdict="unusable", tip="Stay silent for at least 2 seconds so the room can be measured.", usable_reference=False)
    x = np.asarray(x, dtype=np.float32)
    level = float(20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9))
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256)) + 1e-9
    flat = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    frame_db = 20 * np.log10(librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0] + 1e-9)
    events = float(np.percentile(frame_db, 95) - np.median(frame_db))
    hum = _hum(x, sr)
    hum_db = hum[1] if hum else 0.0
    peak = float(np.max(np.abs(x)))
    out = dict(level_db=level, flatness=flat, event_range_db=events, hum_hz=hum[0] if hum and hum_db > 12 else None,
               hum_strength_db=hum_db, peak=peak)
    if level > LOUD_VOICE_DB or peak >= 0.98:
        out.update(verdict="too_loud", usable_reference=False,
                   tip="That clip is very loud. Stay silent, then try again; if it is still loud, move away from the noise source.")
    elif events > 12:
        out.update(verdict="noisy", usable_reference=False,
                   tip="A sudden sound happened during the silence (a click, voice or knock), so it is not a steady room tone. Try again.")
    elif out["hum_hz"]:
        out.update(verdict="noisy", usable_reference=True,
                   tip=f"There is a steady {out['hum_hz']:.0f} Hz hum (a power supply, light or appliance). Switching it off or moving away helps.")
    elif level > NOISY_DB:
        out.update(verdict="noisy", usable_reference=True, tip="The room is noisy (fan, traffic or air conditioning). A quieter room or a closer microphone helps.")
    else:
        out.update(verdict="good", usable_reference=True, tip="The room is quiet enough.")
    out["verdict_thresholds_db"] = dict(good=GOOD_DB, noisy=NOISY_DB, preliminary=True)
    return out
