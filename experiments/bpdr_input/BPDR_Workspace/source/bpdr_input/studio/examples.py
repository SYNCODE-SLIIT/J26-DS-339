"""Disclosed demonstration inputs. Sung examples come only from the 21 training-role Vocadito sources;
noise mixtures are digital and labelled as controlled; the speech clip is a public LibriSpeech excerpt."""
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .evidence import SR

WORKSPACE = Path(__file__).resolve().parents[3]
VOC = WORKSPACE / "datasets/vocadito"
NOISE = WORKSPACE / "datasets/environmental_noise"
OUT = WORKSPACE / "results/studio/examples"
NOISES = {"rain": "1-17367-A-10.wav", "vacuum cleaner": "1-100210-A-36.wav", "clock tick": "1-21934-A-38.wav",
          "breathing": "1-18631-A-23.wav"}


def _sing(i, seconds=8.0):
    x, _ = librosa.load(VOC / "Audio" / f"vocadito_{i}.wav", sr=SR, duration=seconds)
    return x.astype(np.float32)


def _mix(x, noise_name, snr_db):
    n, _ = librosa.load(NOISE / NOISES[noise_name], sr=SR)
    n = np.resize(n, len(x))
    g = np.sqrt((x ** 2).mean() / ((n ** 2).mean() * 10 ** (snr_db / 10)))
    y = x + g * n
    return (y / np.abs(y).max() * 0.9).astype(np.float32)


def build():
    """Create (once) and return the example list."""
    OUT.mkdir(parents=True, exist_ok=True)
    refs = lambda i: [str(VOC / "Annotations/Notes" / f"vocadito_{i}_notes{a}.csv") for a in ("A1", "A2")]
    clean10 = _sing(10)
    speech, _ = librosa.load(librosa.ex("libri1"), sr=SR, duration=12)
    speech = speech.astype(np.float32)
    spec = [
        ("hum_rain_0db", "Sung phrase + rain at 0 dB (controlled mix)", _mix(clean10, "rain", 0), refs(10),
         "Digitally mixed noise at the same level as the voice. Human note annotations exist for scoring."),
        ("hum_vacuum_0db", "Sung phrase + vacuum cleaner at 0 dB (controlled mix)", _mix(clean10, "vacuum cleaner", 0), refs(10),
         "A harsher, steady noise source on the same phrase."),
        ("hum_ticks_10db", "Different singer + clock ticks at 10 dB (controlled mix)", _mix(_sing(25), "clock tick", 10), refs(25),
         "Transient noise over another singer."),
        ("clean_hum", "Expressive clean phrase (nothing should be edited)", _sing(13), refs(13),
         "A good recording. The right behaviour is to leave it alone."),
        ("clipped", "Overloaded recording (heavy clipping)", np.clip(_sing(10) * 30, -1, 1).astype(np.float32), None,
         "Clipping is not repaired; the right behaviour is located re-record guidance."),
        ("silence", "Silence", (np.random.default_rng(1).standard_normal(SR) * 1e-7).astype(np.float32), None,
         "Nothing to prepare; technical failure, no accepted audio."),
        ("speech", "Reading text aloud (public speech clip)", speech, None,
         "Not a melody. The studio says so, and offers the optional speech-to-melody mode."),
        ("speech_noise", "Reading aloud with vacuum-cleaner noise at 5 dB", _mix(speech, "vacuum cleaner", 5), None,
         "Speech plus noise, to show the optional creative mode on a messy input."),
    ]
    items = []
    for key, label, wav, ref, note in spec:
        path = OUT / f"{key}.wav"
        if not path.exists():
            sf.write(path, wav, SR, subtype="PCM_24")
        items.append({"id": key, "label": label, "path": str(path), "reference": ref, "note": note})
    return items
