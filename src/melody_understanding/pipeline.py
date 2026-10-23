"""Versioned baseline output for the melody-understanding component."""
from importlib.metadata import version
from pathlib import Path
import time
import numpy as np
from .audio import prepare_audio
from .features import extract_features, segment_notes

# Running the Complete Vocal Analysis Baseline
def analyze_vocal(path, trim=False):
    # Prepare the waveform, extract features and segment notes in source time.
    started = time.perf_counter()
    audio = prepare_audio(path, trim=trim)
    features = extract_features(audio)
    end = min(audio.source_duration_s, audio.offset_s + len(audio.samples) / audio.sample_rate)
    notes = segment_notes(features, end)
    # Summarize note durations by pitch class, then normalize to a distribution.
    pitch_class = np.zeros(12)
    for note in notes:
        pitch_class[note["pitch_midi"] % 12] += note["duration_s"]
    if pitch_class.sum():
        pitch_class /= pitch_class.sum()
    # Package predictions, processing details and method limitations as JSON-ready data.
    result = {
        "schema_version": "0.1.0", "component": "melody_understanding",
        "method": "pyin_baseline", "source_name": Path(path).name,
        "time_reference": "seconds from original audio start",
        "source_duration_s": audio.source_duration_s,
        "preprocessing": audio.metadata, "sample_rate": audio.sample_rate,
        "notes": notes,
        "pitch": {"times_s": features["times_s"].tolist(),
            "f0_hz": [float(x) if np.isfinite(x) else None for x in features["f0_hz"]],
            "voiced": features["voiced"].tolist(),
            "voiced_probability": features["voiced_probability"].tolist()},
        "acoustic_features": {"mel_shape": list(features["mel_db"].shape),
            "mel_unit": "dB relative to recording maximum", "hop_length": features["hop_length"],
            "frame_length": features["frame_length"], "cfp": None},
        "harmony_features": {"pitch_class_duration_distribution": pitch_class.tolist(),
            "provenance": "derived from baseline predicted notes; not chord labels"},
        "beats_s": None, "tempo_bpm": None, "key": None,
        "phrase_boundaries_s": None, "melody_embedding": None,
        "unsupported_reason": "CFP and learned musical attributes are pending implementation and supervision audit",
        "warnings": ["Voicing probability is not calibrated note correctness confidence",
            "Nearest-semitone segmentation may fragment vibrato or merge repeated same-pitch notes",
            "No supervised CNN-Transformer model has been trained"],
        "versions": {p: version(p) for p in ["numpy", "librosa", "soundfile"]},
        "inference_seconds": time.perf_counter() - started,
    }
    # Flag recordings where no note survives the minimum-duration requirement.
    if not notes:
        result["warnings"].append("No note events survived baseline segmentation")
    return result, features
