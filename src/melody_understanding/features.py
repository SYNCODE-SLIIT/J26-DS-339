"""Deterministic acoustic baseline; no trained research model is claimed."""
import numpy as np
import librosa

def extract_features(audio, hop_length=256, frame_length=4096):
    sr = audio.sample_rate
    mel = librosa.feature.melspectrogram(y=audio.samples, sr=sr,
        n_fft=frame_length, hop_length=hop_length, n_mels=80, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    f0, voiced, probability = librosa.pyin(audio.samples,
        fmin=librosa.note_to_hz("C1"), fmax=min(librosa.note_to_hz("C7"), sr / 2 - 1),
        sr=sr, frame_length=frame_length, hop_length=hop_length,
        fill_na=np.nan)
    times = audio.offset_s + np.arange(len(f0)) * hop_length / sr
    # Centered analysis can yield a frame exactly at the waveform end.
    keep = times < min(audio.source_duration_s,
        audio.offset_s + len(audio.samples) / sr)
    return {"mel_db": mel_db[:, :len(times)][:, keep], "f0_hz": f0[keep],
        "voiced": voiced[keep], "voiced_probability": probability[keep],
        "times_s": times[keep], "hop_length": hop_length,
        "frame_length": frame_length}

def segment_notes(features, end_s, minimum_duration_s=.06):
    """Group contiguous voiced frames by nearest MIDI pitch; preserve rests."""
    f0 = features["f0_hz"]
    valid = features["voiced"] & np.isfinite(f0) & (f0 > 0)
    pitches = np.full(len(f0), -1, dtype=int)
    pitches[valid] = np.rint(librosa.hz_to_midi(f0[valid])).astype(int)
    times = features["times_s"]
    notes = []
    start = 0
    while start < len(pitches):
        stop = start + 1
        while stop < len(pitches) and pitches[stop] == pitches[start]:
            stop += 1
        end = min(float(times[stop]) if stop < len(times) else end_s, end_s)
        onset = float(times[start])
        if 0 <= pitches[start] <= 127 and end - onset >= minimum_duration_s:
            notes.append({"pitch_midi": int(pitches[start]), "onset_s": onset,
                "duration_s": end - onset,
                "voicing_probability_mean": float(np.mean(features["voiced_probability"][start:stop]))})
        start = stop
    return notes
