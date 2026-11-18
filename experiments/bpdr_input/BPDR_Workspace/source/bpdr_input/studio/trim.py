"""Silence trimming and <= 20 s phrase selection for vocal recordings.

In line with proposal scope limits:
- Maximum supported phrase duration is 20 s.
- Useful outcome for recordings > 20 s: offer the best <= 20 s phrase and trim
  leading/trailing silence with user confirmation, rather than arbitrary hard truncation.
"""
from typing import Dict, List, Optional, Tuple
import librosa
import numpy as np

from .evidence import SR, MAX_SECONDS, MIN_SECONDS


def detect_active_intervals(
    x: np.ndarray,
    sr: int = SR,
    top_db: float = 30.0,
    frame_length: int = 1024,
    hop_length: int = 256,
    min_silence_s: float = 0.4
) -> List[Tuple[float, float]]:
    """Detect non-silent intervals in seconds.

    Consecutive intervals separated by less than min_silence_s are merged to keep
    natural breathing and short pauses within a musical phrase.
    """
    if len(x) == 0:
        return []
    intervals = librosa.effects.split(
        x, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    if len(intervals) == 0:
        return []

    # Merge intervals separated by short pauses
    min_silence_samples = int(min_silence_s * sr)
    merged = [intervals[0].tolist()]
    for start, end in intervals[1:]:
        if start - merged[-1][1] < min_silence_samples:
            merged[-1][1] = int(end)
        else:
            merged.append([int(start), int(end)])

    return [(start / sr, end / sr) for start, end in merged]


def trim_silence(
    x: np.ndarray,
    sr: int = SR,
    top_db: float = 30.0,
    margin_s: float = 0.15
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Trim leading and trailing silence with a conservative margin.

    Returns:
        (trimmed_audio, metadata_dict)
    """
    orig_dur = len(x) / sr
    if orig_dur < MIN_SECONDS or np.max(np.abs(x)) < 1e-5:
        return x, {
            "original_duration_s": orig_dur,
            "start_s": 0.0,
            "end_s": orig_dur,
            "trimmed_leading_s": 0.0,
            "trimmed_trailing_s": 0.0,
            "was_trimmed": False
        }

    intervals = detect_active_intervals(x, sr=sr, top_db=top_db)
    if not intervals:
        return x, {
            "original_duration_s": orig_dur,
            "start_s": 0.0,
            "end_s": orig_dur,
            "trimmed_leading_s": 0.0,
            "trimmed_trailing_s": 0.0,
            "was_trimmed": False
        }

    start_s = max(0.0, intervals[0][0] - margin_s)
    end_s = min(orig_dur, intervals[-1][1] + margin_s)

    start_idx = int(start_s * sr)
    end_idx = min(len(x), int(end_s * sr))

    trimmed_x = x[start_idx:end_idx]
    trimmed_leading = float(start_s)
    trimmed_trailing = float(orig_dur - end_s)
    was_trimmed = (trimmed_leading > 0.05 or trimmed_trailing > 0.05)

    return trimmed_x, {
        "original_duration_s": float(orig_dur),
        "start_s": float(start_s),
        "end_s": float(end_s),
        "trimmed_leading_s": trimmed_leading,
        "trimmed_trailing_s": trimmed_trailing,
        "was_trimmed": bool(was_trimmed)
    }


def select_best_phrase(
    x: np.ndarray,
    sr: int = SR,
    max_duration_s: float = MAX_SECONDS,
    top_db: float = 30.0,
    margin_s: float = 0.15
) -> Tuple[np.ndarray, Dict[str, any]]:
    """Find and extract the best continuous <= max_duration_s phrase.

    If the recording is already <= max_duration_s, it performs leading/trailing
    silence trimming. If the recording exceeds max_duration_s, it searches active
    intervals to find the <= 20 s window containing the maximum voice activity/energy.

    Returns:
        (selected_audio, phrase_info_dict)
    """
    orig_dur = len(x) / sr
    trimmed_x, trim_meta = trim_silence(x, sr=sr, top_db=top_db, margin_s=margin_s)
    trimmed_dur = len(trimmed_x) / sr

    if trimmed_dur <= max_duration_s:
        info = {
            "original_duration_s": float(orig_dur),
            "start_s": trim_meta["start_s"],
            "end_s": trim_meta["end_s"],
            "duration_s": float(trimmed_dur),
            "phrase_selected": False,
            "trimmed_silence": trim_meta["was_trimmed"],
            "trimmed_leading_s": trim_meta["trimmed_leading_s"],
            "trimmed_trailing_s": trim_meta["trimmed_trailing_s"],
            "candidate_windows": []
        }
        return trimmed_x, info

    # Audio exceeds max_duration_s: find the best <= max_duration_s window
    intervals = detect_active_intervals(x, sr=sr, top_db=top_db)
    if not intervals:
        # Fallback to first 20s if no active intervals detected
        selected = x[: int(max_duration_s * sr)]
        return selected, {
            "original_duration_s": float(orig_dur),
            "start_s": 0.0,
            "end_s": float(max_duration_s),
            "duration_s": float(max_duration_s),
            "phrase_selected": True,
            "trimmed_silence": False,
            "trimmed_leading_s": 0.0,
            "trimmed_trailing_s": float(orig_dur - max_duration_s),
            "candidate_windows": []
        }

    # Evaluate candidate windows starting at interval onsets
    candidates = []
    hop = 256
    frame_rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=hop)[0]
    frame_times = librosa.frames_to_time(np.arange(len(frame_rms)), sr=sr, hop_length=hop)

    for i, (int_start, _) in enumerate(intervals):
        win_start = max(0.0, int_start - margin_s)
        win_end = min(orig_dur, win_start + max_duration_s)
        if win_end - win_start < MIN_SECONDS:
            continue

        # Score window by active RMS energy within it
        mask = (frame_times >= win_start) & (frame_times <= win_end)
        score = float(frame_rms[mask].sum()) if mask.any() else 0.0

        # Bonus for containing complete sub-intervals
        contained = sum(1 for s, e in intervals if s >= win_start and e <= win_end)
        score += contained * 0.1

        candidates.append({
            "start_s": float(win_start),
            "end_s": float(win_end),
            "duration_s": float(win_end - win_start),
            "score": float(score),
            "active_intervals_covered": contained
        })

    # Pick the highest-scoring candidate window
    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0] if candidates else {
        "start_s": 0.0,
        "end_s": float(max_duration_s),
        "duration_s": float(max_duration_s),
        "score": 0.0,
        "active_intervals_covered": 1
    }

    start_idx = int(best["start_s"] * sr)
    end_idx = min(len(x), int(best["end_s"] * sr))
    selected = x[start_idx:end_idx]

    info = {
        "original_duration_s": float(orig_dur),
        "start_s": best["start_s"],
        "end_s": best["end_s"],
        "duration_s": float(len(selected) / sr),
        "phrase_selected": True,
        "trimmed_silence": True,
        "trimmed_leading_s": float(best["start_s"]),
        "trimmed_trailing_s": float(orig_dur - best["end_s"]),
        "candidate_windows": candidates[:3]
    }
    return selected, info
