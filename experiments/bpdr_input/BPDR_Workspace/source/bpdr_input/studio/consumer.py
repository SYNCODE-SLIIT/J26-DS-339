"""The shared melody extractor exactly as the PP1 adapter calls it (analyze_vocal, trim=False). Read-only."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]


def shared_extractor_notes(path):
    """Notes (onset_s, duration_s, pitch_midi) from the shared pYIN baseline, or None if it is not importable."""
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from melody_understanding.pipeline import analyze_vocal
    except Exception:
        return None
    result, _features = analyze_vocal(str(path), trim=False)
    return [{"pitch_midi": n["pitch_midi"], "onset_s": n["onset_s"], "duration_s": n["duration_s"]} for n in result["notes"]]
