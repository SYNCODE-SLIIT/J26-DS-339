"""Hand a melody to the teammates' existing chord models (Pasan planner, Rahul refinement), read-only.

BPDR does not own this stage; it only calls the already-built adapters in ``pp1_demo`` so the panel
can hear the effect of better input. If the models or their checkpoints are unavailable the call
fails with a clear message and nothing is fabricated.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _root_pc(label):
    m = re.match(r"([A-G])([b#]?)", label)
    return (PC[m.group(1)] + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)) % 12


def label_to_midi(label, octave_base=48):
    """Voice a chord label as a close-position triad (+ minor 7th when the label has a 7)."""
    rest = label.split(":", 1)[1] if ":" in label else label[1:].lstrip("b#")
    root = octave_base + _root_pc(label)
    low = rest.lower()
    if low.startswith("dim") or "dim" in low or "hdim" in low or "m7b5" in low:
        iv = [0, 3, 6]
    elif "aug" in low:
        iv = [0, 4, 8]
    elif low.startswith("min") or (low.startswith("m") and not low.startswith("maj")):
        iv = [0, 3, 7]
    elif "sus" in low:
        iv = [0, 5, 7]
    else:
        iv = [0, 4, 7]
    if "7" in low and "maj7" not in low:
        iv = iv + [10]
    elif "maj7" in low:
        iv = iv + [11]
    return [root + i for i in iv]


def _tile_to_sixteen_bars(notes, bpm, meter=4):
    """Pasan's input is a fixed 16-bar window; repeat the phrase to fill it (a documented creative choice)."""
    bar_s = meter * 60.0 / bpm
    total = 16 * bar_s
    length = max(n["onset_s"] + n["duration_s"] for n in notes)
    period = max(bar_s, -(-length // bar_s) * bar_s)
    out, k = [], 0
    while k * period < total:
        for n in notes:
            on = n["onset_s"] + k * period
            if on < total:
                out.append(dict(n, onset_s=on, duration_s=min(n["duration_s"], total - on)))
        k += 1
    return out, total


def harmonize(notes, bpm, tonic_pc, genre="pop"):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    try:
        from pp1_demo.adapters import pasan, rahul
        from pp1_demo.contracts import Settings, beat_grid
    except Exception as exc:
        raise RuntimeError("Chord stage unavailable (pp1_demo adapters could not be imported): " + str(exc)[:200])
    tiled, total = _tile_to_sixteen_bars(notes, bpm)
    settings = Settings(tonic_pc=tonic_pc, bpm=bpm, genre=genre)
    beats = [dict(b, meter=4) for b in beat_grid(total, settings)]
    plan = pasan.predict(tiled, beats, settings)
    refined = rahul.predict(tiled, plan["chords"], settings)
    chords = [{"start_s": c["start_s"], "end_s": c["end_s"], "label": c["label"], "midi": label_to_midi(c["label"])}
              for c in refined["chords"]]
    return {"chords": chords, "tiled_notes": tiled, "duration_s": total, "genre": genre,
            "models": {"planner": "Pasan temporal Transformer", "refiner": "Rahul multi-task Transformer"},
            "note": "Melody phrase repeated to fill the planner's fixed 16-bar window."}
