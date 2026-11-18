"""Optional note-F1 against human references (uses the workspace's isolated mir_eval 0.8.2 copy)."""
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[3]
DEPS = WORKSPACE / "results/transformer_usability/evaluation_dependencies/mir_eval_0_8_2"


def _mir_eval():
    if str(DEPS) not in sys.path and DEPS.exists():
        sys.path.append(str(DEPS))
    import mir_eval
    return mir_eval


def note_f1(reference_csvs, notes, max_s=None):
    """Mean onset+pitch F1 (50 c / 50 ms) over annotator files. CSV columns: onset_s, hz, duration_s."""
    me = _mir_eval()
    scores = []
    for csv in reference_csvs:
        m = np.loadtxt(csv, delimiter=",", ndmin=2)
        if max_s is not None:
            k = m[:, 0] < max_s
            m = m[k]; m[:, 2] = np.minimum(m[:, 2], max_s - m[:, 0])
        if not len(m) or not notes:
            scores.append(0.0); continue
        ri, rh = np.c_[m[:, 0], m[:, 0] + m[:, 2]], m[:, 1]
        ei = np.array([[n["onset_s"], n["onset_s"] + n["duration_s"]] for n in notes])
        eh = 440 * 2 ** ((np.array([n["pitch_midi"] for n in notes]) - 69) / 12)
        scores.append(float(me.transcription.precision_recall_f1_overlap(ri, rh, ei, eh, offset_ratio=None)[2]))
    return float(np.mean(scores))
