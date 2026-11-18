"""Step-0 control: is the PESTO-vs-pYIN gap a tracker effect or a frame-size/fmin effect?

Declared in documentation/BPDR_Studio_Build_Plan_2026-11-10.md section 14 BEFORE this was run.
Same 21 training-role Vocadito sources, first 8 s, same noise clip/SNRs, same nearest-semitone segmenter,
hop fixed at 256. Only the pYIN frame length and fmin vary. Development evidence only.
Run: python tools/studio_frame_control.py  -> results/studio/evidence/frame_control.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "tools"))
import librosa
import numpy as np

import studio_evidence as SE
from bpdr_input.studio import evidence as E

mir = SE.mir_eval
IDS, SINGER_OF = SE.TRAIN_ROLE_IDS, SE.SINGER_OF
ARMS = {"pyin_4096_C1_today": (4096, "C1"), "pyin_4096_C2": (4096, "C2"),
        "pyin_2048_C2": (2048, "C2"), "pyin_1024_C2": (1024, "C2")}


def scores(i, notes):
    """mean over A1/A2 of (onset+pitch F1 @50ms/50c, onset-only F1 @50ms, onset-only F1 @100ms)."""
    out = []
    for a in ("A1", "A2"):
        ri, rh = SE.ref(i, a)
        if not notes or not len(rh):
            out.append((0.0, 0.0, 0.0)); continue
        ei = np.array([[n["onset_s"], n["onset_s"] + n["duration_s"]] for n in notes])
        eh = np.array([440 * 2 ** ((n["pitch_midi"] - 69) / 12) for n in notes])
        full = mir.transcription.precision_recall_f1_overlap(ri, rh, ei, eh, offset_ratio=None)[2]
        o50 = mir.transcription.onset_precision_recall_f1(ri, ei, onset_tolerance=0.05)[2]
        o100 = mir.transcription.onset_precision_recall_f1(ri, ei, onset_tolerance=0.10)[2]
        out.append((full, o50, o100))
    return np.mean(out, axis=0)


def main():
    noise, _ = librosa.load(SE.NOISE, sr=E.SR)
    names = list(ARMS) + ["pesto_normalised"]
    rows = {c: {a: [] for a in names} for c in ("clean", "10dB", "0dB")}
    for i in IDS:
        x0 = SE.load8(i)
        for cond, x in (("clean", x0), ("10dB", SE.mix(x0, noise, 10)), ("0dB", SE.mix(x0, noise, 0))):
            dur = len(x) / E.SR
            for n, (fl, fm) in ARMS.items():
                rows[cond][n].append(scores(i, E.notes_from_track(E.pyin_track(x, frame_length=fl, fmin_note=fm), dur)).tolist())
            rows[cond]["pesto_normalised"].append(scores(i, E.notes_from_track(E.pesto_track(E.normalize(x)), dur)).tolist())
        print("done", i, flush=True)
    rng = np.random.default_rng(339)
    singers = sorted({SINGER_OF[i] for i in IDS})
    metric = ("onset_pitch_f1_50ms", "onset_only_f1_50ms", "onset_only_f1_100ms")
    summary = {}
    for cond, d in rows.items():
        a = {k: np.array(v) for k, v in d.items()}
        s = {"mean": {k: dict(zip(metric, v.mean(0).round(4).tolist())) for k, v in a.items()}, "vs_today": {}}
        for arm in names[1:]:
            for mi, mn in enumerate(metric):
                diff = a[arm][:, mi] - a["pyin_4096_C1_today"][:, mi]
                per = {sg: np.mean([diff[k] for k, i in enumerate(IDS) if SINGER_OF[i] == sg]) for sg in singers}
                bs = [np.mean([per[singers[j]] for j in rng.integers(0, len(singers), len(singers))]) for _ in range(2000)]
                s["vs_today"][f"{arm}|{mn}"] = {"gain": round(float(diff.mean()), 4),
                                                "bootstrap95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
        summary[cond] = s
    out = ROOT / "results/studio/evidence"
    out.mkdir(parents=True, exist_ok=True)
    (out / "frame_control.json").write_text(json.dumps({
        "scope": "21 training-role Vocadito sources, 14 singers, first 8 s, hop 256 fixed, singer-level bootstrap seed 339. Development evidence.",
        "metrics": metric, "arms": {k: list(v) for k, v in ARMS.items()}, "per_file": rows, "summary": summary}, indent=1), encoding="utf-8")
    print(json.dumps({c: s["mean"] for c, s in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
