"""Reproduce the Studio's downstream evidence on the 21 training-role Vocadito sources.

Arms, all scored with the SAME nearest-semitone segmenter and mir_eval (onset 50 ms, pitch 50 c),
mean of annotators A1/A2, first 8 s, singer-unit bootstrap (seed 339):
  pyin_original      = what the shared extractor does today
  pesto_unnormalized = pitch-evidence swap only, audio at its recorded level
  pesto_original     = evidence swap + peak normalisation (no cleanup)
  pesto_cleaned      = evidence + always-on spectral gate (also normalised)
  pesto_studio       = evidence + normalisation + the Studio's do-no-harm choice
Development evidence only: no reserved source or annotation is read.
Run: python tools/studio_evidence.py   (writes results/studio/evidence/pesto_vs_pyin.json)
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
import librosa
import numpy as np
import soundfile as sf

from bpdr_input.studio import clean, evidence as E

from bpdr_input.studio import scoring  # noqa: E402
mir_eval = scoring._mir_eval()           # the workspace's isolated mir_eval 0.8.2 copy

TRAIN_ROLE_IDS = [1, 3, 6, 8, 10, 13, 14, 18, 20, 25, 26, 27, 28, 29, 31, 32, 35, 36, 37, 38, 39]
import csv
SINGER_OF = {int(r["track_id"]): r["singer_id"] for r in csv.DictReader(open(ROOT / "datasets/vocadito/vocadito_metadata.csv"))}
NOISE = ROOT / "datasets/environmental_noise/1-100210-A-36.wav"
V = ROOT / "datasets/vocadito"
OUT = ROOT / "results/studio/evidence"


def load8(i):
    x, _ = librosa.load(V / "Audio" / f"vocadito_{i}.wav", sr=E.SR, duration=8.0)
    return x.astype(np.float32)


def mix(x, noise, snr):
    n = np.resize(noise, len(x)); g = np.sqrt((x ** 2).mean() / ((n ** 2).mean() * 10 ** (snr / 10)))
    y = x + g * n
    return (y / np.abs(y).max() * 0.9).astype(np.float32)


def ref(i, a):
    m = np.loadtxt(V / "Annotations/Notes" / f"vocadito_{i}_notes{a}.csv", delimiter=",", ndmin=2)
    k = m[:, 0] < 8.0
    on, hz, du = m[k, 0], m[k, 1], np.minimum(m[k, 2], 8.0 - m[k, 0])
    return np.c_[on, on + du], hz


def f1(i, notes):
    out = []
    for a in ("A1", "A2"):
        ri, rh = ref(i, a)
        if not notes or not len(rh):
            out.append(0.0); continue
        ei = np.array([[n["onset_s"], n["onset_s"] + n["duration_s"]] for n in notes])
        eh = np.array([440 * 2 ** ((n["pitch_midi"] - 69) / 12) for n in notes])
        out.append(mir_eval.transcription.precision_recall_f1_overlap(ri, rh, ei, eh, offset_ratio=None)[2])
    return float(np.mean(out))


def main():
    noise, _ = librosa.load(NOISE, sr=E.SR)
    arms = ("pyin_original", "pesto_unnormalized", "pesto_original", "pesto_cleaned", "pesto_studio")
    rows = {c: {a: [] for a in arms} for c in ("clean", "10dB", "0dB")}
    chosen = {c: [] for c in rows}
    for i in TRAIN_ROLE_IDS:
        x0 = load8(i)
        for cond, x in (("clean", x0), ("10dB", mix(x0, noise, 10)), ("0dB", mix(x0, noise, 0))):
            dur = len(x) / E.SR
            trk_u = E.pesto_track(x); trk_o = E.pesto_track(E.normalize(x)); cleaned = clean.spectral_gate(x); trk_c = E.pesto_track(cleaned)
            use, _ = clean.choose(trk_o, trk_c)
            rows[cond]["pyin_original"].append(f1(i, E.notes_from_track(E.pyin_track(x), dur)))
            rows[cond]["pesto_unnormalized"].append(f1(i, E.notes_from_track(trk_u, dur)))
            rows[cond]["pesto_original"].append(f1(i, E.notes_from_track(trk_o, dur)))
            rows[cond]["pesto_cleaned"].append(f1(i, E.notes_from_track(trk_c, dur)))
            rows[cond]["pesto_studio"].append(f1(i, E.notes_from_track(trk_c if use else trk_o, dur)))
            chosen[cond].append(bool(use))
        print("done", i, flush=True)
    rng = np.random.default_rng(339)
    summary = {}
    for cond, d in rows.items():
        a = {k: np.array(v) for k, v in d.items()}
        s = {"files": len(TRAIN_ROLE_IDS), "singers": len({SINGER_OF[i] for i in TRAIN_ROLE_IDS}), "mean_f1": {k: float(v.mean()) for k, v in a.items()},
             "cleaned_chosen": int(sum(chosen[cond]))}
        for arm in arms[1:]:
            diff = a[arm] - a["pyin_original"]
            singers = sorted({SINGER_OF[i] for i in TRAIN_ROLE_IDS})            # resample SINGERS, not files
            per = {sg: np.mean([diff[k] for k, i in enumerate(TRAIN_ROLE_IDS) if SINGER_OF[i] == sg]) for sg in singers}
            bs = [np.mean([per[singers[j]] for j in rng.integers(0, len(singers), len(singers))]) for _ in range(2000)]
            s[arm + "_minus_pyin"] = {"gain": float(diff.mean()), "bootstrap95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                      "better": int((diff > .02).sum()), "worse": int((diff < -.02).sum())}
        summary[cond] = s
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pesto_vs_pyin.json").write_text(json.dumps({
        "scope": "21 training-role Vocadito sources, first 8 s, ESC-50 clip 1-100210-A-36 mixed digitally. "
                 "Bootstrap resamples the 14 singers. Development evidence; one pre-chosen PESTO threshold (0.5); one digital noise clip (vacuum cleaner).",
        "per_file": rows, "summary": summary}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
