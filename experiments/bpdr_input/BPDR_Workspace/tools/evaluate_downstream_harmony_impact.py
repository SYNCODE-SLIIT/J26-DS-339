"""R-3. DECLARED BEFORE RUN (2026-11-23). Read-only downstream study on the 39 Vocadito training-role windows.

Question: do the notes Component 1 produces change the chords Pasan -> Rahul plan, and are those chords closer
to the chords planned from the human reference notes (A1)?

Arms (same bpm=100, genre=pop, tonic = most frequent pitch class of the A1 notes in ALL arms, so only notes differ):
  pyin_4096                  today's shared extractor (pYIN, 4096 frame, C1)
  pesto_notes_from_track     PESTO track, nearest-semitone runs
  studio_v1                  PESTO track + learned boundary hints + segment-median rule
  studio_v1_aic_k3_oracle    studio_v1 plus 3 EIG-selected boundary confirmations answered by the A1 oracle (upper bound)
  oracle_A1                  human reference notes (the target)

Metric: chord label sampled every 0.25 s over the window; agreement (%) of each arm with oracle_A1 chords.
Paired singer-level 95% bootstrap of (arm - pyin_4096). Report all results including nulls. The strong-beat
melody-chord compatibility metric from the proposal is NOT computed here (needs the downstream compatibility scorer).
Reserved splits (HumTrans M03-M05, VALID, TEST) are not touched.
"""
import json
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WS / "source"), str(WS)]

from bpdr_input.studio import boundary, chords as C, clean, evidence as E, question_engine as qe
from tools.analyze_r1_e2e import RNG_SEED, bootstrap_ci, singer_metric_ci
from tools.benchmark_studio_vocadito import load_vocadito_windows
from tools.evaluate_aic_dev import _apply_boundary_answers

BPM = 100
OUT = WS / "results" / "studio" / "aic" / "downstream_harmony_impact.json"


def ref_to_notes(ref):
    notes = []
    for i, (on, midi) in enumerate(ref):
        nxt = ref[i + 1][0] if i + 1 < len(ref) else on + 0.5
        notes.append(dict(onset_s=float(on), duration_s=float(max(0.05, min(1.0, nxt - on))), pitch_midi=int(round(midi))))
    return notes


def chord_labels(notes, tonic):
    if not notes:
        return None
    h = C.harmonize(notes, BPM, tonic)
    t = np.arange(0, h["duration_s"], 0.25)
    return np.array([next((c["label"] for c in h["chords"] if c["start_s"] <= x < c["end_s"]), "N") for x in t])


def main():
    rows = []
    for w in load_vocadito_windows():
        x, _ = E.load_mono(w["wav_path"], sr=E.SR, max_s=None)
        s0 = int(round(w["offset_s"] * E.SR))
        win = x[s0:s0 + int(round(w["duration_s"] * E.SR))]
        dur = len(win) / float(E.SR)
        a1 = w["refs"].get("A1", [])
        if len(a1) < 3:
            continue
        ref_notes = ref_to_notes(a1)
        tonic = int(np.bincount([n["pitch_midi"] % 12 for n in ref_notes], minlength=12).argmax())

        xn = E.normalize(win)
        trk_o = E.pesto_track(xn)
        cleaned = clean.spectral_gate(win)
        trk_c = E.pesto_track(cleaned)
        use_clean, _ = clean.choose(trk_o, trk_c)
        prep = cleaned if use_clean else xn
        trk_p = trk_c if use_clean else trk_o

        hints = boundary.extract_boundary_hints(trk_p, x=prep, sr=E.SR)
        cands = [c for c in qe.extract_ambiguity_candidates(trk_p, x=prep, sr=E.SR, boundary_hints=hints,
                                                           span_diagnostics={"spans": []}, duration_s=dur)
                 if c["factor"] == "boundary"]
        cands = qe.calibrate_candidate_probabilities(cands)
        sel = qe.select_questions(cands, k=3, strategy="eig", seed=RNG_SEED)
        a1_on = np.array([o for o, _ in a1], dtype=float)
        aic_peaks = _apply_boundary_answers(list(hints.get("peaks_s", [])), sel, a1_on)

        arms = {
            "pyin_4096": E.notes_from_track(E.pyin_track(win, frame_length=4096, fmin_note="C1"), dur),
            "pesto_notes_from_track": E.notes_from_track(trk_p, dur),
            "studio_v1": boundary.notes_from_track_with_boundary_hints(trk_p, hints, end_s=dur, sr=E.SR),
            "studio_v1_aic_k3_oracle": boundary.notes_from_track_with_boundary_hints(trk_p, {"peaks_s": aic_peaks}, end_s=dur, sr=E.SR),
        }
        oracle = chord_labels(ref_notes, tonic)
        row = {"id": w["id"], "singer": w["person"]}
        ok = oracle is not None
        for name, notes in arms.items():
            lab = chord_labels(notes, tonic) if notes else None
            if lab is None or not ok:
                ok = False
                break
            n = min(len(lab), len(oracle))
            row[name] = float(np.mean(lab[:n] == oracle[:n]))
        if ok:
            rows.append(row)
        print("done", w["id"], "ok" if ok else "skip", flush=True)

    names = ["pyin_4096", "pesto_notes_from_track", "studio_v1", "studio_v1_aic_k3_oracle"]
    by_singer = {}
    for r in rows:
        by_singer.setdefault(r["singer"], []).append(r)
    singer_mean = {s: {n: float(np.mean([r[n] for r in rs])) for n in names} for s, rs in by_singer.items()}
    out = {
        "schema": "downstream_harmony_impact/0.1",
        "n_windows": len(rows),
        "n_singers": len(singer_mean),
        "chord_agreement_with_A1_chords": {n: [round(v, 4) for v in singer_metric_ci({s: m[n] for s, m in singer_mean.items()})] for n in names},
        "paired_minus_pyin_4096": {
            n: [round(v, 4) for v in bootstrap_ci({s: m[n] - m["pyin_4096"] for s, m in singer_mean.items()})]
            for n in names[1:]
        },
        "not_computed": ["strong_beat_melody_chord_compatibility (needs the downstream compatibility scorer)"],
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_windows", "chord_agreement_with_A1_chords", "paired_minus_pyin_4096")}, indent=2))


if __name__ == "__main__":
    main()
