"""Short probe: does a global tuning offset break the downstream pYIN note extractor,
and does blind re-centering fix it? f0 is shifted analytically (exactly what a
physical pitch shift does to the f0 track); pYIN is run once per file."""
import sys, os, json, glob
S = os.path.dirname(os.path.abspath(__file__))  # tools/
sys.path.insert(0, os.path.join(os.path.dirname(S), "results/transformer_usability/evaluation_dependencies/mir_eval_0_8_2"))
ROOT = r"C:\Users\lnipu\Projects\Music Rp\RP\J26-DS-339"
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, mir_eval
from melody_understanding.pipeline import analyze_vocal
from melody_understanding.features import segment_notes

V = os.path.join(ROOT, r"experiments\bpdr_input\BPDR_Workspace\datasets\vocadito")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
ids = [1,3,6,8,10,13,14,18,20,25,26,27,28,29,31,32,35,36,37,38,39][:N]
cache = os.path.join(S, "f0cache.json")
C = json.load(open(cache)) if os.path.exists(cache) else {}

def f0_for(i):
    k = str(i)
    if k not in C:
        r, _f = analyze_vocal(os.path.join(V, "Audio", f"vocadito_{i}.wav"))
        C[k] = {"t": r["pitch"]["times_s"], "f0": r["pitch"]["f0_hz"], "v": r["pitch"]["voiced"],
                "p": r["pitch"]["voiced_probability"], "dur": r["source_duration_s"]}
        json.dump(C, open(cache, "w"))
    d = C[k]
    return (np.array(d["t"]), np.array([np.nan if x is None else x for x in d["f0"]]),
            np.array(d["v"], bool), np.array(d["p"]), d["dur"])

def ref(i):
    a = np.loadtxt(os.path.join(V, "Annotations", "Notes", f"vocadito_{i}_notesA1.csv"), delimiter=",", ndmin=2)
    on, hz, du = a[:, 0], a[:, 1], a[:, 2]
    return np.c_[on, on + du], hz

def cents_center(f0, v, p):
    m = v & np.isfinite(f0) & (f0 > 0) & (p > 0.5)
    if m.sum() < 20: return 0.0
    c = 100 * (69 + 12 * np.log2(f0[m] / 440) % 1 * 0)  # placeholder, replaced below
    midi = 69 + 12 * np.log2(f0[m] / 440)
    res = 100 * (midi - np.rint(midi))              # -50..50
    ang = 2 * np.pi * res / 100
    return float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * 100 / (2 * np.pi))

def score(i, shift_c, mode):
    t, f0, v, p, dur = f0_for(i)
    f = f0 * 2 ** (shift_c / 1200)
    est_off = 0.0
    if mode == "blind":
        est_off = cents_center(f, v, p)
        f = f * 2 ** (-est_off / 1200)
    feats = {"f0_hz": f, "voiced": v, "voiced_probability": p, "times_s": t}
    notes = segment_notes(feats, dur)
    ri, rh = ref(i)
    rh = rh * 2 ** ((shift_c - est_off) / 1200)  # truth moves with the applied shift
    if not notes: return 0, 0, 0, len(rh), est_off, 0
    ei = np.array([[n["onset_s"], n["onset_s"] + n["duration_s"]] for n in notes])
    eh = np.array([440 * 2 ** ((n["pitch_midi"] - 69) / 12) for n in notes])
    _, _, f_pitch, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rh, ei, eh, offset_ratio=None)
    _, _, f_onset, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rh, ei, eh, offset_ratio=None, pitch_tolerance=1e9)
    best = 0
    for k in (-1, 0, 1):
        _, _, fk, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rh, ei, eh * 2 ** (k / 12), offset_ratio=None)
        best = max(best, fk)
    return f_pitch, f_onset, len(notes), len(rh), est_off, best

rows = {}
for i in ids:
    for sh in (0, 25, 50):
        for mode in ("raw", "blind"):
            rows.setdefault((sh, mode), []).append(score(i, sh, mode))
print(f"files={len(ids)}  (note F1 vs annotator A1; onset+pitch within 50c/50ms, and onset-only)")
print("shift_c  mode   F1_onset+pitch  F1_onset_only  F1_transp-inv  est_notes/ref_notes  mean|est offset|")
for (sh, mode), r in sorted(rows.items()):
    a = np.array(r)
    print(f"{sh:5d}   {mode:5s}  {a[:,0].mean():.3f}          {a[:,1].mean():.3f}         {a[:,5].mean():.3f}         {a[:,2].sum()/a[:,3].sum():.2f}                {np.abs(a[:,4]).mean():.1f}")
# per-file natural tuning centers (shift 0, blind estimate)
print("natural per-file tuning centers (cents):", [round(rows[(0,'blind')][k][4]) for k in range(len(ids))])
