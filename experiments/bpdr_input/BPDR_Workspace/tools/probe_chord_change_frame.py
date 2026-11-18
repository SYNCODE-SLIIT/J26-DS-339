"""Plan 2026-11-13 s4 item 3. DECLARED BEFORE RUN: on the 21 training-role Vocadito phrases (clean, first 8 s)
do chords from Pasan->Rahul (read-only, via studio.chords.harmonize) change when the notes come from
pYIN 2048/C2 instead of today's 4096/C1? Same bpm=100, genre=pop, tonic = most frequent pitch class of
the A1 reference notes in ALL arms, so only the notes differ. Chord label sampled every 0.25 s over the
16-bar window (tiled phrase). Reported: (a) % of samples whose label differs between arms; (b) agreement
of each arm with chords from reference notes (A1) -- 'oracle'; singer-level 95% bootstrap on the
(2048 minus 4096) agreement difference. Lead with the chord finding only if that interval excludes 0."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source")); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
import studio_evidence as SE
from bpdr_input.studio import evidence as E, chords as C

BPM = 100
def labels(notes, tonic):
    if not notes: return None
    h = C.harmonize(notes, BPM, tonic)
    t = np.arange(0, h["duration_s"], .25)
    out = []
    for x in t:
        l = next((c["label"] for c in h["chords"] if c["start_s"] <= x < c["end_s"]), "N")
        out.append(l)
    return np.array(out)

rows = []
for i in SE.TRAIN_ROLE_IDS:
    x = SE.load8(i); dur = len(x) / E.SR
    ri, rh = SE.ref(i, "A1")
    refn = [dict(onset_s=float(a), duration_s=float(b - a), pitch_midi=int(round(69 + 12 * np.log2(h / 440)))) for (a, b), h in zip(ri, rh)]
    tonic = int(np.bincount([n["pitch_midi"] % 12 for n in refn], minlength=12).argmax())
    arms = {"today_4096": E.notes_from_track(E.pyin_track(x, frame_length=4096, fmin_note="C1"), dur),
            "pyin_2048": E.notes_from_track(E.pyin_track(x, frame_length=2048, fmin_note="C2"), dur), "oracle": refn}
    L = {k: labels(v, tonic) for k, v in arms.items()}
    if any(v is None for v in L.values()): print("skip", i); continue
    rows.append(dict(id=i, singer=SE.SINGER_OF[i],
                     differ=float(np.mean(L["today_4096"] != L["pyin_2048"])),
                     agree_today=float(np.mean(L["today_4096"] == L["oracle"])),
                     agree_2048=float(np.mean(L["pyin_2048"] == L["oracle"]))))
    print(rows[-1], flush=True)
d = np.array([r["agree_2048"] - r["agree_today"] for r in rows]); sing = np.array([r["singer"] for r in rows]); us = sorted(set(sing))
rng = np.random.default_rng(339); bs = []
for _ in range(2000):
    idx = np.concatenate([np.where(sing == s)[0] for s in rng.choice(us, len(us))]); bs.append(d[idx].mean())
res = dict(n=len(rows), singers=len(us), mean_differ=float(np.mean([r["differ"] for r in rows])),
           mean_agree_today=float(np.mean([r["agree_today"] for r in rows])), mean_agree_2048=float(np.mean([r["agree_2048"] for r in rows])),
           diff_agree=float(d.mean()), diff_ci=[float(v) for v in np.percentile(bs, [2.5, 97.5])], rows=rows)
print({k: v for k, v in res.items() if k != "rows"})
json.dump(res, open(ROOT / "results/studio/evidence/chord_change_frame.json", "w"), indent=1)
