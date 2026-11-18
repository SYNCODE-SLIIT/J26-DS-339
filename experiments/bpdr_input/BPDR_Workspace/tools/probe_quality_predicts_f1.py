"""Plan 2026-11-13 s4 item 2. DECLARED BEFORE RUN: do quality_report measures predict per-file
extractor F1 (pYIN 2048/C2, onset+pitch @50ms/50c, mean of A1/A2) on the 63 development cases
(21 training-role sources x clean/10dB/0dB)? A measure is KEPT only if pooled Spearman |rho|>=0.30 and
its singer-level bootstrap 95% interval (2000 resamples, seed 339) excludes 0. Development evidence only."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source")); sys.path.insert(0, str(ROOT / "tools"))
import librosa, numpy as np
from scipy.stats import spearmanr
import studio_evidence as SE
from studio_frame_control import scores
from bpdr_input.studio import evidence as E

noise, _ = librosa.load(SE.NOISE, sr=E.SR)
M = ["level_db", "noise_floor_db", "snr_estimate_db", "voiced_ratio", "mean_confidence_voiced", "octave_jumps", "clipped_fraction"]
rows = []
for i in SE.TRAIN_ROLE_IDS:
    x0 = SE.load8(i)
    for cond, x in (("clean", x0), ("10dB", SE.mix(x0, noise, 10)), ("0dB", SE.mix(x0, noise, 0))):
        trk = E.pyin_track(x, frame_length=2048, fmin_note="C2")
        f1 = float(scores(i, E.notes_from_track(trk, len(x) / E.SR))[0])
        q = E.quality_report(x, trk)
        rows.append(dict(id=i, singer=SE.SINGER_OF[i], cond=cond, f1=f1, **{m: q[m] for m in M}))
f1 = np.array([r["f1"] for r in rows]); sing = np.array([r["singer"] for r in rows])
us = sorted(set(sing)); rng = np.random.default_rng(339); out = {}
for m in M:
    v = np.array([r[m] for r in rows], float)
    if np.ptp(v) == 0: out[m] = dict(rho=None, note="constant"); continue
    rho = spearmanr(v, f1)[0]; bs = []
    for _ in range(2000):
        pick = rng.choice(us, len(us)); idx = np.concatenate([np.where(sing == s)[0] for s in pick])
        if np.ptp(v[idx]) and np.ptp(f1[idx]): bs.append(spearmanr(v[idx], f1[idx])[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    within = {c: float(spearmanr([r[m] for r in rows if r["cond"] == c], [r["f1"] for r in rows if r["cond"] == c])[0]) for c in ("clean", "10dB", "0dB")}
    out[m] = dict(rho=float(rho), ci=[float(lo), float(hi)], within_condition=within,
                  keep=bool(abs(rho) >= .30 and (lo > 0 or hi < 0)))
    print(m, out[m], flush=True)
json.dump(dict(n=len(rows), singers=len(us), mean_f1=float(f1.mean()), measures=out, rows=rows),
          open(ROOT / "results/studio/evidence/quality_predicts_f1.json", "w"), indent=1)
