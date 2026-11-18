"""Align each HumTrans clip to its score-like MIDI (label timing and transposition are not frame-accurate).
Declared before running: subsequence DTW between the sung reliable pitch frames and the MIDI note frames, searching whole-semitone
transpositions -12..+12 (cost = |sung - midi - shift| clipped at 300 c); keep the lowest normalised cost. The reference for each sung frame
is the matched MIDI pitch + shift (an intended-pitch grid). A clip is KEPT only if the median |sung - ref| after per-clip tuning offset removal
is <= 60 c and >= 50% of reliable frames are matched with a monotone path. Reports kept share per role/person. No model fit."""
import json, sys
from pathlib import Path
import numpy as np, librosa
WS = Path(__file__).resolve().parents[1]; A = WS / 'datasets/humtrans_r1/analysis'
HOP = .02
def run(c, ok, notes):
    qi = np.flatnonzero(ok); q = c[qi]
    if len(q) < 30: return None
    mid = []; mt = []; mnote = []
    for k, (on, hz, du, _) in enumerate(notes):
        n = max(1, int(round(du / HOP))); mid += [1200 * np.log2(hz / 440)] * n; mt += list(on + np.arange(n) * HOP); mnote += [k] * n
    mid = np.array(mid); mnote = np.array(mnote); best = None
    for s in range(-12, 13):
        C = np.minimum(np.abs(q[:, None] - (mid[None, :] + 100 * s)), 300.) / 100.
        D, wp = librosa.sequence.dtw(C=C, subseq=True)
        j = int(np.argmin(D[-1])); cost = D[-1, j] / len(q)
        if best is None or cost < best[0]: best = (cost, s, wp[::-1].copy())
    cost, s, wp = best
    ref = np.full(len(c), np.nan); nid = np.full(len(c), -1); path = {}
    for a, b in wp: path.setdefault(a, []).append(b)
    for a, bs in path.items():
        bs = [b for b in bs if 0 <= b < len(mid)]
        if a < len(qi) and bs: ref[qi[a]] = np.median(mid[bs]) + 100 * s; nid[qi[a]] = int(np.median(mnote[bs]))
    return ref, cost, s, nid
rows = []; man = json.load(open(WS / 'datasets/humtrans_r1/manifest_analysis.json'))
for k, s in enumerate(man):
    f = A / f"{s['key']}.npz"
    if 'sha256' not in s or not f.exists(): continue
    out = A / f"{s['key']}.ref.npz"
    z = np.load(f); c = z['fine'].astype(float); ok = z['reliable'].astype(bool) & np.isfinite(c)
    notes = json.load(open(A / f"{s['key']}.notes.json")); r = run(c, ok, notes) if notes else None
    if r is None: rows.append(dict(key=s['key'], person=s['person'], role=s['role'], kept=False)); continue
    ref, cost, sh, nid = r; m = ok & ~np.isnan(ref); d = (c - ref)[m]
    medabs = float(np.abs(d - np.median(d)).mean() if len(d) else 1e9); frac = float(m.sum() / max(ok.sum(), 1))
    kept = bool(medabs <= 60 and frac >= .5); np.savez(out, ref=ref, kept=kept, shift=sh, nid=nid)
    rows.append(dict(key=s['key'], person=s['person'], role=s['role'], kept=kept, mean_abs_after_offset=medabs, matched=frac, shift=sh, cost=float(cost)))
    if k % 50 == 0: print(k, len(man), flush=True)
json.dump(rows, open(WS / 'results/studio/r1/humtrans_alignment.json', 'w'), indent=1)
import collections
tot = collections.Counter((r['role'], r['person']) for r in rows); kept = collections.Counter((r['role'], r['person']) for r in rows if r['kept'])
print({f'{a}/{b}': f'{kept[(a, b)]}/{tot[(a, b)]}' for a, b in sorted(tot)})
