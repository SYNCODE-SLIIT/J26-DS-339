"""R1 go/no-go v2 on the real 498 records (v64 data: 53 originals + 445 pairs), with non-learned baselines.
Declared before running: metric = mean absolute change (cents) of the per-frame span-median pitch-centre under the saved split/merge/jitter
boundary variants (corrupt_boundaries, seed 339, edges = column-70 peaks as in the boundary audit); singer(group)-macro mean per kind.
Baselines: plain median; median trimmed +-60 ms around each edge; confidence-weighted median. Also the same with deviation-from-semitone.
If the best baseline's jitter sensitivity is <= 2 c, no model can show a stability benefit: STOP. Skipped records are counted."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
WS = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(WS / 'source')]
from bpdr_input.repair.interventions import corrupt_boundaries
DATA = WS / 'results/transformer_usability/2026-11-04_v64_reliable_melodic_pairs'
KINDS = ('split', 'merge', 'jitter'); TRIM = .06
def dev(c): return (c + 50) % 100 - 50
def ids_of(edges, t): return np.digitize(t, edges)
def wmedian(v, w):
    o = np.argsort(v); cw = np.cumsum(w[o]); return v[o][np.searchsorted(cw, cw[-1] / 2)]
def centre(cents, ok, conf, ids, t, edges, mode):
    out = np.full(len(cents), np.nan)
    for s in np.unique(ids):
        sel = ids == s; m = sel & ok
        if mode == 'trimmed' and len(edges):
            near = np.min(np.abs(t[:, None] - edges[None, :]), axis=1) < TRIM
            m = m & ~near
            if m.sum() < 3: m = sel & ok          # fall back rather than drop short spans
        if m.sum() < 3: continue
        out[sel] = wmedian(cents[m], np.maximum(conf[m], 1e-3)) if mode == 'weighted' else np.median(cents[m])
    return out
rows, skipped = [], []
for name in ('train_originals', 'train_pairs'):
    for i, r in enumerate(json.load(open(DATA / f'{name}.json'))):
        if 'excluded' in r: continue
        path = r['analysis_A'] if name == 'train_originals' else r['analysis_C']
        z = np.load(path); raw = z['raw_views'][0]; t = np.arange(len(raw)) * .02
        if len(z['fine']) != len(raw): skipped.append(str(path)); continue
        ok = z['reliable'].astype(bool) & np.isfinite(z['fine']); conf = raw[:, 65]
        cents = np.where(ok, z['fine'], 0.)
        pk, _ = find_peaks(raw[:, 70], height=.5, distance=5); edges = t[pk]
        rec = dict(group=r['group'])
        for mode in ('plain', 'trimmed', 'weighted'):
            base = centre(cents, ok, conf, ids_of(edges, t), t, edges, mode); rec[mode] = {}
            for k in KINDS:
                try: e2 = corrupt_boundaries(edges, t, k, seed=339)
                except ValueError: continue
                alt = centre(cents, ok, conf, ids_of(e2, t), t, e2, mode)
                m = ok & ~np.isnan(base) & ~np.isnan(alt)
                if m.sum(): rec[mode][k] = [float(np.mean(np.abs(alt - base)[m])), float(np.mean(np.abs(dev(alt) - dev(base))[m]))]
        rows.append(rec)
print('records', len(rows), 'skipped', len(skipped), 'groups', len({r['group'] for r in rows}))
out = {'records': len(rows), 'skipped': len(skipped)}
for mode in ('plain', 'trimmed', 'weighted'):
    for j, lab in enumerate(('centre', 'semitone_dev')):
        per = {}
        for k in KINDS:
            by = {}
            for r in rows:
                if k in r[mode]: by.setdefault(r['group'], []).append(r[mode][k][j])
            per[k] = float(np.mean([np.mean(v) for v in by.values()]))
        per['macro'] = float(np.mean([per[k] for k in KINDS])); out[f'{mode}|{lab}'] = per
        print(mode, lab, {a: round(b, 2) for a, b in per.items()})
best = min(out[f'{m}|semitone_dev']['jitter'] for m in ('plain', 'trimmed', 'weighted'))
out['best_baseline_jitter_semitone_dev'] = best; out['verdict'] = 'STOP: no stability headroom' if best <= 2.0 else 'headroom exists'
print(out['verdict'], round(best, 2))
json.dump(out, open(WS / 'results/studio/evidence/r1_baselines_498.json', 'w'), indent=1)
