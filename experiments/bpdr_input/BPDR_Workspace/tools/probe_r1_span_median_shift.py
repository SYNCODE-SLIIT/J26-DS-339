"""R1 go/no-go (plan 2026-11-13 s3.1): how far does the plain span-median deviation move
when span boundaries are falsely split, merged or jittered? No training, read-only.
Rule declared before running: singer-macro mean over the three kinds of the mean absolute
change (cents) >= 5.0 -> GO, else NO-GO."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
sys.path.insert(0, str(Path(__file__).parent.parent / 'source'))
from bpdr_input.repair.interventions import corrupt_boundaries

WS = Path(__file__).parent.parent
DATA = WS / 'results/transformer_usability/2026-11-02_v40_short_phrase_data'
KINDS = ('split', 'merge', 'jitter'); RULE = 5.0

def dev(c): return (c + 50) % 100 - 50          # cents from nearest semitone (A440 grid)

def spans(edges, n_t, t):
    b = np.r_[t[0], edges, t[-1] + 1e-6]
    return np.digitize(t, b[1:-1])               # span id per frame

def medians(cents, ok, ids):
    out = np.full(len(cents), np.nan)
    for s in np.unique(ids):
        m = (ids == s) & ok
        if m.sum() >= 3: out[ids == s] = np.median(cents[m])
    return out

rows = []
for name in ('train_originals', 'train_pairs'):
    for i, r in enumerate(json.load(open(DATA / f'{name}.json'))):
        if 'excluded' in r: continue
        path = r['analysis_A'] if name == 'train_originals' else r['analysis_C']
        z = np.load(path); raw = z['raw_views'][0]
        t = np.arange(len(raw)) * .02
        f0 = z['f0_hz']; ok = z['voiced_mask'].astype(bool) & (f0 > 0)
        if len(f0) != len(raw): continue
        cents = np.where(ok, 1200 * np.log2(np.maximum(f0, 1e-3) / 440.), 0.)
        pk, _ = find_peaks(raw[:, 70], height=.5, distance=5); edges = t[pk]
        base = medians(cents, ok, spans(edges, len(t), t))
        rec = dict(group=r['group'], id=r['source_id'] if 'source_id' in r else str(i))
        for k in KINDS:
            try: e2 = corrupt_boundaries(edges, t, k, seed=339)
            except ValueError: rec[k] = None; continue
            alt = medians(cents, ok, spans(e2, len(t), t))
            m = ok & ~np.isnan(base) & ~np.isnan(alt)
            if m.sum() == 0: rec[k] = None; continue
            rec[k] = [float(np.mean(np.abs(alt - base)[m])),
                      float(np.mean(np.abs(dev(alt) - dev(base))[m]))]
        rows.append(rec)
print('records', len(rows))
out = {}
for j, lab in enumerate(('centre_shift_c', 'semitone_deviation_shift_c')):
    per = {}
    for k in KINDS:
        by = {}
        for r in rows:
            if r[k] is not None: by.setdefault(r['group'], []).append(r[k][j])
        per[k] = float(np.mean([np.mean(v) for v in by.values()]))
        per[k + '_n'] = sum(len(v) for v in by.values())
    per['macro_mean'] = float(np.mean([per[k] for k in KINDS])); out[lab] = per
    print(lab, per)
print('groups', len({r['group'] for r in rows}))
verdict = 'GO' if out['semitone_deviation_shift_c']['macro_mean'] >= RULE else 'NO-GO'
print('verdict (semitone-deviation metric, rule 5.0 c):', verdict)
out['verdict'] = verdict
(WS / 'results/studio/evidence').mkdir(parents=True, exist_ok=True)
json.dump(out, open(WS / 'results/studio/evidence/r1_span_median_shift.json', 'w'), indent=1)
