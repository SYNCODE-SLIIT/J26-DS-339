"""Boundary-target headroom (no training). DECLARED BEFORE RUN.
Question: how well do the extractor's own boundaries (column-70 peaks, as in the boundary audit) and its onset score (column-69 peaks,
height 0.5, min gap 100 ms) match true note onsets? Onset F1 at 50 ms and 100 ms, one-to-one nearest matching, person-macro.
True onsets: HumTrans = first matched frame of each aligned MIDI note (kept clips only; label timing is DTW-derived, so 50 ms is a hard bar);
Vocadito = A1 annotation onsets inside the 10 s window (accurate labels). Also reports each clip's oracle ceiling if boundaries were
the true onsets (1.0 by construction) and how labels compare with a naive pitch-change onset rule, to show the headroom."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

WS = Path(__file__).resolve().parents[1]
A = WS / 'datasets/humtrans_r1/analysis'
V = WS / 'datasets/vocadito/Annotations/Notes'
DATA = WS / 'results/transformer_usability/2026-11-04_v64_reliable_melodic_pairs'


def f1(ref, est, tol):
    ref, est = np.sort(ref), np.sort(est)
    if len(ref) == 0 or len(est) == 0:
        return 0.0
    used, hit = set(), 0
    for r in ref:
        j = [k for k in range(len(est)) if k not in used and abs(est[k] - r) <= tol]
        if j:
            k = min(j, key=lambda k: abs(est[k] - r)); used.add(k); hit += 1
    p, rc = hit / len(est), hit / len(ref)
    return 0.0 if hit == 0 else 2 * p * rc / (p + rc)


def peaks(x, t, h=.5):
    pk, _ = find_peaks(x, height=h, distance=5)
    return t[pk]


def pitch_change_onsets(c, ok, t, jump=70.):
    out, last = [], None
    for i in range(1, len(c)):
        if ok[i] and ok[i - 1] and abs(c[i] - c[i - 1]) >= jump:
            out.append(t[i])
    return np.array(out)


rows = []
man = json.load(open(WS / 'datasets/humtrans_r1/manifest_analysis.json'))
for s in man:
    f = A / f"{s['key']}.ref.npz"
    if 'sha256' not in s or not f.exists():
        continue
    rf = np.load(f)
    if not bool(rf['kept']):
        continue
    z = np.load(A / f"{s['key']}.npz"); raw = z['raw_views'][0]; t = np.arange(len(raw)) * .02
    nid = rf['nid']; true = np.array([t[np.flatnonzero(nid == k)[0]] for k in np.unique(nid[nid >= 0])])
    if len(true) < 3:
        continue
    lo, hi = true.min() - .05, true.max() + .5
    seg = lambda x: x[(x >= lo) & (x <= hi)]
    rows.append(dict(set='humtrans_' + s['role'], person=s['person'], n_true=len(true),
                     **{f'{name}_{int(tol * 1000)}': f1(true, seg(est), tol)
                        for name, est in (('boundary', peaks(raw[:, 70], t)), ('onsetscore', peaks(raw[:, 69], t)))
                        for tol in (.05, .10)}))
for r in json.load(open(DATA / 'train_originals.json')):
    if 'excluded' in r or not r['source_id'].startswith('vocadito'):
        continue
    f = V / f"{r['parent_source_id']}_notesA1.csv"
    if not f.exists():
        continue
    a = np.loadtxt(f, delimiter=',', ndmin=2); z = np.load(r['analysis_A']); raw = z['raw_views'][0]; t = np.arange(len(raw)) * .02
    T = t + r['crop_offset_seconds']; true = a[:, 0][(a[:, 0] >= T[0]) & (a[:, 0] < T[-1])] - r['crop_offset_seconds']
    if len(true) < 3:
        continue
    rows.append(dict(set='vocadito_train_role', person=r['group'], n_true=len(true),
                     **{f'{name}_{int(tol * 1000)}': f1(true, est, tol)
                        for name, est in (('boundary', peaks(raw[:, 70], t)), ('onsetscore', peaks(raw[:, 69], t)))
                        for tol in (.05, .10)}))
out = {}
for st in sorted({r['set'] for r in rows}):
    sel = [r for r in rows if r['set'] == st]; by = {}
    for r in sel:
        by.setdefault(r['person'], []).append(r)
    keys = [k for k in sel[0] if k not in ('set', 'person', 'n_true')]
    out[st] = {k: round(float(np.mean([np.mean([r[k] for r in v]) for v in by.values()])), 3) for k in keys}
    out[st]['clips'] = len(sel); out[st]['persons'] = len(by); out[st]['mean_true_onsets'] = round(float(np.mean([r['n_true'] for r in sel])), 1)
    print(st, out[st])
json.dump(out, open(WS / 'results/studio/r1/boundary_baseline.json', 'w'), indent=1)
