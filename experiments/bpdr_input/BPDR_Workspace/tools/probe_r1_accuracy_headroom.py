"""R1 accuracy headroom (no training). DECLARED BEFORE RUN: on the training-role Vocadito original windows with annotations
(A1 note annotations, crop offset applied), for reliable frames, per-record median offset removed (global tuning is not a lever):
mean absolute error in cents of (a) frame pitch itself, (b) span median using the EXTRACTOR boundaries (column-70 peaks),
(c) span median using the TRUE annotated note boundaries. Headroom for a boundary-robust estimator = (b) minus (c): the most it
could recover from false boundaries. Group-macro means."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
WS = Path(__file__).resolve().parents[1]
DATA = WS / 'results/transformer_usability/2026-11-04_v64_reliable_melodic_pairs'
V = WS / 'datasets/vocadito/Annotations/Notes'
def spanmed(c, ok, ids):
    o = np.full(len(c), np.nan)
    for s in np.unique(ids):
        m = (ids == s) & ok
        if m.sum() >= 3: o[ids == s] = np.median(c[m])
    return o
def jump_ids(c, ok, jump, min_frames=4):
    ids = np.zeros(len(c), int); k = 0; run = []
    last = None
    for i in range(len(c)):
        if not ok[i]: continue
        if last is not None and abs(c[i] - np.median(run[-5:])) > jump and len(run) >= min_frames: k += 1; run = []
        run.append(c[i]); ids[i] = k; last = i
        ids[i] = k
    return ids
def rolling(c, ok, w):
    o = np.full(len(c), np.nan); h = w // 2
    for i in range(len(c)):
        m = ok[max(0, i - h): i + h + 1]; v = c[max(0, i - h): i + h + 1][m]
        if m.sum() >= 3: o[i] = np.median(v)
    return o
rows = []
for r in json.load(open(DATA / 'train_originals.json')):
    if 'excluded' in r or not r['source_id'].startswith('vocadito'): continue
    f = V / f"{r['parent_source_id']}_notesA1.csv"
    if not f.exists(): continue
    a = np.loadtxt(f, delimiter=',', ndmin=2); z = np.load(r['analysis_A']); raw = z['raw_views'][0]
    t = np.arange(len(raw)) * .02; T = t + r['crop_offset_seconds']
    ok = z['reliable'].astype(bool) & np.isfinite(z['fine']); c = z['fine']
    ref = np.full(len(t), np.nan); nid = np.full(len(t), -1)
    for k, (on, hz, du) in enumerate(a[:, :3]):
        m = (T >= on) & (T < on + du); ref[m] = 1200 * np.log2(hz / 440); nid[m] = k
    ok = ok & ~np.isnan(ref)
    if ok.sum() < 10: continue
    c = c - np.median((c - ref)[ok])
    pk, _ = find_peaks(raw[:, 70], height=.5, distance=5); ext = np.digitize(t, t[pk])
    e = dict(frame=np.abs(c - ref), ext=np.abs(spanmed(c, ok, ext) - ref), true=np.abs(spanmed(c, ok, nid) - ref),
             jump60=np.abs(spanmed(c, ok, jump_ids(c, ok, 60)) - ref), jump100=np.abs(spanmed(c, ok, jump_ids(c, ok, 100)) - ref),
             roll10=np.abs(rolling(c, ok, 10) - ref), roll20=np.abs(rolling(c, ok, 20) - ref))
    m = ok & ~np.isnan(e['ext']) & ~np.isnan(e['true']) & ~np.isnan(e['jump60']) & ~np.isnan(e['roll20'])
    if m.sum() < 10: continue
    rows.append(dict(group=r['group'], n=int(m.sum()), **{k: float(v[m].mean()) for k, v in e.items()}))
print('records', len(rows), 'groups', len({x['group'] for x in rows}))
out = {}
for k in ('frame', 'ext', 'jump60', 'jump100', 'roll10', 'roll20', 'true'):
    by = {}
    for x in rows: by.setdefault(x['group'], []).append(x[k])
    out[k] = float(np.mean([np.mean(v) for v in by.values()]))
out['headroom_ext_minus_true'] = out['ext'] - out['true']
print({k: round(v, 2) for k, v in out.items()})
json.dump(dict(summary=out, rows=rows), open(WS / 'results/studio/evidence/r1_accuracy_headroom.json', 'w'), indent=1)
