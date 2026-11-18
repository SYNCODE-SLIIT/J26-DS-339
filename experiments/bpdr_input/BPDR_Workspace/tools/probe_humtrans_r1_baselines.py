"""HumTrans R1 data check (no training). (1) Label alignment: best global time offset of MIDI notes vs sung pitch per clip.
(2) Baselines for the note-centre task, per role: frame pitch, plain extractor-boundary median, rolling 0.2 s median, oracle true-boundary median.
Per-clip median offset removed (global tuning is not a lever). Person-macro means. Declared before running."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
WS = Path(__file__).resolve().parents[1]; A = WS / 'datasets/humtrans_r1/analysis'
def spanmed(c, ok, ids):
    o = np.full(len(c), np.nan)
    for s in np.unique(ids):
        m = (ids == s) & ok
        if m.sum() >= 3: o[ids == s] = np.median(c[m])
    return o
def rolling(c, ok, w=10):
    o = np.full(len(c), np.nan)
    for i in range(len(c)):
        m = ok[max(0, i - w // 2): i + w // 2 + 1]; v = c[max(0, i - w // 2): i + w // 2 + 1][m]
        if m.sum() >= 3: o[i] = np.median(v)
    return o
def refgrid(notes, T):
    ref = np.full(len(T), np.nan); nid = np.full(len(T), -1)
    for k, (on, hz, du, _) in enumerate(notes):
        m = (T >= on) & (T < on + du); ref[m] = 1200 * np.log2(hz / 440); nid[m] = k
    return ref, nid
man = json.load(open(WS / 'datasets/humtrans_r1/manifest_analysis.json')); rows = []; offs = []
for s in man:
    f = A / f"{s['key']}.npz"
    if 'sha256' not in s or not f.exists(): continue
    z = np.load(f); raw = z['raw_views'][0]; t = np.arange(len(raw)) * .02; c = z['fine'].astype(float)
    ok0 = z['reliable'].astype(bool) & np.isfinite(c); notes = json.load(open(A / f"{s['key']}.notes.json"))
    if ok0.sum() < 30 or not notes: continue
    best = None
    for sh in np.arange(-.6, .61, .04):
        ref, _ = refgrid(notes, t - sh); m = ok0 & ~np.isnan(ref)
        if m.sum() < 30: continue
        d = (c - ref)[m]; e = float(np.abs(d - np.median(d)).mean())
        if best is None or e < best[0]: best = (e, sh)
    if best is None: continue
    offs.append(best[1]); ref, nid = refgrid(notes, t); ok = ok0 & ~np.isnan(ref)
    if ok.sum() < 30: continue
    c = c - np.median((c - ref)[ok]); pk, _ = find_peaks(raw[:, 70], height=.5, distance=5)
    e = dict(frame=np.abs(c - ref), ext=np.abs(spanmed(c, ok, np.digitize(t, t[pk])) - ref), roll10=np.abs(rolling(c, ok) - ref), true=np.abs(spanmed(c, ok, nid) - ref))
    m = ok & ~np.isnan(e['ext']) & ~np.isnan(e['roll10']) & ~np.isnan(e['true'])
    if m.sum() < 30: continue
    rows.append(dict(person=s['person'], role=s['role'], **{k: float(v[m].mean()) for k, v in e.items()}))
print('clips', len(rows), 'alignment: median best offset', round(float(np.median(offs)), 2), 's; share within +-0.12 s', round(float(np.mean(np.abs(offs) <= .12)), 2))
out = {'alignment_median_offset_s': float(np.median(offs)), 'alignment_share_within_0.12': float(np.mean(np.abs(offs) <= .12)), 'clips': len(rows)}
for role in ('train', 'development'):
    sel = [r for r in rows if r['role'] == role]; by = {}
    for r in sel: by.setdefault(r['person'], []).append(r)
    out[role] = {k: float(np.mean([np.mean([r[k] for r in v]) for v in by.values()])) for k in ('frame', 'ext', 'roll10', 'true')}; out[role]['persons'] = len(by); out[role]['clips'] = len(sel)
    print(role, {k: round(v, 2) if isinstance(v, float) else v for k, v in out[role].items()})
json.dump(out, open(WS / 'results/studio/r1/humtrans_baselines.json', 'w'), indent=1)
