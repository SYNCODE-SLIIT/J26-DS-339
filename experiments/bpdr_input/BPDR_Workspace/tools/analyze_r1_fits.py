import json, numpy as np
from pathlib import Path
WS = Path(__file__).resolve().parents[1]
d = json.load(open(WS / 'results/studio/r1/fit_results.json'))['rows']
ARMS = ('frame_only', 'augmentation_only', 'bpdr')
print('fits', len({(r['fold'], r['seed'], r['arm']) for r in d}), 'healthy share',
      round(np.mean([h for h in {(r['fold'], r['seed'], r['arm']): r['healthy'] for r in d}.values()]), 3))
def gm(rows, key):
    by = {}
    for r in rows:
        if key in r: by.setdefault(r['group'], []).append(r[key])
    return {g: np.mean(v) for g, v in by.items()}
acc = {a: gm([r for r in d if r['arm'] == a], 'mae') for a in ARMS}
for a in ARMS:
    print(a, 'MAE', round(np.mean(list(acc[a].values())), 2), {k: round(np.mean(list(gm([r for r in d if r['arm'] == a], 'chg_' + k).values())), 2) for k in ('split', 'merge', 'jitter')})
# baseline: rolling median 0.2 s on the same test windows is in r1_accuracy_headroom.json (33 records, group-macro 17.85)
rng = np.random.default_rng(339)
gs = sorted(acc['bpdr'])
def ci(a, b):
    diff = np.array([acc[a][g] - acc[b][g] for g in gs]); bs = [diff[rng.integers(0, len(gs), len(gs))].mean() for _ in range(5000)]
    return round(diff.mean(), 2), [round(x, 2) for x in np.percentile(bs, [2.5, 97.5])]
print('bpdr - frame_only', ci('bpdr', 'frame_only')); print('bpdr - aug_only', ci('bpdr', 'augmentation_only'))
h = json.load(open(WS / 'results/studio/evidence/r1_accuracy_headroom.json'))['rows']
base = {}
for r in h: base.setdefault(r['group'], []).append(r['roll10'])
print('rolling-median baseline (headroom file)', round(np.mean([np.mean(v) for v in base.values()]), 2))
