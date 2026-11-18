"""Apply the declared bar (results/studio/r1/declaration_boundary.json) to boundary_fit_results.json and print a plain verdict."""
import json, sys
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]; R = WS / 'results/studio/r1'
f = R / 'boundary_fit_results.json'
d = json.load(open(f if f.exists() else R / 'boundary_fit_partial.json')); rows = d['rows']
print('final results' if f.exists() else 'PARTIAL results', '| fits:', len({(r['fold'], r['seed'], r['arm']) for r in rows}),
      '| healthy share:', round(np.mean(list({(r['fold'], r['seed'], r['arm']): r['healthy'] for r in rows}.values())), 3))
ARMS = ('frame_only', 'augmentation_only', 'bpdr'); KINDS = ('clean', 'split', 'merge', 'jitter'); rng = np.random.default_rng(339)


def per_clip(split, key):
    out = {}
    for a in ARMS:
        m = {}
        for r in rows:
            if r['split'] == split and r['arm'] == a: m.setdefault((r['person'], r['id']), []).append(r[key])
        out[a] = {k: float(np.mean(v)) for k, v in m.items()}
    return out


def macro(clips):
    by = {}
    for (p, _), v in clips.items(): by.setdefault(p, []).append(v)
    return float(np.mean([np.mean(v) for v in by.values()])), len(by)


def ci(diffs):  # diffs: {(person,id): value}; resample persons
    by = {}
    for (p, _), v in diffs.items(): by.setdefault(p, []).append(v)
    ps = list(by); stat = lambda sel: np.mean([np.mean(by[p]) for p in sel])
    bs = [stat([ps[i] for i in rng.integers(0, len(ps), len(ps))]) for _ in range(4000)]
    return float(stat(ps)), [float(x) for x in np.percentile(bs, [2.5, 97.5])]


verdict = {}
for split in ('heldout', 'dev_M02', 'vocadito'):
    if not any(r['split'] == split for r in rows): print(f'\n[{split}] not available yet'); continue
    print(f'\n=== {split} (onset F1 at 50 ms, person-macro) ===')
    tab = {}
    ext = per_clip(split, 'ext_clean')['bpdr']
    for k in KINDS:
        e = macro(per_clip(split, 'ext_' + k)['bpdr'])[0]
        tab[k] = {'extractor': e, **{a: macro(per_clip(split, 'f1_' + k)[a])[0] for a in ARMS}}
    print(f"{'input boundary track':<22}{'extractor':>10}{'frame_only':>12}{'aug_only':>10}{'bpdr':>8}")
    for k in KINDS: print(f"{k:<22}{tab[k]['extractor']:>10.3f}{tab[k]['frame_only']:>12.3f}{tab[k]['augmentation_only']:>10.3f}{tab[k]['bpdr']:>8.3f}")
    diff = {c: per_clip(split, 'f1_clean')['bpdr'][c] - ext[c] for c in ext}
    m, (lo, hi) = ci(diff); print(f'bpdr minus extractor, clean: {m:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]  ({macro(ext)[1]} persons)')
    drop = lambda a: np.mean([tab['clean'][a] - tab[k][a] for k in KINDS[1:]])
    print('mean drop under corruption:', {a: round(drop(a), 3) for a in ('extractor',) + ARMS})
    verdict[split] = dict(tab=tab, diff=(m, lo, hi), drop={a: float(drop(a)) for a in ('extractor',) + ARMS})
if 'vocadito' in verdict:
    v = verdict['vocadito']; t = v['tab']; m, lo, hi = v['diff']
    c1 = lo > 0; c2 = all(t[k]['bpdr'] >= t[k]['extractor'] for k in KINDS[1:])
    c3 = t['clean']['bpdr'] >= max(t['clean']['frame_only'], t['clean']['augmentation_only']) - .01
    c4 = v['drop']['bpdr'] <= .7 * v['drop']['augmentation_only']
    h = verdict.get('heldout'); c5 = (h is None) or h['diff'][0] > 0
    print('\n=== DECLARED BAR (decisive: Vocadito, independent labels) ===')
    for n, ok, txt in ((1, c1, 'bpdr beats the extractor, clean, interval excludes 0'), (2, c2, 'bpdr >= extractor under split, merge and jitter'),
                       (3, c3, 'bpdr within 0.01 of the best other arm or better'), (4, c4, 'bpdr drop <= 70% of augmentation_only drop'),
                       (5, c5, 'HumTrans held-out does not reverse the sign')): print(f"  ({n}) {'PASS' if ok else 'FAIL'}  {txt}")
    print('\nVERDICT:', 'ACCURACY IMPROVED (all five criteria met)' if all((c1, c2, c3, c4, c5)) else
          ('IMPROVED over the extractor but not all criteria met' if c1 else 'NOT IMPROVED on the independent test'))
