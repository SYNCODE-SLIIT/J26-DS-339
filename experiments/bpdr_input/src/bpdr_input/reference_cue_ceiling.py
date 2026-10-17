"""Gate E R0: reference-cue ceiling, executing runs/reference_cue_ceiling/declaration.json.

TRAINING records only. Order: authenticate declaration -> compute and hash every mixture tuning scalar
(full track and crop window) -> compute YIN fine pitch on existing waveforms -> probes and bootstrap.
No neural fit, no new separation/GAME/crops/shifted audio, no development or holdout data.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .cue_features import _window_sum, HALF_WIDTHS, STEP_WIDTHS
from .interventions import independent_pitch
from .runtime_cue_diagnosis import PREP, base_scalars

COMPONENT = Path(__file__).resolve().parents[2]
OUT = COMPONENT/'runs/reference_cue_ceiling'
DECLARATION_SHA256 = '3b4396a7d406ea9313605cb6cbc696314bf82bb6a13f7f4821466f269e3f0fcb'
MIXTURES = COMPONENT/'data/mtg-jamendo/vocal-tagged-audio'
SR = 22050


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cue(pitch, voiced, offset):
    """Grid-phase cue relative to a given tuning offset (cents); same windows as cue_features but explicit reference."""
    theta = 2*np.pi*(pitch-offset)/100
    z = np.where(voiced, np.exp(1j*theta), 0)
    out = [np.where(voiced, np.sin(theta), 0), np.where(voiced, np.cos(theta), 0)]
    v = voiced.astype(np.float64)
    for w in HALF_WIDTHS:
        count = _window_sum(v, w, w)
        s = _window_sum(z.real, w, w)+1j*_window_sum(z.imag, w, w)
        ok = count > 0
        out += [np.where(voiced & ok, np.angle(s)/np.pi, 0), np.where(voiced & ok, np.abs(s)/np.maximum(count, 1), 0)]
    pv = np.where(voiced, pitch, 0)
    for w in STEP_WIDTHS:
        lc, rc = _window_sum(v, w, -1), _window_sum(v, -1, w)
        lm, rm = _window_sum(pv, w, -1)/np.maximum(lc, 1), _window_sum(pv, -1, w)/np.maximum(rc, 1)
        out.append(np.where((lc > 0) & (rc > 0) & voiced, np.clip((rm-lm)/100, -2, 2), 0))
    return np.column_stack(out)


def own_offset(pitch, voiced):
    if voiced.sum() < 20:
        return 0.
    return float(np.angle(np.mean(np.exp(2j*np.pi*(pitch[voiced] % 100)/100)))*100/(2*np.pi))


def mixture_tuning(record):
    """Full-track and crop-window tuning (cents re A440 grid) of the mixture mp3; one scalar each per track."""
    import librosa
    n = int(record['source_id'].split('_')[1])
    path = MIXTURES/f'{n % 100:02d}'/f'{n}.low.mp3'
    crop = json.loads((Path(record['waveform_paths']['A']).parent/'record.json').read_text())['crop']
    y, _ = librosa.load(path, sr=SR, mono=True)
    full = 100*float(librosa.estimate_tuning(y=y, sr=SR, resolution=.01))
    a, b = int(crop['offset_seconds']*SR), int((crop['offset_seconds']+crop['duration_seconds'])*SR)
    window = 100*float(librosa.estimate_tuning(y=y[a:b], sr=SR, resolution=.01))
    return dict(mp3=str(path), mp3_sha256=sha(path), full_track_cents=full, crop_window_cents=window,
                crop_offset_seconds=crop['offset_seconds'], crop_duration_seconds=crop['duration_seconds'], track_seconds=len(y)/SR)


def yin_on_frames(wav, times):
    import soundfile
    audio, rate = soundfile.read(wav, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    f0 = independent_pitch(audio, rate)
    t = np.arange(len(f0))*441/rate
    idx = np.clip(np.searchsorted(t, times), 0, len(t)-1)
    prev = np.maximum(idx-1, 0)
    idx = np.where(abs(t[prev]-times) < abs(t[idx]-times), prev, idx)
    return 1200*np.log2(np.maximum(f0[idx], 1e-6)/440.)


def load(records, yin_cache):
    out = []
    for r in records:
        d = Path(r['input_directory'])
        with np.load(d/'A_input.npz') as z: a = z['boundary_features_71'].astype(np.float32)
        with np.load(d/'C_input.npz') as z: c = z['boundary_features_71'].astype(np.float32)
        with np.load(r['target_path']) as z:
            mask = z['common_reliable_mask'].astype(bool); det = -z['correction_target_cents'].astype(np.float64); times = z['times_seconds']
        yin = {}
        for view in 'AC':
            key = r['waveform_paths'][view]
            if key not in yin_cache:
                yin_cache[key] = yin_on_frames(key, times)
            if len(yin_cache[key]) != len(times):
                raise ValueError('YIN/feature frame mismatch')
            yin[view] = yin_cache[key]
        out.append(dict(ds=r['dataset'], grp=r['group_id'], src=r['source_id'], inj=float(r['cents']), A=a, C=c, yinA=yin['A'], yinC=yin['C'], S=mask, det=det))
    return out


def view_features(r, view, reference, tuning):
    raw = r[view]
    voiced = (raw[:, 66] > .5) & (raw[:, 64] != 0)
    if reference == 'ref0':
        pitch = raw[:, 64]; offset = own_offset(pitch, voiced)
    else:
        pitch = r['yin'+view]
        offset = own_offset(pitch, voiced) if reference == 'ref1' else tuning[r['src']]['full_track_cents' if reference == 'ref2' else 'crop_window_cents']
    return np.column_stack([base_scalars(raw), cue(pitch, voiced, offset)]), pitch, voiced, offset


def frames(records, reference, tuning, stride=3):
    X, Y, W, R, V = [], [], [], [], []
    for i, r in enumerate(records):
        P = r['S'] & (r['det'] != 0); O = r['S'] & (r['det'] == 0)
        for view in 'AC':
            feat = view_features(r, view, reference, tuning)[0]
            regions = {0: r['S']} if view == 'A' else {1: P, 2: O}
            target = np.zeros(len(feat)) if view == 'A' else -r['det']
            for code, m in regions.items():
                idx = np.flatnonzero(m)
                if len(idx) == 0: continue
                sel = idx[::stride]
                X.append(feat[sel]); Y.append(target[sel]); W.append(np.full(len(sel), {0: .5, 1: .25, 2: .25}[code]/len(idx)*stride))
                R.append(np.full(len(sel), i)); V.append(np.full(len(sel), code))
    return np.vstack(X), np.concatenate(Y), np.concatenate(W), np.concatenate(R), np.concatenate(V)


def probe(records, reference, tuning, seed=339):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    X, Y, W, R, V = frames(records, reference, tuning)
    groups = np.array([records[i]['grp'] for i in R])
    pred = np.zeros(len(Y))
    for tr, te in GroupKFold(5).split(X, Y, groups):
        m = HistGradientBoostingRegressor(loss='absolute_error', max_iter=150, max_leaf_nodes=31, random_state=seed)
        m.fit(X[tr], Y[tr], sample_weight=W[tr]/W[tr].mean())
        pred[te] = m.predict(X[te])
    rows = []
    for i in range(len(records)):
        e = {}
        for code, name in ((0, 'original'), (1, 'correction'), (2, 'outside')):
            k = (R == i) & (V == code)
            e[name] = (float(np.abs(pred[k]-Y[k]).mean()), float(np.abs(Y[k]).mean())) if k.any() else (np.nan, np.nan)
        rows.append(e)
    return rows


def gains(rows, idx):
    cm = np.array([rows[i]['correction'][0] for i in idx]); cz = np.array([rows[i]['correction'][1] for i in idx])
    om = np.array([.5*rows[i]['original'][0]+.25*rows[i]['correction'][0]+.25*rows[i]['outside'][0] for i in idx])
    oz = np.array([.25*rows[i]['correction'][1] for i in idx])
    return 1-cm.mean()/cz.mean(), 1-om.mean()/oz.mean()


def bootstrap(records, rows_by_ref, base='ref0', n=2000, seed=339):
    groups = sorted({r['grp'] for r in records})
    members = {g: [i for i, r in enumerate(records) if r['grp'] == g] for g in groups}
    rng = np.random.default_rng(seed)
    draws = {ref: [] for ref in rows_by_ref}
    diff = {ref: [] for ref in rows_by_ref if ref != base}
    for _ in range(n):
        idx = [i for g in rng.choice(groups, len(groups)) for i in members[g]]
        g = {ref: gains(rows, idx) for ref, rows in rows_by_ref.items()}
        for ref in rows_by_ref:
            draws[ref].append(g[ref])
        for ref in diff:
            diff[ref].append((g[ref][0]-g[base][0], g[ref][1]-g[base][1]))
    q = lambda a: dict(correction=[float(np.quantile(np.array(a)[:, 0], .025)), float(np.quantile(np.array(a)[:, 0], .975))],
                       objective=[float(np.quantile(np.array(a)[:, 1], .025)), float(np.quantile(np.array(a)[:, 1], .975))])
    return dict(gain_intervals={k: q(v) for k, v in draws.items()}, difference_from_ref0_intervals={k: q(v) for k, v in diff.items()},
                resamples=n, unit='artist/singer group')


def scatter(records, reference, tuning):
    """Off-reference deviation of UNSHIFTED original plateau pitch (view A on the region later shifted)."""
    devs = []
    for r in records:
        _, pitch, voiced, offset = view_features(r, 'A', reference, tuning)
        P = r['S'] & (r['det'] != 0) & voiced
        if P.sum() < 3: continue
        devs.append(float(np.median(((pitch[P]-offset+50) % 100)-50)))
    d = np.array(devs)
    return dict(records=len(d), sd_cents=float(d.std()), mad_cents=float(np.median(np.abs(d-np.median(d)))), median_abs_cents=float(np.median(np.abs(d))))


def run():
    if sha(OUT/'declaration.json') != DECLARATION_SHA256:
        raise ValueError('Declaration changed after it was written; refusing to run')
    decl = json.loads((OUT/'declaration.json').read_text())
    catalog = json.loads((PREP/'training_catalog.json').read_text())['records']
    if sha(PREP/'training_catalog.json') != decl['data']['training_catalog_sha256']:
        raise ValueError('Training catalog changed')
    jam = [r for r in catalog if r['dataset'] == 'jamendo']
    started = datetime.now(timezone.utc).isoformat()
    # 1. References first, hashed before any probe.
    tuning = {}
    for r in jam:
        if r['source_id'] not in tuning:
            tuning[r['source_id']] = mixture_tuning(r)
    (OUT/'mixture_tuning.json').write_text(json.dumps(tuning, indent=2), encoding='utf-8')
    tuning_sha = sha(OUT/'mixture_tuning.json')
    print('MIXTURE_TUNING', len(tuning), tuning_sha, flush=True)
    # 2. Fine pitch on existing waveforms.
    cache = {}
    records = load(catalog, cache)
    print('YIN waveforms', len(cache), flush=True)
    jr = [r for r in records if r['ds'] == 'jamendo']; vr = [r for r in records if r['ds'] == 'vocadito']
    ev = dict(declaration_sha256=DECLARATION_SHA256, started_utc=started, mixture_tuning_sha256=tuning_sha, yin_waveforms=len(cache),
              interpreter=sys.executable, source_sha256=sha(__file__),
              mixture_tuning_summary=dict(full_track_sd_cents=float(np.std([t['full_track_cents'] for t in tuning.values()])),
                                          full_minus_crop_median_abs_cents=float(np.median([abs(t['full_track_cents']-t['crop_window_cents']) for t in tuning.values()]))))
    # 3. Probes.
    results = {}
    for ds, recs, refs in (('jamendo', jr, ('ref0', 'ref1', 'ref2', 'ref3')), ('vocadito', vr, ('ref0', 'ref1'))):
        rows = {ref: probe(recs, ref, tuning) for ref in refs}
        all_idx = list(range(len(recs)))
        results[ds] = dict(
            scatter={ref: scatter(recs, ref, tuning) for ref in refs},
            gains={ref: dict(zip(('correction', 'objective'), gains(rows[ref], all_idx))) for ref in refs},
            regions={ref: {name: [float(np.nanmean([x[name][0] for x in rows[ref]])), float(np.nanmean([x[name][1] for x in rows[ref]]))]
                           for name in ('original', 'correction', 'outside')} for ref in refs},
            by_injected={ref: {str(c): dict(zip(('correction', 'objective'), gains(rows[ref], [i for i, r in enumerate(recs) if r['inj'] == c])))
                               for c in sorted({r['inj'] for r in recs})} for ref in refs},
            bootstrap=bootstrap(recs, rows))
        print(ds, json.dumps(results[ds]['gains']), flush=True)
    ev['results'] = results
    t = decl['thresholds']; j = results['jamendo']
    verdict = {}
    for ref in ('ref1', 'ref2', 'ref3'):
        verdict[ref] = dict(correction_gain=j['gains'][ref]['correction'], objective_gain=j['gains'][ref]['objective'],
                            difference_lower_bounds=[j['bootstrap']['difference_from_ref0_intervals'][ref]['correction'][0],
                                                     j['bootstrap']['difference_from_ref0_intervals'][ref]['objective'][0]],
                            passes=bool(j['gains'][ref]['correction'] >= t['jamendo_correction_gain_min']
                                        and j['gains'][ref]['objective'] >= t['jamendo_objective_gain_min']
                                        and j['bootstrap']['difference_from_ref0_intervals'][ref]['correction'][0] > 0))
    ev['decision'] = dict(per_reference=verdict, r0_passes=any(v['passes'] for v in verdict.values()),
                          note='ref3 is a declared sensitivity check; a pass on ref3 alone is reported but does not unlock R1.',
                          unlocks_r1=bool(verdict['ref1']['passes'] or verdict['ref2']['passes']))
    ev['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (OUT/'evidence.json').write_text(json.dumps(ev, indent=2), encoding='utf-8')
    print(json.dumps(ev['decision'], indent=2))
    return ev


if __name__ == '__main__':
    run()
