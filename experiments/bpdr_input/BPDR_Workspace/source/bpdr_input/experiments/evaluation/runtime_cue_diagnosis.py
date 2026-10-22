"""Gate E step 1: runtime-cue, conflicting-target and gradient-flow diagnosis on TRAINING records only.

No model is fitted for adoption and no development/holdout data is read. The declaration (thresholds, inputs,
source hash) is written before any analysis. Probe learners are tiny tree/neighbour models used only to ask
whether the target is predictable from runtime-available cues.
"""
from bpdr_input import COMPONENT_ROOT, module_source
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid
import numpy as np
from bpdr_input.experiments.evaluation.cue_features import cue_features, NAMES, VERSION as CUE_VERSION

COMPONENT = COMPONENT_ROOT
PREP = COMPONENT/'runs/expanded_experiment_preparation/eeddb03700ad4ddc989dd73b3b113616'
THRESHOLDS = dict(
    plateau_gain_overall_min=.10, plateau_gain_jamendo_min=.10, objective_gain_min=.05,
    note='Gains are relative reductions of artist-grouped cross-validated example-mean absolute error versus zero output. '
         'Proceed to the declared input adapter only if all three hold; otherwise stop and reframe.')
SETTINGS = dict(folds=5, frame_stride=3, seed=339, learner='HistGradientBoostingRegressor(absolute_error, 150 iter, 31 leaves)',
                weights='per-example region weights 0.5 original / 0.25 correction / 0.25 outside, as in plateau-balanced-objective-0.1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_records():
    records = json.loads((PREP/'training_catalog.json').read_text())['records']
    out = []
    for r in records:
        d = Path(r['input_directory'])
        with np.load(d/'A_input.npz') as z: a = z['boundary_features_71'].astype(np.float32)
        with np.load(d/'C_input.npz') as z: c = z['boundary_features_71'].astype(np.float32)
        with np.load(r['target_path']) as z:
            mask = z['common_reliable_mask'].astype(bool); det = -z['correction_target_cents'].astype(np.float32)
        out.append(dict(ds=r['dataset'], grp=r['group_id'], inj=float(r['cents']), A=a, C=c, S=mask, det=det))
    return out


def base_scalars(raw):
    """Features the model already has as scalars (no tuning cue): standardized-scale pitch, confidence, voicing, change, energy, onset."""
    return np.column_stack([raw[:, 64]/715., raw[:, 65], raw[:, 66], raw[:, 67]/100., raw[:, 68]/20., raw[:, 69]])


def frames(records, use_cue, stride):
    """Stack subsampled frames of the reliable mask: features, target, weight, record id, view, region."""
    X, Y, W, R, V, G = [], [], [], [], [], []
    for i, r in enumerate(records):
        P = r['S'] & (r['det'] != 0); O = r['S'] & (r['det'] == 0)
        for view, raw in (('A', r['A']), ('C', r['C'])):
            feat = base_scalars(raw)
            if use_cue: feat = np.column_stack([feat, cue_features(raw)])
            if view == 'A': regions = {0: r['S']}; target = np.zeros(len(raw), np.float32)
            else: regions = {1: P, 2: O}; target = -r['det']
            for code, m in regions.items():
                idx = np.flatnonzero(m)
                n = len(idx)
                if n == 0: continue
                weight = {0: .5, 1: .25, 2: .25}[code]/n*stride   # example-level region weight
                sel = idx[::stride]
                X.append(feat[sel]); Y.append(target[sel]); W.append(np.full(len(sel), weight)); R.append(np.full(len(sel), i)); V.append(np.full(len(sel), code))
    return np.vstack(X), np.concatenate(Y), np.concatenate(W), np.concatenate(R), np.concatenate(V)


def cross_validate(records, use_cue):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    X, Y, W, R, V = frames(records, use_cue, SETTINGS['frame_stride'])
    groups = np.array([records[i]['grp'] for i in R])
    pred = np.zeros(len(Y))
    for tr, te in GroupKFold(SETTINGS['folds']).split(X, Y, groups):
        model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=150, max_leaf_nodes=31, random_state=SETTINGS['seed'])
        model.fit(X[tr], Y[tr], sample_weight=W[tr]/W[tr].mean())
        pred[te] = model.predict(X[te])
    return Y, pred, R, V


def summarize(records, Y, pred, R, V):
    """Example-mean absolute errors per region (A=0, correction=1, outside=2), overall and by dataset, against zero output."""
    table = {}
    for name, subset in (('all', None), ('jamendo', 'jamendo'), ('vocadito', 'vocadito')):
        row = {}
        for code, label in ((0, 'original'), (1, 'correction'), (2, 'outside')):
            model_err, zero_err = [], []
            for i in np.unique(R[V == code]):
                if subset and records[i]['ds'] != subset: continue
                m = (R == i) & (V == code)
                model_err.append(np.abs(pred[m]-Y[m]).mean()); zero_err.append(np.abs(Y[m]).mean())
            row[label] = dict(model=float(np.mean(model_err)), zero=float(np.mean(zero_err)), examples=len(model_err))
        row['objective'] = dict(model=.5*row['original']['model']+.25*row['correction']['model']+.25*row['outside']['model'],
                                zero=.5*row['original']['zero']+.25*row['correction']['zero']+.25*row['outside']['zero'])
        row['correction_gain'] = 1-row['correction']['model']/row['correction']['zero']
        row['objective_gain'] = 1-row['objective']['model']/row['objective']['zero']
        table[name] = row
    return table


def pitch_column_facts(records):
    pc = np.concatenate([np.concatenate([r['A'][:, 64], r['C'][:, 64]]) for r in records])
    equal = {}
    for inj in sorted({r['inj'] for r in records}):
        fr = []
        for r in records:
            if r['inj'] != inj: continue
            P = r['S'] & (r['det'] != 0) & (r['A'][:, 66] > .5) & (r['C'][:, 66] > .5)
            if P.any(): fr.append(float((r['A'][P, 64] == r['C'][P, 64]).mean()))
        equal[str(inj)] = float(np.mean(fr))
    return dict(pitch_scale_cents=float(pc.std()), thirty_cents_in_sd=30/float(pc.std()), ten_cents_in_sd=10/float(pc.std()),
                plateau_frames_with_identical_pitch_column_by_injected_cents=equal)


def conflicting_targets(records):
    """For each correction example, find the nearest OTHER-group correction/original examples in cue space and compare targets."""
    from sklearn.neighbors import NearestNeighbors
    X, Y, _, R, V = frames(records, True, 6)
    keep = V == 1
    own = np.where(keep)[0]
    Xc = X[:, 6:]
    pool = np.where((V == 1) | (V == 0))[0]
    sd = Xc[pool].std(axis=0); sd[sd == 0] = 1
    groups = np.array([records[i]['grp'] for i in R])
    nn = NearestNeighbors(n_neighbors=21).fit(Xc[pool]/sd)
    rng = np.random.default_rng(SETTINGS['seed']); sample = rng.choice(own, size=min(4000, len(own)), replace=False)
    _, nb = nn.kneighbors(Xc[sample]/sd)
    spread, opposite, share_original = [], [], []
    for row, j in zip(nb, sample):
        cand = pool[row]; cand = cand[groups[cand] != groups[j]][:20]
        if len(cand) == 0: continue
        t = Y[cand]
        spread.append(float(np.median(np.abs(t-np.median(t)))))
        opposite.append(float(np.mean(np.sign(t) == -np.sign(Y[j])) if Y[j] != 0 else np.nan))
        share_original.append(float(np.mean(V[cand] == 0)))
    return dict(sampled_correction_frames=len(spread),
                median_neighbour_target_spread_cents=float(np.median(spread)),
                mean_fraction_of_neighbours_with_opposite_sign_target=float(np.nanmean(opposite)),
                mean_fraction_of_neighbours_that_are_unshifted_originals=float(np.mean(share_original)),
                reading='Neighbours are the 20 nearest other-artist frames in cue space. Original (target 0) neighbours and opposite-sign neighbours both pull the best constant answer to 0.')


def declare(out):
    d = dict(declared_utc=datetime.now(timezone.utc).isoformat(), version='runtime-cue-diagnosis-0.1',
             purpose='Decide, before any new fit, whether the correction target is predictable from runtime-only cues and where gradients conflict.',
             data='995 frozen TRAINING records of the expanded catalog; development, PP1 and holdout data are not read',
             training_catalog_sha256=sha(PREP/'training_catalog.json'), cue_features_version=CUE_VERSION, cue_names=NAMES,
             thresholds=THRESHOLDS, settings=SETTINGS,
             source_sha256={n: sha(module_source(n+'.py')) for n in ('runtime_cue_diagnosis', 'cue_features')},
             controls='Same learner on base scalar features only (no tuning cue) as the no-cue control.',
             forbidden='No target, private mask, paired original or development data enters any cue feature; no shared-pipeline execution; no model adoption.',
             interpreter=sys.executable)
    (out/'declaration.json').write_text(json.dumps(d, indent=2), encoding='utf-8')
    return d


def run():
    out = COMPONENT/'runs/runtime_cue_diagnosis'/uuid.uuid4().hex
    out.mkdir(parents=True)
    declaration = declare(out)
    print('DIAGNOSIS_RUN:', out, flush=True)
    records = load_records()
    evidence = dict(declaration_sha256=sha(out/'declaration.json'), pitch_column=pitch_column_facts(records))
    for use_cue in (False, True):
        name = 'with_cue' if use_cue else 'no_cue_control'
        Y, pred, R, V = cross_validate(records, use_cue)
        evidence[name] = summarize(records, Y, pred, R, V)
        print(name, {k: round(v['correction_gain'], 4) for k, v in evidence[name].items()}, {k: round(v['objective_gain'], 4) for k, v in evidence[name].items()}, flush=True)
    evidence['conflicting_targets'] = conflicting_targets(records)
    c = evidence['with_cue']; t = THRESHOLDS
    evidence['decision'] = dict(
        correction_gain_overall=c['all']['correction_gain'], correction_gain_jamendo=c['jamendo']['correction_gain'],
        objective_gain=c['all']['objective_gain'],
        passes=bool(c['all']['correction_gain'] >= t['plateau_gain_overall_min'] and c['jamendo']['correction_gain'] >= t['plateau_gain_jamendo_min']
                    and c['all']['objective_gain'] >= t['objective_gain_min']))
    (out/'evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence['decision'], indent=2))
    return out, evidence


if __name__ == '__main__':
    run()
