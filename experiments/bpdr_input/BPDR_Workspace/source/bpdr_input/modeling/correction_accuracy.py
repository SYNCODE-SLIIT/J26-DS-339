"""Reference-controlled correction metrics; never an intended-note classifier."""
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F


ARMS = ('baseline', 'preservation', 'continuity', 'correction_limit', 'combined')
FOLDS = (('female1', 'male1'), ('female2', 'male3'),
         ('female3', 'male4'), ('female4',))


def improvement_loss(predictions, target, *, arm, response_active,
                     neighbour_predictions=None, neighbour_targets=None,
                     neighbour_valid=None, original_predictions=None):
    """Inputs are normalized by 30 cents. Neighbours are actual adjacent centers.

    Original anchors are zero. Unknown ramp labels cannot enter derivative loss.
    This retains the engineering experiment's squared response terms; it is not
    the proposal-aligned MAE-response comparison.
    """
    if arm not in ARMS:
        raise ValueError('Unknown objective arm')
    a, b, c, d = predictions.unbind(-1)
    original = .5*(a.square().mean()+b.square().mean())
    shifted = .5*((c-target).square().mean()+(d-target).square().mean())
    anchor = .5*(original+shifted)
    boundary = .5*((a-b).square().mean()+(c-d).square().mean())
    response = .5*((c-a-target).square().mean()+(d-b-target).square().mean())
    total = anchor + (.25*response+.1*boundary if response_active else 0.)
    zero = predictions.sum()*0.
    preservation = gradient = over_limit = zero
    if arm in ('preservation', 'combined'):
        # Double the original term and penalize C/D changes on known outside.
        preservation = .5*original
        outside = target == 0
        if outside.any():
            preservation = preservation + .5*predictions[outside, 2:].square().mean()
        if original_predictions is None:
            raise ValueError('Preservation requires broader original-only anchors')
        preservation = preservation + .5*original_predictions.square().mean()
        total = total+preservation
    if arm in ('continuity', 'combined'):
        if neighbour_predictions is None or neighbour_targets is None or neighbour_valid is None:
            raise ValueError('Continuity requires aligned labelled neighbours')
        wanted = torch.stack([torch.zeros_like(neighbour_targets)]*2+[neighbour_targets]*2, -1)
        wanted = wanted[:, 1:]-wanted[:, :-1]
        valid = neighbour_valid[:, 1:] & neighbour_valid[:, :-1]
        differences = neighbour_predictions[:, 1:]-neighbour_predictions[:, :-1]
        if valid.any():
            gradient = F.smooth_l1_loss(differences[valid], wanted[valid])
        total = total+.1*gradient
    if arm in ('correction_limit', 'combined'):
        over_limit = torch.relu(predictions.abs()-1.).square().mean()
        total = total+.1*over_limit
    return total, dict(anchor=anchor, original=original, shifted=shifted,
        response=response, boundary=boundary, preservation=preservation,
        gradient=gradient, over_limit=over_limit)


def assert_independent(training_groups, training_sources, rows):
    """Reject reuse of a fitted singer/source, including excluded descendants."""
    groups = {r.get('group', r['source']['singer_id']) for r in rows}
    sources = {r['source']['sha256'] for r in rows}
    if set(training_groups) & groups or set(training_sources) & sources:
        raise ValueError('Evaluation overlaps training singers or source recordings')


def prediction_row(pair, a, c):
    a, c = np.asarray(a), np.asarray(c)
    n = len(pair['target'])
    if a.shape != (3,n) or c.shape != (3,n) or not np.isfinite(a).all() or not np.isfinite(c).all():
        raise ValueError('Expected three finite aligned prediction views')
    plateau = pair['mask'] & (pair['target'] != 0)
    outside = pair['mask'] & (pair['target'] == 0)
    original = pair['A']['reliable']
    if not plateau.any() or not original.any():
        raise ValueError('Scoreable pair has empty reference/original mask')
    error = c[0,plateau]-pair['target'][plateau]
    return dict(group=pair['group'], source_id=pair['source_id'],
        source_sha256=pair['source']['sha256'], cents=float(pair['cents']),
        baseline_mae_cents=float(np.abs(pair['target'][plateau]).mean()),
        raw_mae_cents=float(np.abs(error).mean()),
        raw_rmse_cents=float(np.sqrt(np.mean(error**2))),
        within_5c_percent=float(100*np.mean(np.abs(error)<=5.)),
        within_10c_percent=float(100*np.mean(np.abs(error)<=10.)),
        response_gain=float(np.mean((c[0,plateau]-a[0,plateau])/pair['target'][plateau])),
        original_abs_cents=float(np.abs(a[0,original]).mean()),
        original_p95_cents=float(np.quantile(np.abs(a[0,original]),.95)),
        original_over_3c_percent=float(100*np.mean(np.abs(a[0,original])>3)),
        outside_abs_cents=float(np.abs(c[0,outside]).mean()) if outside.any() else None,
        over_30c_percent=float(100*np.mean(np.abs(c[0,plateau])>30.)),
        boundary_disagreement_cents=float(np.mean(np.abs(c[1:]-c[0]))),
        scored_frames=int(plateau.sum()))


def summarize(rows):
    """Average cases within seed/singer, seeds within singer, then singers.

    Seeds and frames are not additional independent people. Unscoreable cases
    remain in attempt counts; paired repeated originals use the same weighting
    as the prior experiment, explicitly distinct from a unique-source metric.
    """
    valid = [r for r in rows if 'excluded' not in r]
    result = dict(attempted_case_evaluations=len(rows), scoreable_case_evaluations=len(valid),
        excluded_case_evaluations=len(rows)-len(valid),
        unique_attempted_sources=len({r.get('source_sha256', r.get('source_id')) for r in rows
                                      if r.get('source_sha256',r.get('source_id')) is not None}))
    if not valid:
        return dict(summary=result, by_singer={}, by_shift={}, prediction_gate_passed=False)
    metrics = [k for k,v in valid[0].items() if isinstance(v,(int,float)) and k not in ('cents','seed','scored_frames')]
    buckets = defaultdict(list)
    for r in valid:
        buckets[(r.get('seed',0),r['group'])].append(r)
    per_seed_group = {}
    for key, rr in buckets.items():
        per_seed_group[key] = {m:float(np.mean([r[m] for r in rr if r.get(m) is not None]))
            if any(r.get(m) is not None for r in rr) else None for m in metrics}
    by_singer = {}
    for group in sorted({r['group'] for r in rows}):
        members = [v for (s,g),v in per_seed_group.items() if g==group]
        if not members:
            by_singer[group] = dict(scoreable=0)
            continue
        by_singer[group] = {m:float(np.mean([r[m] for r in members if r[m] is not None]))
            if any(r[m] is not None for r in members) else None for m in metrics}
        by_singer[group].update(scoreable=sum(r['group']==group for r in valid),
            original_recordings=len({r['source_sha256'] for r in valid if r['group']==group}))
    represented = [v for v in by_singer.values() if v['scoreable']]
    for m in metrics:
        values = [r[m] for r in represented if r.get(m) is not None]
        result[m] = float(np.mean(values)) if values else None
    result.update(singers=len(represented), original_recordings=len({r['source_sha256'] for r in valid}),
        error_reduction_percent=100*(1-result['raw_mae_cents']/result['baseline_mae_cents']),
        balanced_error_cents=.35*result['raw_mae_cents']+.5*result['original_abs_cents']+.15*(result['outside_abs_cents'] or 0.))
    by_shift = {}
    for shift in sorted({r['cents'] for r in valid}):
        rr = [r for r in valid if r['cents']==shift]
        groups = sorted({r['group'] for r in rr})
        by_shift[str(shift)] = dict(raw_mae_cents=float(np.mean([np.mean([r['raw_mae_cents'] for r in rr if r['group']==g]) for g in groups])),
                                   case_evaluations=len(rr), singers=len(groups))
    gate = all(g['original_abs_cents']<=3. and g['raw_mae_cents']<g['baseline_mae_cents']
               and g['response_gain']>=.25 for g in represented)
    # Descriptive singer bootstrap: only a small development cohort.
    values = np.array([g['raw_mae_cents'] for g in represented])
    rng = np.random.default_rng(339)
    boot = values[rng.integers(len(values),size=(2000,len(values)))].mean(axis=1)
    result['development_singer_bootstrap_95_mae_cents'] = np.quantile(boot,[.025,.975]).tolist()
    return dict(summary=result, by_singer=by_singer, by_shift=by_shift,
                prediction_gate_passed=bool(gate))
