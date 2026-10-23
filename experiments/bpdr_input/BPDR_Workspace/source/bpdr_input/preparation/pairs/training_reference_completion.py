"""Constrain original-only completion to previously unavailable training phrases."""
from copy import deepcopy


def completion_phrases(manifest, prior_rows):
    training = [p for p in manifest['phrases'] if p['split'] == 'train']
    lookup = {p['phrase_id']: p for p in training}
    rows = {r['phrase_id']: r for r in prior_rows}
    if len(lookup) != len(training) or len(rows) != len(prior_rows) or set(lookup) != set(rows):
        raise ValueError('Completion inventory must exactly match frozen training phrases')
    selected = []
    for pid in sorted(lookup):
        p, r = lookup[pid], rows[pid]
        if any(p[k] != r[k] for k in ('split', 'source_id', 'singer_id')):
            raise ValueError('Training provenance changed')
        if r['status'] == 'unavailable':
            if r.get('geometry_available') is not None or r.get('reliable_frames') is not None:
                raise ValueError('Unavailable measurements must remain unavailable')
            selected.append(deepcopy(p))
        elif r['status'] != 'prepared':
            raise ValueError('Prior refusal or unknown state cannot be silently retried')
    return selected


def merge_completion(prior_rows, additions):
    """Retain every prepared row verbatim; no silent retry, insertion or omission."""
    prior = {r['phrase_id']: r for r in prior_rows}
    extra = {r['phrase_id']: r for r in additions}
    missing = {pid for pid, r in prior.items() if r['status'] == 'unavailable'}
    if len(prior) != len(prior_rows) or len(extra) != len(additions) or set(extra) != missing:
        raise ValueError('Only all previously unavailable phrases may be completed')
    for pid, r in extra.items():
        if any(r[k] != prior[pid][k] for k in ('source_id', 'singer_id', 'split')) or r['split'] != 'train':
            raise ValueError('Completion crossed a source or partition boundary')
        if r.get('training_eligible') is not False or r['status'] not in ('prepared', 'technical_refusal'):
            raise ValueError('Original preparation cannot authorize training or omit refusals')
        if r['status'] == 'technical_refusal' and (r.get('geometry_available') is not None or r.get('reliable_frames') is not None):
            raise ValueError('Technical refusal has no final acoustic measurements')
    return [deepcopy(extra.get(r['phrase_id'], r)) for r in prior_rows]
