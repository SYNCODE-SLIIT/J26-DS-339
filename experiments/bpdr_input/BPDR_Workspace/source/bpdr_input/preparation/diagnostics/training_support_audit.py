"""Frozen training-only inventory for the predeclared support acoustic audit."""
from copy import deepcopy

SHIFTS=(-30.,30.)
TERMINAL={'measured','execution_refused','not_attempted_geometry_refused','not_attempted_identity_or_source_refused'}


def audit_inventory(completion,plan,manifest):
    rows=completion['rows'];lookup={r['phrase_id']:r for r in rows}
    train={p['phrase_id']:p for p in manifest['phrases'] if p['split']=='train'}
    if completion['status']!='complete' or len(rows)!=30 or len(lookup)!=30 or set(lookup)!=set(train):
        raise ValueError('Expected the complete thirty-phrase frozen training inventory')
    if plan['status']!='declared_not_executed' or plan['partition']!='train' or plan['shifts_cents']!=list(SHIFTS):
        raise ValueError('Only the declared training +/-30-cent protocol is allowed')
    candidates=sorted((r for r in rows if r['geometry_available']),key=lambda r:r['phrase_id'])
    if len(candidates)!=23:raise ValueError('Frozen twenty-three-candidate inventory changed')
    expected=[(r['phrase_id'],c) for r in candidates for c in SHIFTS]
    if [(c['phrase_id'],c['cents']) for c in plan['cases']]!=expected:
        raise ValueError('Missing, duplicated, reordered or undeclared shift key')
    for r in rows:
        p=train[r['phrase_id']]
        if r['split']!='train' or r['status']!='prepared' or r['training_eligible'] is not False or any(r[k]!=p[k] for k in ['source_id','singer_id']):
            raise ValueError('Prepared original training provenance changed')
    for c in plan['cases']:
        r=lookup[c['phrase_id']]
        if (c['support_seconds']!=r['multi_interior_candidate']['support_seconds']
                or any(c[k]!=r[k] for k in ['arrays_directory','selected_cache','source_id','singer_id'])):
            raise ValueError('Frozen support or original-cache provenance changed')
    attempts=[dict(phrase_id=r['phrase_id'],cents=c,passed=False,
        status='not_started' if r['geometry_available'] else 'not_attempted_geometry_refused',
        reason=None if r['geometry_available'] else 'No support under unchanged final reference masks and geometry rules')
        for r in rows for c in SHIFTS]
    return deepcopy(candidates),attempts


def summarize(attempts,sources):
    keys={(s['phrase_id'],c) for s in sources for c in SHIFTS}
    if len(sources)!=30 or len(attempts)!=len(keys) or {(a['phrase_id'],a['cents']) for a in attempts}!=keys:
        raise ValueError('Incomplete, duplicate or foreign audit outcome')
    if any(a['status'] not in TERMINAL or (a['passed'] and a['status']!='measured') for a in attempts):
        raise ValueError('Unfinished or improperly promoted audit outcome')
    passed=[a for a in attempts if a['passed']];ids={a['phrase_id'] for a in passed}
    both={s['phrase_id'] for s in sources if all(a['passed'] for a in attempts if a['phrase_id']==s['phrase_id'])}
    return dict(training_phrases=30,all_source_shift_keys=len(keys),declared_candidate_shift_keys=46,
        measured_shifts=sum(a['status']=='measured' for a in attempts),passed_shifts=len(passed),
        failed_measured_shifts=sum(a['status']=='measured' and not a['passed'] for a in attempts),
        geometry_refused_keys=sum(a['status']=='not_attempted_geometry_refused' for a in attempts),
        identity_or_source_refused_keys=sum(a['status']=='not_attempted_identity_or_source_refused' for a in attempts),
        execution_refused_keys=sum(a['status']=='execution_refused' for a in attempts),
        source_identity_passes=sum(s.get('identity_pass',False) for s in sources),
        phrases_with_any_pass=len(ids),recordings_with_any_pass=len({s['source_id'] for s in sources if s['phrase_id'] in ids}),
        singers_with_any_pass=len({s['singer_id'] for s in sources if s['phrase_id'] in ids}),
        phrases_with_both_directions_pass=len(both),recordings_with_both_directions_pass=len({s['source_id'] for s in sources if s['phrase_id'] in both}),
        singers_with_both_directions_pass=len({s['singer_id'] for s in sources if s['phrase_id'] in both}),
        negative_passes=sum(a['cents']<0 for a in passed),positive_passes=sum(a['cents']>0 for a in passed),
        new_pairs=0,new_optimizer_updates=0,research_ready_examples=0)
