"""Development-only adapter for previously audited multi-interior Vocadito cases."""
from copy import deepcopy
import numpy as np
from bpdr_input.preparation.pairs.development_preparation import validate_development_record


CORRUPTIONS=('split','merge','jitter')


def packaging_selection(audit, manifest, training):
    if audit['status']!='complete' or audit['new_pairs'] or audit['new_optimizer_updates'] or audit['research_ready_examples']:
        raise ValueError('Expected complete exploratory audit before packaging')
    dev={p['phrase_id']:p for p in manifest['phrases'] if p['split']=='development'}
    expected={(pid,c) for pid in dev for c in (-30.,30.)}
    attempts=audit['attempts']
    if len(dev)!=14 or len(attempts)!=28 or {(a['phrase_id'],a['cents']) for a in attempts}!=expected:
        raise ValueError('Keep the full frozen development audit inventory')
    if any(a['status'] not in {'measured','not_attempted_geometry_refused'} or
           (a['passed'] and a['status']!='measured') for a in attempts):
        raise ValueError('Unfinished or improperly promoted audit attempt')
    if any(r['split']!='train' for r in training):
        raise ValueError('Training catalog has an unexpected partition')
    train_sources={p['source_id'] for p in manifest['phrases'] if p['split']=='train'}
    train_singers={p['singer_id'] for p in manifest['phrases'] if p['split']=='train'}
    train_sources|={r['source_id'] for r in training if r['dataset']=='vocadito'}
    train_singers|={r['singer_id'] for r in training if r['dataset']=='vocadito'}
    selected=[];exclusions=[]
    for a in attempts:
        p=dev[a['phrase_id']]
        if p['source_id'] in train_sources or p['singer_id'] in train_singers:
            raise ValueError('Development recording/singer overlaps frozen training')
        if a['passed']:
            selected.append(dict(attempt=deepcopy(a),phrase=deepcopy(p)))
        else:
            exclusions.append(deepcopy(a))
    if len(selected)!=7 or len(exclusions)!=21:
        raise ValueError('Expected seven passing cases and all twenty-one excluded keys')
    return selected,exclusions


def adapt_private_targets(raw, cents):
    if cents not in (-30.,30.):raise ValueError('Undeclared shift')
    mask=np.asarray(raw['anchor_mask']);target=np.asarray(raw['correction_target_cents']);times=np.asarray(raw['times_seconds'])
    if (mask.dtype!=bool or mask.ndim!=1 or len(mask)==0 or target.shape!=mask.shape or times.shape!=mask.shape
            or not np.isfinite(target).all() or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or np.any(target[~mask]!=0)):
        raise ValueError('Invalid private audit targets')
    plateau=mask&(target!=0);outside=mask&(target==0)
    if plateau.sum()<10 or outside.sum()<10 or np.any(target[plateau]!=-cents):
        raise ValueError('Missing passed plateau/outside anchors or changed target sign')
    return dict(times_seconds=times.copy(),anchor_mask=mask.copy(),common_reliable_mask=mask.copy(),
        detuned_plateau_mask=plateau,zero_anchor_mask=outside,correction_target_cents=target.copy())


def development_metadata(record):
    if (record['dataset']!='vocadito' or record['split']!='development' or record['training_eligible'] is not False
            or not record['audit_passed'] or not record['exploratory'] or record['review_status']!='pending'
            or record['data_status']!='exploratory_not_research' or record['reference_type']!='human_annotation'
            or record['group_kind']!='annotated_singer_id' or record['group_id']!='vocadito:'+record['singer_id']
            or record['boundary_encoding']!='gaussian_max_sigma_0.04_seconds'
            or record['support_protocol']!='multi-interior-geometry-0.1' or record['corruption'] not in CORRUPTIONS):
        raise ValueError('Expected exploratory human-reference development metadata')


def validate_packaged_record(record):
    development_metadata(record)
    return validate_development_record(record)
