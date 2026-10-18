"""Candidate-only packaging of frozen passed training audits; no pool adoption."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from bpdr_input.experiments.learning.comparison_preparation import verify_record
from bpdr_input.preparation.pairs.support_development_packaging import adapt_private_targets, CORRUPTIONS
from bpdr_input.audio.pipeline import checksum


def select_cases(audit,manifest,legacy):
    if audit['status']!='complete' or any(audit[k] for k in ['new_pairs','new_optimizer_updates','research_ready_examples']):
        raise ValueError('A completed exploratory acoustic audit is required')
    train={p['phrase_id']:p for p in manifest['phrases'] if p['split']=='train'}
    attempts=audit['attempts'];expected={(pid,c) for pid in train for c in [-30.,30.]}
    if len(train)!=30 or len(attempts)!=60 or {(a['phrase_id'],a['cents']) for a in attempts}!=expected:
        raise ValueError('Retain all sixty frozen training audit keys')
    if any(r['split']!='train' for r in legacy):raise ValueError('Legacy catalog crossed partitions')
    other=[p for p in manifest['phrases'] if p['split']!='train']
    other_sources={p['source_id'] for p in other};other_singers={p['singer_id'] for p in other}
    sources={s['phrase_id']:s for s in audit['sources']}
    selected=[];excluded=[]
    for a in attempts:
        p=train[a['phrase_id']]
        if a['status'] not in {'measured','not_attempted_geometry_refused','not_attempted_identity_or_source_refused','execution_refused'}:
            raise ValueError('Unfinished audit key')
        if a['passed']:
            s=sources[p['phrase_id']]
            if a['status']!='measured' or not s['identity_pass']:
                raise ValueError('Blocked case cannot become a packaged training example')
            if s['split']!='train' or any(s[k]!=p[k] for k in ['source_id','singer_id']):
                raise ValueError('Passed source provenance changed')
            if p['source_id'] in other_sources or p['singer_id'] in other_singers:
                raise ValueError('Training source or singer overlaps a reserved partition')
            selected.append(dict(phrase=deepcopy(p),attempt=deepcopy(a),source=deepcopy(s)))
        else:excluded.append(deepcopy(a))
    if len(selected)!=28 or len(excluded)!=32:raise ValueError('Frozen passed/excluded case counts changed')
    return selected,excluded


def candidate_metadata(r):
    if (r['dataset']!='vocadito' or r['split']!='train' or not r['audit_passed'] or not r['exploratory']
            or r['training_eligible'] is not False or r['review_status']!='pending' or r['data_status']!='exploratory_not_research'
            or r['reference_type']!='human_annotation' or r['group_kind']!='annotated_singer_id'
            or r['group_id']!='vocadito:'+r['singer_id'] or r['boundary_encoding']!='gaussian_max_sigma_0.04_seconds'
            or r['support_protocol']!='multi-interior-geometry-0.1' or r['corruption'] not in CORRUPTIONS
            or r['adoption_status']!='candidate_not_adopted'):
        raise ValueError('Expected pending-review human-reference candidate training metadata')


def validate_candidate_record(r):
    candidate_metadata(r);verify_record(r)
    if checksum(r['source_audit_targets_path'])!=r['source_audit_targets_sha256']:
        raise ValueError('Source audit targets changed')
    with np.load(r['source_audit_targets_path'],allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    expected=adapt_private_targets(raw,r['cents'])
    with np.load(r['target_path'],allow_pickle=False) as z:
        if set(z.files)!=set(expected) or any(not np.array_equal(z[k],expected[k]) for k in expected):
            raise ValueError('Private target adaptation changed saved audit anchors or values')
    manifest=json.loads(Path(r['manifest_path']).read_text())
    phrase=next(p for p in manifest['phrases'] if p['phrase_id']==r['phrase_id'])
    if any(phrase[k]!=r[k] for k in ['source_id','singer_id','split']) or phrase['sha256']!=r['crop_sha256']:
        raise ValueError('Candidate does not match frozen training crop')
    return True
