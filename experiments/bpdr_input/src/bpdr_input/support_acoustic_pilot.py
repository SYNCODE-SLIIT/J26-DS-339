"""Bounded, development-only audit of supports fixed by the geometry diagnostic."""
import numpy as np
from .interventions import AuditPolicy, audit_shift
from .audit_diagnostics import failure_reasons


PHRASES = ('vocadito_2_p00', 'vocadito_5_p00')
SHIFTS = (-30., 30.)
EXACT_CHECKS = ('identity_pass', 'same_sample_count', 'outside_support_exact', 'float_wav_roundtrip_exact')


def pilot_sources(evidence):
    """Original manifest order; the explicitly named pair controls selection."""
    selected=[];sources=set();singers=set()
    for row in evidence['phrases']:
        if row['split'] != 'development' or row['candidate'] is None:
            continue
        if row['source_id'] in sources or row['singer_id'] in singers:
            continue
        if row['evidence_level'] != 'cached_final_reliable_mask_exact_replay' or row['training_eligible']:
            raise ValueError('Expected unchanged development final-mask geometry')
        selected.append(row);sources.add(row['source_id']);singers.add(row['singer_id'])
        if len(selected)==2:break
    if tuple(r['phrase_id'] for r in selected) != PHRASES:
        raise ValueError('Frozen named pilot sources changed; no substitute or fallback')
    for row in selected:
        x=row['candidate'];a,b=x['support_seconds']
        if not 0<=a<b<=row['duration_seconds'] or x['plateau_reference_frames']<10 or x['outside_reference_frames']<10:
            raise ValueError('Invalid frozen support geometry')
    return selected


def audited_candidate(before, after, yin_before, yin_after, reliable, support, cents, exact_checks):
    """Keep original audit, then require physical invariants; refuse failed targets."""
    if cents not in SHIFTS:
        raise ValueError('Only the declared +/-30-cent pilot is allowed')
    if set(exact_checks)!=set(EXACT_CHECKS) or any(type(v) is not bool for v in exact_checks.values()):
        raise ValueError('All physical identity checks must be explicit booleans')
    policy=AuditPolicy()
    audit,anchors,target=audit_shift(before,after,yin_before,yin_after,reliable,support,cents,policy)
    audit['core_pitch_audit_pass']=audit['passed']
    audit.update(exact_checks,requested_cents=cents)
    audit['passed']=bool(audit['passed'] and all(exact_checks.values()))
    audit['failure_reasons']=failure_reasons(audit,policy)
    if not exact_checks['float_wav_roundtrip_exact']:
        audit['failure_reasons'].append('FLOAT_WAV_ROUNDTRIP_EXACT')
    if not audit['passed']:
        anchors=np.zeros_like(anchors);target=np.zeros_like(target)
    return audit,anchors,target
