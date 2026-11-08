"""Subset review of already permitted local edits, with final WAV re-audit."""

import copy
import json
from pathlib import Path
import uuid

import numpy as np
import soundfile as sf

from bpdr_input.audio.contracts import AcceptedAudio, ProcessingResult, Status
from bpdr_input.audio.pipeline import checksum, validate_package
from bpdr_input.repair.demo_runtime import eligible_preview
from bpdr_input.repair.primary_handoff import WORKSPACE
from bpdr_input.repair.repair_controller import decision_hash, waveform_hash


def _mask(times, intervals):
    mask = np.zeros(len(times), dtype=bool)
    for start, end in intervals:
        mask |= (times >= start) & (times < end)
    return mask


def _audit_and_assess(path, original, rate, times, curve, reliable, intervals, intent):
    from bpdr_input import COMPONENT_ROOT
    from bpdr_input.audio.analysis import extract
    from bpdr_input.repair.demo_review import _assessment
    from bpdr_input.repair.framewise_audit import audit_curve
    from bpdr_input.repair.learned_runtime import audit_delivered_wav
    from bpdr_input.repair.rule_correction import fine_hz
    before = extract(original, rate, COMPONENT_ROOT / 'checkpoints')[0]
    np.testing.assert_array_equal(before['times_seconds'], times)
    after_features = {}

    def auditor(source, candidate, sr, support):
        after = extract(candidate, sr, COMPONENT_ROOT / 'checkpoints')[0]
        after_features.update(after)
        return audit_curve(before, after, fine_hz(source, sr, times), fine_hz(candidate, sr, times),
                           reliable, support, curve)

    audit = audit_delivered_wav(path, original, rate, intervals, auditor)
    return audit, _assessment(path, after_features, intent)


def select_intervals(directory, report, review, selected):
    """Accept a subset of one verified local preview, or reject all its edits.

    Any failure raises without mutating the previous review or waveform.
    Whole-clip denoising and arbitrary user-supplied edit spans are excluded.
    """
    directory = Path(directory).resolve()
    if not directory.is_relative_to(WORKSPACE):
        raise ValueError('Local review must remain inside BPDR_Workspace.')
    preview = eligible_preview(report)
    plan = report.get('permission')
    if report.get('noise_candidate') or not preview or not plan or not plan.get('permitted') or plan.get('identity'):
        raise ValueError('Local selection requires an already permitted local pitch preview.')
    if plan.get('decision_sha256') != decision_hash(plan):
        raise ValueError('The original permission decision changed.')
    accepted = report['handoff']['accepted_audio']
    audio = AcceptedAudio(**{k: accepted[k] for k in ['path', 'checksum', 'sample_rate_hz', 'sample_count', 'channels']})
    validate_package(ProcessingResult(report['handoff']['request_id'], report['handoff']['source_id'],
                                     Status(report['handoff']['status']), audio))
    original, rate = sf.read(audio.path, dtype='float32')
    if plan['source_sha256'] != waveform_hash(original):
        raise ValueError('Local permission does not belong to the current source.')
    if checksum(preview['path']) != preview['sha256']:
        raise ValueError('Preview checksum changed.')
    if review.get('original', {}).get('waveform_sha256') != audio.checksum or review.get('preview', {}).get('waveform_sha256') != preview['sha256']:
        raise ValueError('Review does not belong to the selected source and preview.')
    candidate, candidate_rate = sf.read(preview['path'], dtype='float32')
    if candidate_rate != rate or candidate.shape != original.shape or not np.isfinite(candidate).all():
        raise ValueError('Local preview has a different sample grid.')
    intervals = plan['intervals_seconds']
    if not intervals or any(not np.isfinite([a, b]).all() or not 0 <= a < b <= len(original) / rate for a, b in intervals):
        raise ValueError('Invalid permitted intervals.')
    if any(intervals[i][1] > intervals[i + 1][0] for i in range(len(intervals) - 1)):
        raise ValueError('Permitted intervals overlap or are not ordered.')
    if any(type(i) is not int or not 0 <= i < len(intervals) for i in selected) or len(set(selected)) != len(selected):
        raise ValueError('Select each existing edit index at most once.')
    sample_times = np.arange(len(original)) / rate
    if not np.array_equal(candidate[~_mask(sample_times, intervals)], original[~_mask(sample_times, intervals)]):
        raise ValueError('Existing preview changes samples outside its permission.')
    chosen = [intervals[i] for i in sorted(selected)]
    updated, assessment = copy.deepcopy(report), copy.deepcopy(review)
    provenance = dict(parent_preview_sha256=preview['sha256'], parent_decision_sha256=plan['decision_sha256'],
                      parent_intervals_seconds=intervals,
                      selected_edit_indices=sorted(selected), rejected_edit_indices=sorted(set(range(len(intervals))) - set(selected)),
                      intervals_seconds=chosen, final_decoded_audit_required=bool(chosen), musical_approval=False)
    if not chosen:
        withheld = copy.deepcopy(plan)
        withheld.update(permitted=False, intervals_seconds=[], affected_fraction=0., reasons=['USER_REJECTED_EDITS'])
        withheld['decision_sha256'] = decision_hash(withheld)
        updated.update(outcome='USER_REJECTED_EDITS_UNCHANGED', local_selection=provenance,
                       proposed_intervals_seconds=[], permission=withheld)
        updated.pop('candidate', None)
        assessment.pop('preview', None)
        return updated, assessment
    with np.load(directory / 'model_predictions.npz') as archive:
        times = archive['times'].copy()
        requested = archive['proposed_correction'].copy()
        reliable = archive['reliable'].copy()
    if waveform_hash(requested) != plan['proposal_sha256'] or requested.shape != times.shape or reliable.shape != times.shape or reliable.dtype != bool:
        raise ValueError('Saved prediction grid does not match its permission.')
    curve = requested.copy()
    curve[~_mask(times, chosen)] = 0
    combined = original.copy()
    inside = _mask(sample_times, chosen)
    combined[inside] = candidate[inside]
    protection = report.get('user_protection', {}).get('intervals_seconds', [])
    protected = _mask(sample_times, protection)
    if not np.array_equal(combined[protected], original[protected]):
        raise ValueError('Selection changes protected samples.')
    attempt = directory / 'review_combinations' / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    path = attempt / 'unverified_combination.wav'
    sf.write(path, combined, rate, subtype='PCM_24')
    receipt = dict(**provenance, source_sha256=audio.checksum, success=False)
    try:
        decoded, decoded_rate = sf.read(path, dtype='float32')
        if decoded_rate != rate or not np.array_equal(decoded, combined):
            raise ValueError('Combination encoding changed selected samples.')
        audit, assessed = _audit_and_assess(path, original, rate, times, curve, reliable,
                                          tuple(chosen), review.get('intent', 'unsure'))
        if audit.get('passed') is not True or audit.get('sha256') != checksum(path):
            raise ValueError('Final combined decoded-WAV audit failed.')
        if checksum(audio.path) != audio.checksum or checksum(preview['path']) != preview['sha256']:
            raise ValueError('Source or preview changed during combined audit.')
        final = path.with_name('verified_combination.wav')
        path.rename(final)
        new_plan = copy.deepcopy(plan)
        new_plan.update(intervals_seconds=chosen, affected_fraction=float(inside.mean()),
                        proposal_sha256=waveform_hash(curve), requested_max_abs_cents=float(abs(curve).max()))
        new_plan['decision_sha256'] = decision_hash(new_plan)
        updated.update(outcome='VERIFIED_FIXTURE_CANDIDATE', permission=new_plan,
            candidate=dict(path=str(final), sha256=checksum(final)), proposed_intervals_seconds=chosen,
            local_selection=dict(provenance, final_decoded_audit=audit),
            transaction=dict(outcome='VERIFIED_FIXTURE_CANDIDATE', post_render_audit=audit['acoustic_audit'],
                             delivered_wav_audit=audit))
        assessed['waveform_sha256'] = checksum(final)
        assessment['preview'] = assessed
        receipt.update(success=True, final_sha256=checksum(final), final_decoded_audit=audit)
        return updated, assessment
    except Exception as exc:
        receipt.update(reason=str(exc), error_type=type(exc).__name__)
        raise ValueError('Final combination was not accepted: ' + str(exc)) from exc
    finally:
        (attempt / 'review_selection.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
