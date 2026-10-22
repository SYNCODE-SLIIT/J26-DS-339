"""Provisional acoustic review and explicit user-reviewed demo handoff.

This never changes the shared runtime's accepted waveform or model policy.
"""
import io
import json
from pathlib import Path
import shutil
import uuid
import zipfile

import numpy as np
import soundfile as sf

from bpdr_input import COMPONENT_ROOT
from bpdr_input.audio.contracts import AcceptedAudio, ProcessingResult, Status
from bpdr_input.audio.pipeline import checksum, validate_package
from bpdr_input.repair.demo_runtime import eligible_preview


def _assessment(path, features, intent):
    from bpdr_input.audio.usability import assess
    from bpdr_input.repair.rule_correction import fine_hz
    audio, sr = sf.read(path, dtype='float32')
    independent = fine_hz(audio, sr, features['times_seconds'])
    result = assess(features, 'ACCEPT_UNCHANGED', intent=intent, independent_f0_hz=independent, duration_seconds=len(audio)/sr)
    result['waveform_sha256'] = checksum(path)
    return result


def review(directory, report, intent='unsure'):
    from bpdr_input.audio.usability import assess
    from bpdr_input.audio.analysis import extract
    directory = Path(directory)
    accepted = report['handoff'].get('accepted_audio')
    result = dict(intent=intent, shared_handoff_changed=False,
                  scope='Provisional acoustic evidence, not musical correctness or an intention classifier')
    if not accepted:
        result['original'] = assess({}, report['handoff']['status'], intent=intent)
    else:
        with np.load(directory / 'features.npz') as archive:
            features = {key: archive[key] for key in archive.files}
        if checksum(accepted['path']) != accepted['checksum']:
            raise ValueError('Accepted input changed before review')
        result['original'] = _assessment(accepted['path'], features, intent)
        preview = eligible_preview(report)
        if preview:
            if checksum(preview['path']) != preview['sha256']:
                raise ValueError('Preview changed before review')
            cached = directory / 'noise_candidate_features.npz'
            if report.get('noise_candidate') and cached.exists():
                with np.load(cached) as archive:
                    after = {key: archive[key] for key in archive.files}
            else:
                audio, sr = sf.read(preview['path'], dtype='float32')
                after = extract(audio, sr, COMPONENT_ROOT / 'checkpoints')[0]
            result['preview'] = _assessment(preview['path'], after, intent)
    (directory / 'usability_review.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def export_reviewed(directory, report, assessment, *, use_preview=False, confirmed=False, confirmation_kind='user'):
    """Export only an explicitly confirmed waveform with sufficient demo evidence."""
    if not confirmed:
        raise ValueError('Confirm that the selected audio preserves your intended musical idea before exporting.')
    if confirmation_kind not in ('user', 'automated_test_fixture'):
        raise ValueError('Unknown confirmation kind.')
    if not report['handoff'].get('accepted_audio'):
        raise ValueError('No technically accepted audio is available.')
    role = 'preview' if use_preview else 'original'
    if assessment.get(role, {}).get('decision') != 'PASS_EVIDENCE':
        raise ValueError('This version lacks sufficient confirmed acoustic evidence for demo handoff.')
    item = (report.get('noise_candidate') or report.get('candidate')) if use_preview else report['handoff']['accepted_audio']
    if not item:
        raise ValueError('No verified prepared preview is available.')
    if use_preview and report['outcome'] not in ('VERIFIED_FIXTURE_CANDIDATE', 'NOISE_PREVIEW_PASSED'):
        raise ValueError('The processing outcome does not authorize this preview.')
    if use_preview:
        if report.get('noise_candidate'):
            passed = report.get('noise', {}).get('audit', {}).get('passes') is True
        else:
            transaction = report.get('transaction') or report.get('rule', {}).get('transaction', {})
            passed = transaction.get('post_render_audit', {}).get('passed') is True
        if not passed:
            raise ValueError('The candidate has no passed waveform audit.')
    expected = item.get('sha256') or item.get('checksum')
    if checksum(item['path']) != expected:
        raise ValueError('Selected waveform checksum mismatch.')
    if assessment[role].get('waveform_sha256') != expected:
        raise ValueError('Acoustic review does not belong to the selected waveform.')
    destination = Path(directory).resolve() / 'reviewed_handoffs' / uuid.uuid4().hex
    destination.mkdir(parents=True)
    path = destination / 'accepted.wav'
    shutil.copyfile(item['path'], path)
    if checksum(path) != expected:
        raise ValueError('Copied waveform no longer matches the reviewed source.')
    info = sf.info(path)
    accepted = AcceptedAudio(str(path), expected, info.samplerate, info.frames, info.channels)
    status = Status.ACCEPT_PREPARED if use_preview else Status.ACCEPT_UNCHANGED
    message = 'User-reviewed demo waveform; acoustic policy is provisional.' if confirmation_kind == 'user' else 'Automated integration fixture; no human musical review was performed.'
    handoff = ProcessingResult(destination.name, report['handoff']['source_id'], status, accepted,
                              user_message=message,
                              policy_version='demo-user-reviewed-0.1')
    validate_package(handoff)
    diagnostics = dict(scope='User-reviewed demonstration export, not validated autonomous musical acceptance' if confirmation_kind == 'user' else 'Automated integration fixture, not human musical approval',
                       user_confirmed=confirmation_kind == 'user', confirmation_kind=confirmation_kind,
                       shared_handoff_changed=False, research_approval=False,
                       melodic_usability='provisional_unvalidated', intended_note_inference=False,
                       selection=role, processing_outcome=report['outcome'], usability=assessment[role],
                       original_processing_handoff=report['handoff'],
                       changed_intervals_seconds=(report.get('proposed_intervals_seconds') or report.get('rule', {}).get('plan', {}).get('intervals_seconds', [])) if use_preview else [],
                       noise_operation=report.get('noise') if use_preview else None,
                       model_sha256=report.get('model_sha256'), method=report.get('method', report.get('version')))
    (destination / 'handoff.json').write_text(json.dumps(handoff.to_dict(), indent=2), encoding='utf-8')
    (destination / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding='utf-8')
    portable = handoff.to_dict()
    portable['accepted_audio']['path'] = 'accepted.wav'
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(path, 'accepted.wav')
        archive.writestr('handoff.json', json.dumps(portable, indent=2))
        archive.write(destination / 'diagnostics.json', 'diagnostics.json')
    (destination / 'accepted_package.zip').write_bytes(stream.getvalue())
    return destination, handoff.to_dict()
