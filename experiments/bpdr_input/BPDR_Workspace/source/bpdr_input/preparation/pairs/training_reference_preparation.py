"""Training-only accepted-cache authentication and frozen reference-mask replay.

No audio writer, intervention renderer, pair builder or optimizer belongs here.
Phase-1 features precede PCM24 encoding and cannot stand in for decoded features.
"""
from pathlib import Path
import numpy as np
import soundfile as sf
from bpdr_input.audio.contracts import AcceptedAudio, ProcessingResult, Status
from bpdr_input.audio.pipeline import checksum, validate_package
from bpdr_input.preparation.sources.dataset import DataPolicy, reference_mask
from bpdr_input.repair.interventions import AuditPolicy, select_support
from bpdr_input.preparation.diagnostics.support_geometry import note_pair_ids, select_multi_interior, geometry_masks


def require_train(phrase):
    if phrase.get('split') != 'train':
        raise ValueError('Only the frozen training partition is allowed')


def authenticate_cache(phrase, diagnostics):
    """Bind original bytes, preserved copy and decoded accepted package to a train crop."""
    require_train(phrase)
    source = diagnostics['source']
    if source['sha256'] != phrase['sha256']:
        raise ValueError('Cache belongs to a different crop')
    for path in (phrase['audio_path'], source['path'], source['preserved_path']):
        if checksum(path) != phrase['sha256']:
            raise ValueError('Original or preserved source checksum changed')
    record = diagnostics['result']
    accepted = record.get('accepted_audio')
    package = None if accepted is None else AcceptedAudio(**{
        k: accepted[k] for k in ('path', 'checksum', 'sample_rate_hz', 'sample_count', 'channels')})
    result = ProcessingResult(request_id=record['request_id'], source_id=record['source_id'],
        status=Status(record['status']), accepted_audio=package,
        reason_codes=tuple(record['reason_codes']), policy_version=record['policy_version'])
    accepted = validate_package(result)
    if accepted.sample_rate_hz != 22050 or accepted.sample_count != round(
            phrase['duration_seconds'] * accepted.sample_rate_hz):
        raise ValueError('Accepted timeline differs from frozen crop')
    audio, sr = sf.read(accepted.path, dtype='float32')
    return accepted, audio, sr


def decoded_cache_identity(accepted_audio, cached_waveform):
    """Encoding must not be guessed from filenames or aligned by truncation."""
    cached, sr = sf.read(cached_waveform, dtype='float32')
    if sr != accepted_audio[1] or not np.array_equal(cached, accepted_audio[0]):
        raise ValueError('Cached baseline waveform is not exactly the decoded accepted audio')
    return True


def prepare_masks(phrase, before, yin, f0, notes, data_policy=DataPolicy(), policy=AuditPolicy()):
    require_train(phrase)
    times = np.asarray(before['times_seconds'])
    yin = np.asarray(yin)
    voiced = np.asarray(before['voiced_mask'])
    pitch = np.asarray(before['f0_hz'])
    confidence = np.asarray(before['confidence'])
    if (yin.shape != times.shape or voiced.dtype != bool or voiced.shape != times.shape
            or pitch.shape != times.shape or confidence.shape != times.shape
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or np.any(voiced != ((confidence >= .8) & np.isfinite(pitch) & (pitch > 0)))):
        raise ValueError('Invalid aligned original acoustic arrays')
    annotation, reference = reference_mask(times, f0, notes, data_policy)
    both = annotation & voiced & (reference > 0) & np.isfinite(yin) & (yin > 0)
    reliable = both.copy()
    reliable[both] &= abs(1200 * np.log2(yin[both] / pitch[both])) <= policy.tracker_agreement_cents
    ids = note_pair_ids(times, notes, data_policy.boundary_guard_seconds)
    candidate = select_multi_interior(times, reliable, ids, phrase['duration_seconds'], 'train', policy)
    try:
        old = [float(x) for x in select_support(times, reliable, policy)]
        old_reason = None
    except ValueError as exc:
        old = None
        old_reason = str(exc)
    plateau = np.zeros(len(times), dtype=bool)
    outside = plateau.copy()
    if candidate is not None:
        plateau, outside = geometry_masks(times, reliable, candidate['support_seconds'], policy)
    arrays = dict(times_seconds=times, annotation_mask=annotation, reference_f0_hz=reference,
        voiced_mask=voiced, yin_f0_hz=yin, reference_voiced_positive_yin_mask=both,
        reliable_mask=reliable, note_pair_ids=ids, candidate_plateau_mask=plateau,
        candidate_outside_mask=outside)
    row = dict(annotation_frames=int(annotation.sum()), runtime_voiced_frames=int(voiced.sum()),
        reference_voiced_positive_yin_frames=int(both.sum()), reliable_frames=int(reliable.sum()),
        tracker_disagreement_frames=int(both.sum()-reliable.sum()),
        old_continuous_support_seconds=old, old_continuous_refusal=old_reason,
        multi_interior_candidate=candidate, geometry_available=candidate is not None,
        acoustic_shift_audit_passed=None, new_paired_examples=0, training_eligible=False,
        review_status=phrase['review_status'], data_status='exploratory_not_research')
    return arrays, row
