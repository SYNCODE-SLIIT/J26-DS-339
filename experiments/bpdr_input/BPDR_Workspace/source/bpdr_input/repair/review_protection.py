"""User protection veto on an existing preview, without creating new edits."""

import copy

import numpy as np
import soundfile as sf

from bpdr_input.audio.contracts import AcceptedAudio, ProcessingResult, Status
from bpdr_input.audio.pipeline import checksum, validate_package
from bpdr_input.repair.demo_runtime import eligible_preview


def protect(report, review, intervals):
    """Keep the original when any decoded protected sample would change.

    Whole-clip operations are vetoed as a whole. No sample splicing, inferred
    target or relaxed acoustic gate is introduced by a review decision.
    """
    accepted = report['handoff'].get('accepted_audio')
    if not accepted:
        raise ValueError('Protection requires technically accepted source audio.')
    audio = AcceptedAudio(**{k: accepted[k] for k in
        ['path', 'checksum', 'sample_rate_hz', 'sample_count', 'channels']})
    package = ProcessingResult(report['handoff']['request_id'], report['handoff']['source_id'],
                               Status(report['handoff']['status']), audio)
    validate_package(package)
    original, rate = sf.read(audio.path, dtype='float64')
    times = np.arange(len(original)) / rate
    mask = np.zeros(len(original), dtype=bool)
    canonical = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError('A protected range needs a start and end.')
        start, end = map(float, interval)
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= len(original) / rate:
            raise ValueError('Protected ranges must lie inside the recording, with start before end.')
        local = (times >= start) & (times < end)
        if not local.any():
            raise ValueError('Protected range contains no audio samples.')
        canonical.append([start, end])
        mask |= local
    if not canonical:
        raise ValueError('Supply at least one protected range.')
    updated, assessment = copy.deepcopy(report), copy.deepcopy(review)
    preview = eligible_preview(report)
    changed = 0
    candidate_hash = None
    if preview:
        candidate_hash = checksum(preview['path'])
        if candidate_hash != preview['sha256']:
            raise ValueError('Preview changed before protection review.')
        candidate, candidate_rate = sf.read(preview['path'], dtype='float64')
        if candidate_rate != rate or candidate.shape != original.shape or not np.isfinite(candidate).all():
            raise ValueError('Preview cannot be compared on the original sample grid.')
        changed = int(np.count_nonzero(candidate[mask] != original[mask]))
    veto = changed > 0
    updated['user_protection'] = dict(intervals_seconds=canonical,
        source_sha256=audio.checksum, preview_sha256=candidate_hash,
        protected_sample_count=int(mask.sum()), changed_protected_samples=changed,
        preview_vetoed=veto, policy='Exact decoded samples; overlapping preview is vetoed as a whole.',
        musical_quality_claimed=False)
    if veto:
        updated['outcome'] = 'USER_PROTECTED_UNCHANGED'
        updated.pop('candidate', None)
        updated.pop('noise_candidate', None)
        assessment.pop('preview', None)
    if checksum(audio.path) != audio.checksum:
        raise ValueError('Source changed during protection review.')
    return updated, assessment
