"""Time-local recording observations, separate from acceptance and intention."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from bpdr_input.audio.pipeline import Policy, checksum
from bpdr_input.audio.usability import UsabilityPolicy


def _spans(mask, starts, ends, minimum_seconds):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        start, end = float(starts[a]), float(ends[b - 1])
        if end > start and end - start + 1e-9 >= minimum_seconds:
            yield start, end


def timeline(audio, rate, features=None, independent_hz=None):
    """Describe measured spans without assigning note errors or changing status."""
    audio = np.asarray(audio)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or not isinstance(rate, int) or rate <= 0:
        raise ValueError('Feedback needs finite, nonempty mono samples and a positive rate.')
    duration = len(audio) / rate
    policy = UsabilityPolicy()
    hop = max(1, round(rate * .02))
    starts = np.arange(0, len(audio), hop) / rate
    ends = np.minimum(starts + hop / rate, duration)
    quiet, peaks = [], []
    for first in range(0, len(audio), hop):
        block = audio[first:first + hop].astype(np.float64)
        quiet.append(np.sqrt(np.mean(block ** 2)) < Policy().silence_rms)
        peaks.append(bool(np.any(abs(block) >= .999)))
    rows = []

    def add(mask, times, stops, category, label, guidance, minimum):
        for start, end in _spans(np.asarray(mask, dtype=bool), times, stops, minimum):
            rows.append(dict(start_s=start, end_s=end, category=category,
                             observation=label, guidance=guidance))

    add(quiet, starts, ends, 'quiet_or_pause', 'Quiet range or pause',
        'Keep this range if the pause is intentional. If voice should be audible here, move closer to the microphone or try a quieter room.',
        policy.minimum_contiguous_reliable_seconds)
    add(peaks, starts, ends, 'near_full_scale', 'Near-full-scale peaks',
        'Listen for harsh or distorted peaks here. If present, lower the input level and record again.', 0.)
    compared = 0
    if features is not None and independent_hz is not None:
        times = np.asarray(features['times_seconds'], dtype=np.float64)
        f0 = np.asarray(features['f0_hz'], dtype=np.float64)
        confidence = np.asarray(features['confidence'], dtype=np.float64)
        voiced = np.asarray(features['voiced_mask'])
        secondary = np.asarray(independent_hz, dtype=np.float64)
        arrays = [times, f0, confidence, voiced, secondary]
        if any(a.ndim != 1 or a.shape != times.shape for a in arrays) or len(times) < 2:
            raise ValueError('Feedback pitch arrays must share a frame grid.')
        if not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError('Invalid feedback timestamps.')
        step = float(np.median(np.diff(times)))
        if not np.allclose(np.diff(times), step, rtol=1e-4, atol=1e-9):
            raise ValueError('Feedback needs a regular time grid.')
        if voiced.dtype != bool or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError('Invalid feedback confidence or voiced flags.')
        pitch_duration = min(duration, float(features.get('_audio_duration_seconds', duration)))
        if not np.isfinite(pitch_duration) or pitch_duration <= 0:
            raise ValueError('Invalid working-audio duration for feedback.')
        in_audio = (times >= 0) & (times < pitch_duration)
        common = in_audio & voiced & (confidence >= policy.minimum_confidence) & np.isfinite(f0) & (f0 > 0) & np.isfinite(secondary) & (secondary > 0)
        compared = int(common.sum())
        difference = np.zeros(len(times))
        difference[common] = abs(1200 * np.log2(secondary[common] / f0[common]))
        add(common & (difference > policy.maximum_tracker_disagreement_cents), times,
            np.minimum(times + step, pitch_duration), 'uncertain_pitch_measurement', 'Pitch measurements disagree',
            'Listen to this range. A clearer solo take can help if the voice is masked; slides and expression can be intentional.',
            policy.minimum_contiguous_reliable_seconds)
    rows.sort(key=lambda row: (row['start_s'], row['end_s'], row['category']))
    return dict(duration_seconds=duration, resolution_seconds=hop / rate,
        minimum_quiet_or_pitch_span_seconds=policy.minimum_contiguous_reliable_seconds,
        independent_pitch_frames_compared=compared, observations=rows,
        scope='Input-only acoustic observations. No intended-note, error, acceptance or rerecord decision is inferred.',
        acceptance_changed=False, intended_notes_assessed=False)


def for_recording(directory, report, previous=None):
    """Use preserved raw input for level spans and the working grid for pitch."""
    directory = Path(directory)
    diagnostics_path = directory / 'diagnostics.json'
    diagnostics = json.loads(diagnostics_path.read_text(encoding='utf-8')) if diagnostics_path.exists() else {}
    source = Path(diagnostics.get('source', {}).get('preserved_path', report['source']))
    source_hash = checksum(source)
    expected = diagnostics.get('source', {}).get('sha256')
    if expected and source_hash != expected:
        raise ValueError('Preserved input changed before recording feedback.')
    feature_path = directory / 'features.npz'
    accepted = report['handoff'].get('accepted_audio')
    binding = dict(feedback_source_sha256=checksum(__file__), source_sha256=source_hash,
        feature_sha256=checksum(feature_path) if feature_path.exists() else None,
        working_sha256=checksum(accepted['path']) if accepted else None)
    if accepted and binding['working_sha256'] != accepted['checksum']:
        raise ValueError('Working input changed before recording feedback.')
    if previous and previous.get('binding') == binding:
        return previous
    samples, rate = sf.read(source, dtype='float32', always_2d=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Preserved input cannot be measured for feedback.')
    channel = int(np.argmax(np.mean(samples.astype(np.float64) ** 2, axis=0)))
    features, secondary = None, None
    if accepted and feature_path.exists():
        with np.load(feature_path) as saved:
            features = {k: saved[k] for k in ('times_seconds', 'f0_hz', 'confidence', 'voiced_mask')}
        working, working_rate = sf.read(accepted['path'], dtype='float32')
        features['_audio_duration_seconds'] = len(working) / working_rate
        from bpdr_input.repair.rule_correction import fine_hz
        secondary = fine_hz(working, working_rate, features['times_seconds'])
    result = timeline(samples[:, channel], rate, features, secondary)
    if (checksum(source) != source_hash or (accepted and checksum(accepted['path']) != binding['working_sha256'])
            or (feature_path.exists() and checksum(feature_path) != binding['feature_sha256'])):
        raise ValueError('Input changed while producing recording feedback.')
    result.update(binding=binding, selected_input_channel=channel, input_channels=samples.shape[1])
    return result
