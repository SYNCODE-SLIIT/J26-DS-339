"""Read-only distinction between requests, permitted previews and selected WAVs."""

from pathlib import Path

import numpy as np
import soundfile as sf

from bpdr_input.audio.contracts import AcceptedAudio, ProcessingResult, Status
from bpdr_input.audio.pipeline import checksum, validate_package
from bpdr_input.repair.demo_runtime import eligible_preview
from bpdr_input.repair.repair_controller import waveform_hash


def summarize(directory, report, role):
    if role not in ('original', 'preview'):
        raise ValueError('Unknown selected version.')
    data = report['handoff']
    accepted = data.get('accepted_audio')
    if not accepted:
        raise ValueError('No accepted source is available for comparison.')
    audio = AcceptedAudio(**{k: accepted[k] for k in ['path', 'checksum', 'sample_rate_hz', 'sample_count', 'channels']})
    validate_package(ProcessingResult(data['request_id'], data['source_id'], Status(data['status']), audio))
    preview = eligible_preview(report)
    if role == 'preview' and not preview:
        raise ValueError('No permitted preview is available for selection.')
    item = preview if role == 'preview' else accepted
    selected_hash = item.get('sha256') or item.get('checksum')
    if checksum(item['path']) != selected_hash:
        raise ValueError('Selected waveform checksum changed.')
    source, rate = sf.read(audio.path, dtype='float64')
    selected, selected_rate = sf.read(item['path'], dtype='float64')
    if selected_rate != rate or selected.shape != source.shape or not np.isfinite(selected).all():
        raise ValueError('Selected waveform is not on the working source grid.')
    changed = int(np.count_nonzero(source != selected))
    cleanup = bool(preview and report.get('noise_candidate'))
    plan = report.get('permission') or report.get('rule', {}).get('plan') or {}
    intervals = plan.get('intervals_seconds', []) if preview and plan.get('permitted') else []
    for start, end in intervals:
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= len(source) / rate:
            raise ValueError('Invalid preview permission range.')
    model_max = operating_max = None
    permitted_max = None if cleanup or (preview and not intervals) else 0.
    curve = times = None
    prediction_path = Path(directory) / 'model_predictions.npz'
    prediction_hash = None
    if prediction_path.exists() and not cleanup:
        prediction_hash = checksum(prediction_path)
        with np.load(prediction_path) as saved:
            times = saved['times'].copy()
            curve = saved['proposed_correction'].copy()
            predictions = saved['predictions'].copy() if 'predictions' in saved.files else None
        if times.ndim != 1 or curve.shape != times.shape or not np.isfinite(times).all() or not np.isfinite(curve).all():
            raise ValueError('Request grid is invalid.')
        if predictions is not None:
            if predictions.ndim != 2 or predictions.shape[1] != len(times) or not np.isfinite(predictions).all():
                raise ValueError('Model predictions do not share the request grid.')
            model_max = float(np.max(abs(predictions[0])))
        operating_max = float(np.max(abs(curve)))
    elif not cleanup and report.get('rule', {}).get('corrections_cents') is not None:
        curve = np.asarray(report['rule']['corrections_cents'], dtype=np.float64)
        times = np.arange(len(curve)) * .02
        if not len(curve) or not np.isfinite(curve).all():
            raise ValueError('Rule request is invalid.')
        operating_max = float(np.max(abs(curve)))
    if preview and intervals and not cleanup:
        if curve is None:
            permitted_max = None
        else:
            permitted = curve.copy()
            inside = np.zeros(len(times), dtype=bool)
            for start, end in intervals:
                inside |= (times >= start) & (times < end)
            permitted[~inside] = 0
            if plan.get('proposal_sha256') and waveform_hash(permitted) != plan['proposal_sha256']:
                raise ValueError('Permitted request does not match its saved decision.')
            permitted_max = float(np.max(abs(permitted)))
    operation = ('whole_clip_cleanup' if cleanup else 'local_pitch' if preview and intervals
                 else 'verified_preview' if preview else 'none')
    if checksum(audio.path) != audio.checksum or checksum(item['path']) != selected_hash:
        raise ValueError('Source or selected waveform changed during comparison.')
    if prediction_hash and checksum(prediction_path) != prediction_hash:
        raise ValueError('Saved request changed during comparison.')
    return dict(model_request_max_abs_cents=model_max,
        operating_request_max_abs_cents=operating_max,
        permitted_preview_request_max_abs_cents=permitted_max,
        permitted_preview_intervals_seconds=intervals if not cleanup else [[0., len(source) / rate]],
        preview_operation=operation, selected_role=role,
        selected_request_max_abs_cents=0. if role == 'original' else permitted_max,
        actual_changed_samples=changed, actual_sample_count=len(source),
        actual_changed_fraction=changed / len(source), sample_rate_hz=rate,
        source_working_sha256=audio.checksum, selected_sha256=selected_hash,
        predictions_sha256=prediction_hash, acceptance_changed=False,
        scope='Requests and permissions describe proposed processing. Sample changes are measured from the selected decoded WAV; no pitch-accuracy claim.')
