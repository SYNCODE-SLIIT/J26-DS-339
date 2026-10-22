"""Small adapter for the existing Transformer, rule baseline and noise preview."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from bpdr_input import COMPONENT_ROOT
from bpdr_input.audio.pipeline import checksum, process

METHODS = ('BPDR Transformer', 'Conservative stable-note baseline', 'Noise reduction only', 'Validate and preserve')


def available_methods():
    from bpdr_input.repair.learned_runtime import METHOD_NAME, available
    return METHODS + ((METHOD_NAME,) if available() else ())


def run(source, method=METHODS[0], noise_reference=False, output_root=None):
    from bpdr_input.repair.learned_runtime import METHOD_NAME
    if method == METHOD_NAME:
        from bpdr_input.repair.learned_runtime import run as learned
        return learned(source, output_root=output_root)
    if method not in METHODS:
        raise ValueError('Unknown processing method')
    root = Path(output_root or COMPONENT_ROOT / 'BPDR_Workspace/results/demo_sessions')
    if method == METHODS[3]:
        result, directory = process(source, root, COMPONENT_ROOT / 'checkpoints')
        report = dict(version='unchanged-demo-adapter-1', source=str(source), method=method,
                      handoff=result.to_dict(), prepared_handoff_enabled=False,
                      outcome='UNCHANGED_ANALYSIS' if result.accepted_audio else 'NOT_RUN_TECHNICAL_STOP')
        (directory / 'model_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return directory, report
    if method == METHODS[0]:
        from bpdr_input.repair.model_runtime import run as infer
        return infer(source, output_root=root, render_preview=True)
    if method == METHODS[2]:
        if not noise_reference:
            raise ValueError('Noise reduction requires a declared voice-free first 0.7 seconds.')
        from bpdr_input.repair.noise_preview import run as clean
        from bpdr_input.repair.noise_policy_v2 import ColoredNoisePolicy
        return clean(source, (0., .7), output_root=root, noise_policy=ColoredNoisePolicy())

    from bpdr_input.audio.analysis import extract
    from bpdr_input.repair.rule_correction import run as rule
    result, directory = process(source, root, COMPONENT_ROOT / 'checkpoints')
    report = dict(version='conservative-demo-adapter-1', source=str(source), method=method,
                  handoff=result.to_dict(), prepared_handoff_enabled=False,
                  outcome='NOT_RUN_TECHNICAL_STOP',
                  scope='Engineering preview; no intended-note or musical-usability certification')
    if result.accepted_audio:
        try:
            audio, sr = sf.read(result.accepted_audio.path, dtype='float32')
            with np.load(directory / 'features.npz') as archive:
                features = {key: archive[key] for key in archive.files}
            output, detail = rule(audio, sr, features,
                                  lambda waveform: extract(waveform, sr, COMPONENT_ROOT / 'checkpoints')[0])
            report.update(outcome=detail['outcome'], rule=detail)
            if detail['outcome'] == 'VERIFIED_FIXTURE_CANDIDATE':
                candidate = directory / 'experimental_rule_candidate.wav'
                sf.write(candidate, output, sr, subtype='PCM_24')
                report['candidate'] = dict(path=str(candidate), sha256=checksum(candidate))
        except Exception as exc:
            report.update(outcome='RULE_PREVIEW_FAILED_UNCHANGED', error_type=type(exc).__name__, reason=str(exc))
    (directory / 'model_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return directory, report


def eligible_preview(report):
    """A saved file is a usable preview only while its outcome and audit agree."""
    if report['outcome'] == 'NOISE_PREVIEW_PASSED' and report.get('noise_candidate'):
        return report['noise_candidate'] if report.get('noise', {}).get('audit', {}).get('passes') is True else None
    if report['outcome'] == 'VERIFIED_FIXTURE_CANDIDATE' and report.get('candidate'):
        transaction = report.get('transaction') or report.get('rule', {}).get('transaction', {})
        return report['candidate'] if transaction.get('post_render_audit', {}).get('passed') is True else None
    return None


def selected_output(report):
    """Never send a rejected file or a failed candidate to playback as an output."""
    accepted = report['handoff'].get('accepted_audio')
    if not accepted:
        return None, 'No accepted audio'
    preview = eligible_preview(report)
    item = preview or accepted
    path = Path(item['path'])
    expected = item.get('sha256') or item.get('checksum')
    if not expected or checksum(path) != expected:
        raise ValueError('Output checksum mismatch')
    return path, 'Processed preview' if preview else 'Original retained'
