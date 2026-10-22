"""Audio-only adapter for a separately validated stable-note recovery model."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from bpdr_input import COMPONENT_ROOT
from bpdr_input.audio.pipeline import checksum, process, validate_package

WORKSPACE = COMPONENT_ROOT / 'BPDR_Workspace'
CARD = WORKSPACE / 'results/model_recovery/demo_model.json'
ACTIVE_DEMO = WORKSPACE / 'results/notebook_training/current.json'
METHOD_NAME = 'Learned pitch repair (experimental)'


def active_card_path():
    """Use a verified notebook-trained card when explicitly activated."""
    if not ACTIVE_DEMO.exists():
        return CARD
    pointer = json.loads(ACTIVE_DEMO.read_text(encoding='utf-8'))
    path = Path(pointer['card']).resolve()
    root = (WORKSPACE / 'results/model_recovery/notebook_training').resolve()
    if not path.is_relative_to(root) or path.name != 'demo_model.json':
        raise ValueError('Active notebook demo card leaves its training-run directory.')
    if checksum(path) != pointer['card_sha256']:
        raise ValueError('Active notebook demo card changed.')
    card = json.loads(path.read_text(encoding='utf-8'))
    if (Path(card['checkpoint']).resolve().parent != path.parent or
            card['checkpoint_sha256'] != pointer['checkpoint_sha256'] or not card.get('notebook_training')):
        raise ValueError('Active notebook demo pointer and checkpoint disagree.')
    return path


def load_model(card_path=None):
    from bpdr_input.modeling.recovery_model import load_candidate, load_locked_diagnostic
    card_file = Path(card_path) if card_path is not None else active_card_path()
    card = json.loads(card_file.read_text(encoding='utf-8'))
    path = Path(card['checkpoint']).resolve()
    if not path.is_relative_to(WORKSPACE.resolve()):
        raise ValueError('Demo model must remain in BPDR_Workspace.')
    if not card.get('eligible_for_demo') or checksum(path) != card['checkpoint_sha256']:
        raise ValueError('Demo model is not eligible or its checkpoint changed.')
    if card.get('notebook_training'):
        receipt_path = path.parent / 'fit_receipt.json'
        if checksum(receipt_path) != card['fit_receipt_sha256']:
            raise ValueError('Notebook model fit receipt changed.')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if (not receipt.get('fit_completed') or receipt.get('calibration_or_pilot_used_for_fit') is not False
                or receipt['declaration_sha256'] != checksum(path.parent / 'declaration.json')):
            raise ValueError('Notebook model lacks a verified train-only fit receipt.')
    if card.get('validation_status') == 'exploratory_preview_failed_generalization_gate':
        model, payload = load_locked_diagnostic(path)
        verification = path.parent / 'rendered_verification.json'
        if checksum(verification) != card['verification_sha256']:
            raise ValueError('Exploratory preview verification changed.')
        measured = json.loads(verification.read_text(encoding='utf-8'))
        originals, shifts = measured['originals'], measured['shifts']
        if (measured['checkpoint_sha256'] != checksum(path) or len(originals) != 4
                or not all(r['preserved'] for r in originals)
                or not any(r.get('verified_improvement') for r in shifts)
                or not card.get('requires_user_review') or card.get('research_gates_passed') is not False):
            raise ValueError('Exploratory preview lacks intact original anchors or a verified example.')
    else:
        model, payload = load_candidate(path)
    return model, payload, card


def available():
    try:
        load_model()
    except (OSError, ValueError, KeyError, ImportError):
        return False
    return True


def run(source, output_root=None):
    """UI entry point: a model must pass the guarded loader before any inference."""
    model, payload, card = load_model()
    return run_candidate(source, model, payload, card, output_root=output_root)


def audit_delivered_wav(path, original, sr, intervals, auditor):
    """Verify the exact PCM24 bytes that playback and export will consume.

    The in-memory render audit remains separate. These checks repeat its sample
    integrity and outside-support rules on the decoded delivery file, followed
    by the same acoustic auditor. File hashes bind the result to those bytes.
    """
    from bpdr_input.repair.repair_controller import check_source, waveform_hash
    path = Path(path)
    encoded = path.read_bytes()
    info = sf.info(io.BytesIO(encoded))
    candidate, candidate_sr = sf.read(io.BytesIO(encoded), dtype='float32')
    if info.format != 'WAV' or info.subtype != 'PCM_24':
        raise ValueError('ENCODED_CANDIDATE_FORMAT')
    if candidate_sr != sr or candidate.shape != original.shape:
        raise ValueError('ENCODED_CANDIDATE_SAMPLE_GRID')
    check_source(candidate, candidate_sr)
    inside = np.zeros(len(original), dtype=bool)
    sample_times = np.arange(len(original)) / sr
    for start, end in intervals:
        inside |= (sample_times >= start) & (sample_times < end)
    if not np.array_equal(candidate[~inside], original[~inside]):
        raise ValueError('ENCODED_CANDIDATE_OUTSIDE_SUPPORT_CHANGED')
    measured = auditor(original.copy(), candidate.copy(), sr, tuple(intervals))
    if not isinstance(measured, dict) or measured.get('passed') is not True:
        raise ValueError('ENCODED_CANDIDATE_ACOUSTIC_AUDIT_FAILED')
    digest = hashlib.sha256(encoded).hexdigest()
    if checksum(path) != digest:
        raise ValueError('ENCODED_CANDIDATE_CHANGED_DURING_AUDIT')
    return dict(passed=True, sha256=digest, decoded_waveform_sha256=waveform_hash(candidate),
                format=info.format, subtype=info.subtype, sample_rate_hz=candidate_sr,
                sample_count=len(candidate), channels=info.channels,
                outside_support_unchanged=True, acoustic_audit=measured)


def run_candidate(source, model, payload, card, output_root=None):
    """Internal evaluation path using the exact live operating/render algorithm.

    A bounded evaluator may supply a locked candidate before rendered adoption.
    The public UI calls ``run`` and therefore always uses the guarded loader.
    This function neither adopts a model nor creates an eligibility card.
    """
    import torch
    torch.set_num_threads(2)
    from bpdr_input.audio.analysis import extract
    from bpdr_input.modeling.recovery_model import propose
    from bpdr_input.repair.model_runtime import runtime_views
    from bpdr_input.repair.rule_correction import fine_pitch, fine_hz
    from bpdr_input.repair.repair_controller import plan_fixture, transact_fixture
    from bpdr_input.repair.framewise_renderer import render_candidate
    from bpdr_input.repair.framewise_audit import audit_curve

    result, directory = process(Path(source), output_root or WORKSPACE / 'results/demo_sessions', COMPONENT_ROOT / 'checkpoints')
    report = dict(version='learned-stable-note-runtime-1', source=str(source), method=METHOD_NAME,
                  handoff=result.to_dict(), prepared_handoff_enabled=False,
                  model_sha256=card['checkpoint_sha256'], model_path=card['checkpoint'],
                  model_scope=card.get('scope'), model_metrics=card.get('metrics', {}),
                  model_validation_status=card.get('validation_status', 'validated_scoped_demo'),
                  research_gates_passed=card.get('research_gates_passed', True),
                  outcome='NOT_RUN_TECHNICAL_STOP')
    if result.accepted_audio:
        try:
            audio, sr = sf.read(result.accepted_audio.path, dtype='float32')
            with np.load(directory / 'features.npz') as archive:
                before = {key: archive[key] for key in archive.files}
            times = before['times_seconds']
            views, reliable, protected, metadata = runtime_views(before, fine_pitch(audio, sr, times))
            proposal = propose(views, model, payload)
            curve = np.asarray(proposal['requested_curve'], dtype=np.float64)
            predictions = np.asarray(proposal['operating_predictions'], dtype=np.float64)
            reliable = reliable & np.asarray(proposal['reliable'], dtype=bool)
            if curve.shape != times.shape or predictions.shape != (3, len(times)) or not np.isfinite(curve).all():
                raise ValueError('Learned model returned invalid aligned proposals.')
            if np.any(curve[~reliable] != 0):
                raise ValueError('Learned proposal requests an edit outside reliable scope.')
            disagreement = np.asarray(proposal['disagreement'], dtype=np.float64)
            plan = plan_fixture(audio, sr, times, curve, predictions[1], disagreement, reliable, protected)
            def renderer(candidate_audio, rate, intervals):
                return render_candidate(candidate_audio, rate, times, curve, intervals)[0]
            def auditor(original, candidate, rate, intervals):
                after = extract(candidate, rate, COMPONENT_ROOT / 'checkpoints')[0]
                return audit_curve(before, after, fine_hz(original, rate, times), fine_hz(candidate, rate, times), reliable, intervals, curve)
            output, transaction = transact_fixture(audio, sr, plan, renderer, auditor)
            report.update(outcome=transaction['outcome'], permission=plan, transaction=transaction,
                          proposed_intervals_seconds=plan['intervals_seconds'], input_metadata=metadata,
                          scope_metadata=proposal.get('scope_metadata', {}))
            if transaction['outcome'] == 'VERIFIED_FIXTURE_CANDIDATE':
                candidate = directory / 'experimental_learned_candidate.wav'
                sf.write(candidate, output, sr, subtype='PCM_24')
                transaction['pre_encoding_audit'] = transaction['post_render_audit']
                try:
                    delivered = audit_delivered_wav(candidate, audio, sr, plan['intervals_seconds'], auditor)
                except Exception as exc:
                    transaction.update(outcome='ROLLED_BACK', rollback=True,
                                       delivered_wav_audit=dict(passed=False, reason=str(exc)))
                    raise
                transaction['post_render_audit'] = delivered['acoustic_audit']
                transaction['delivered_wav_audit'] = delivered
                report['candidate'] = dict(path=str(candidate), sha256=delivered['sha256'])
            np.savez_compressed(directory / 'model_predictions.npz', times=times, predictions=predictions,
                                proposed_correction=curve, reliable=reliable, protected=protected)
            validate_package(result)
        except Exception as exc:
            report.pop('candidate', None)
            report.update(outcome='LEARNED_PREVIEW_FAILED_UNCHANGED', error_type=type(exc).__name__, reason=str(exc))
    (directory / 'model_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return directory, report
