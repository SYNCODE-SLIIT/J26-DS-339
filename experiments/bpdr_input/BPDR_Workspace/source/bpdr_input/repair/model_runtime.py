"""Local experimental inference on one WAV/FLAC, without annotations or targets.

The accepted downstream package remains unchanged by owner decision. A rendered
candidate is a separate, numerically verified engineering preview only.
"""
from bpdr_input import COMPONENT_ROOT
import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

COMPONENT = COMPONENT_ROOT


def default_checkpoint():
    from bpdr_input.audio.pipeline import checksum
    working=COMPONENT/'BPDR_Workspace/selected_model/model.json'
    if working.exists():
        card=json.loads(working.read_text())
        checkpoint=working.parent/card['checkpoint']
        if checksum(checkpoint)!=card['sha256']:
            raise ValueError('Working checkpoint checksum mismatch')
        return checkpoint
    current=COMPONENT/'runs/presentation_release/current.json'
    if current.exists():
        release=Path(json.loads(current.read_text())['path'])
        card=json.loads((release/'model_card.json').read_text())
        checkpoint=release/card['checkpoint']
        if checksum(checkpoint)!=card['checkpoint_sha256']:
            raise ValueError('Presentation checkpoint checksum mismatch')
        return checkpoint
    root = COMPONENT/'runs/controlled_study'
    run = Path(json.loads((root/'training_current.json').read_text())['path'])
    return run/'bpdr.pt'


def runtime_views(features, fine):
    """Three timing-only boundary views, obtained solely from this input audio."""
    from scipy.signal import find_peaks
    from bpdr_input.preparation.pairs.shared_pair_audit import game_boundary_track
    times = features['times_seconds']
    # Availability comes from voicing, never pitch != 0: A4 is a valid 0 cents.
    # Historical training archives remain frozen with their legacy adapter.
    raw = np.asarray(features['boundary_features_71'],dtype=np.float32).copy()
    original_voiced=raw[:,66]>.5
    agree=original_voiced & np.isfinite(fine) & (np.abs(fine-raw[:,64])<=50.)
    dropped=int((original_voiced & ~agree).sum())
    raw[:,64]=np.where(agree,fine,0.)
    raw[:,66]=np.where(agree,raw[:,66],0.)
    raw[:,67]=0.
    raw[1:,67]=np.where(agree[1:] & agree[:-1],np.diff(fine),0.)
    peaks, _ = find_peaks(features['provisional_boundary_score'], height=.5, distance=5)
    edges = times[peaks]
    # Fixed, documented perturbations; not calibrated uncertainty estimates.
    alternatives = (edges, np.clip(edges+.04, 0, times[-1]), edges[::2])
    views = []
    for e in alternatives:
        x = raw.copy(); x[:, 70] = game_boundary_track(times, e)
        if agree.any():x[:,64]=np.where(agree,x[:,64]-np.median(x[agree,64]),0.)
        views.append(x)
    protected = np.zeros(len(times), bool)
    for edge in edges:
        protected |= np.abs(times-edge) <= .10
    return views, raw[:, 66] > .5, protected, dict(
        boundary_source='Input-audio heuristic peaks >= .5, minimum 100 ms apart; Gaussian timing track',
        alternate_boundaries='Nominal, +40-ms jitter, every second edge removed',
        uncertainty='Maximum prediction disagreement, not calibrated probability',
        pitch_adapter='runtime-fine-v2: explicit voicing, including valid A4=0 cents; historical study adapters remain frozen',
        fine_pitch_guard_dropped_frames=dropped, nominal_edges_seconds=edges.tolist(),
        protected_transition_seconds=.10)


def run(source, output_root=None, checkpoint=None, render_preview=False):
    import torch
    from bpdr_input.audio.pipeline import process, checksum, validate_package
    from bpdr_input.audio.analysis import extract
    from bpdr_input.reporting.bpdr_showcase import load_model, correction_from_prediction
    from bpdr_input.repair.framewise_audit import audit_curve
    from bpdr_input.repair.framewise_renderer import render_candidate
    from bpdr_input.repair.repair_controller import plan_fixture, transact_fixture
    from bpdr_input.repair.rule_correction import fine_pitch, fine_hz
    started = time.perf_counter()
    torch.set_num_threads(2)
    result, directory = process(Path(source), output_root or COMPONENT/'runs/model_runtime', COMPONENT/'checkpoints')
    report = dict(version='experimental-model-runtime-1', source=str(source), handoff=result.to_dict(),
                  model_status='controlled_proof_of_concept_real_repair_unvalidated', prepared_handoff_enabled=False,
                  output_scope='Local preview; no automatic training/research reuse or musical approval',
                  noise_policy='No attenuation: eligible noise-only reference and preservation validation not implemented')
    if result.accepted_audio is None:
        report.update(outcome='NOT_RUN_TECHNICAL_STOP', elapsed_seconds=time.perf_counter()-started)
    else:
        try:
            model_path = Path(checkpoint or default_checkpoint())
            before_model_hash = checksum(model_path)
            model, normalizer, width = load_model(model_path)
            audio, sr = sf.read(result.accepted_audio.path, dtype='float32')
            with np.load(directory/'features.npz') as z:
                features = {k: z[k] for k in z.files}
            times = features['times_seconds']
            views, reliable, protected, metadata = runtime_views(features, fine_pitch(audio, sr, times))
            x = np.stack([normalizer(v[:, :width]) for v in views])
            with torch.inference_mode():
                pred = model(torch.from_numpy(x)).numpy()
            if not np.isfinite(pred).all():
                raise ValueError('NONFINITE_MODEL_OUTPUT')
            disagreement = np.max(np.abs(pred[1:]-pred[0]), axis=0)
            curve = correction_from_prediction(pred[0], pred[1], reliable)
            plan = plan_fixture(audio, sr, times, curve, pred[1], disagreement, reliable, protected)
            transaction = dict(outcome='PREDICTION_ONLY', reasons=plan['reasons'])
            if render_preview:
                def renderer(a, rate, intervals):
                    return render_candidate(a, rate, times, curve, intervals)[0]
                def auditor(a, b, rate, intervals):
                    after = extract(b, rate, COMPONENT/'checkpoints')[0]
                    return audit_curve(features, after, fine_hz(a, rate, times), fine_hz(b, rate, times), reliable, intervals, curve)
                output, transaction = transact_fixture(audio, sr, plan, renderer, auditor)
                if transaction['outcome'] == 'VERIFIED_FIXTURE_CANDIDATE':
                    sf.write(directory/'experimental_candidate.wav', output, sr, subtype='PCM_24')
                    report['candidate'] = dict(path=str(directory/'experimental_candidate.wav'),
                                               sha256=checksum(directory/'experimental_candidate.wav'))
            np.savez_compressed(directory/'model_predictions.npz', times=times, predictions=pred,
                                proposed_correction=curve, reliable=reliable, protected=protected)
            report.update(outcome=transaction['outcome'], model_path=str(model_path), model_sha256=before_model_hash,
                          input_metadata=metadata, permission=plan, transaction=transaction,
                          mean_abs_prediction_cents=float(np.abs(pred[0][reliable]).mean()) if reliable.any() else None,
                          proposed_intervals_seconds=plan['intervals_seconds'])
            if checksum(model_path) != before_model_hash:
                raise ValueError('MODEL_CHANGED_DURING_INFERENCE')
        except Exception as exc:
            # A model/renderer failure cannot destroy a technically accepted input.
            report.update(outcome='MODEL_PREVIEW_FAILED_UNCHANGED', error_type=type(exc).__name__, reason=str(exc))
        validate_package(result)
        report['elapsed_seconds'] = time.perf_counter()-started
    (directory/'handoff.json').write_text(json.dumps(result.to_dict(), indent=2), encoding='utf-8')
    (directory/'model_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return directory, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path); p.add_argument('--checkpoint', type=Path)
    p.add_argument('--render-preview', action='store_true')
    args = p.parse_args()
    directory, report = run(args.source, checkpoint=args.checkpoint, render_preview=args.render_preview)
    print(json.dumps(dict(directory=str(directory), outcome=report['outcome'], handoff=report['handoff']['status']), indent=2))


if __name__ == '__main__':
    main()
