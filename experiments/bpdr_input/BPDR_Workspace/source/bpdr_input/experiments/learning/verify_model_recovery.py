"""Replay the locked pilot through the live audio adapter before demo adoption.

References and intervention masks are read only after audio-only inference.
Withheld/rolled-back proposals are scored using the retained input, never omitted.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import time

import numpy as np
import soundfile as sf

from bpdr_input import COMPONENT_ROOT
from bpdr_input.audio.pipeline import checksum
from bpdr_input.modeling.recovery_model import load_candidate, load_locked_diagnostic
from bpdr_input.repair.demo_runtime import selected_output
from bpdr_input.repair.learned_runtime import run_candidate, CARD
from bpdr_input.repair.rule_correction import fine_pitch

WORKSPACE = COMPONENT_ROOT / 'BPDR_Workspace'
ROOT = WORKSPACE / 'results/model_recovery'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run(folder, diagnostic=False):
    import torch
    torch.set_num_threads(2)
    folder = Path(folder).resolve()
    if not folder.is_relative_to(ROOT.resolve()):
        raise ValueError('Candidate evidence must remain in the model recovery folder')
    checkpoint = folder / 'selected_model.joblib'
    evidence = read(folder / 'evidence.json')
    model, payload = load_locked_diagnostic(checkpoint) if diagnostic else load_candidate(checkpoint, require_rendered=False)
    sources = read(folder / 'pilot_evaluation_sources.json')
    pairs = read(folder / 'pilot_evaluation_pairs.json')
    declaration = read(folder / 'declaration.json')
    manifest_path = WORKSPACE / 'datasets/vocalset_pilot/manifest.json'
    if checksum(manifest_path) != declaration['manifest_sha256']:
        raise ValueError('Dataset manifest changed after the declared experiment')
    manifest = read(manifest_path)
    expected_hashes = {r['sha256'] for r in manifest['records'] if r['split'] == 'pilot_evaluation'}
    source_hashes = [s['source']['sha256'] for s in sources]
    if len(set(source_hashes)) != 4 or set(source_hashes) != expected_hashes:
        raise ValueError('Outer original catalog must contain each declared source exactly once')
    expected_pairs = {(digest, shift) for digest in expected_hashes for shift in (-30., -20., 20., 30.)}
    pair_keys = [(p['source']['sha256'], float(p['cents'])) for p in pairs]
    if len(pair_keys) != 16 or len(set(pair_keys)) != 16 or set(pair_keys) != expected_pairs:
        raise ValueError('Outer intervention catalog must retain all sixteen unique signed attempts')
    expected_singers = sorted({s['source']['singer_id'] for s in sources})
    if len(expected_singers) != 2 or len(sources) != 4 or any(sum(s['source']['singer_id'] == singer for s in sources) != 2 for singer in expected_singers):
        raise ValueError('The complete locked two-singer outer pilot is required')
    meta = dict(checkpoint=str(checkpoint), checkpoint_sha256=checksum(checkpoint),
                scope=payload['scope'], metrics={})
    started = time.perf_counter()
    original_rows, shifted_rows = [], []
    for source in sources:
        if 'excluded' in source and not source.get('working_audio'):
            original_rows.append(dict(singer=source['source']['singer_id'], excluded=source['excluded'], preserved=False))
            continue
        path = Path(source['working_audio'])
        if checksum(path) != source['working_sha256']:
            raise ValueError('Outer original working waveform changed')
        directory, report = run_candidate(path, model, payload, meta, output_root=folder / 'rendered')
        output, label = selected_output(report)
        a, ar = sf.read(path, dtype='float32')
        b, br = sf.read(output, dtype='float32') if output else (np.empty(0), 0)
        preserved = bool(label == 'Original retained' and ar == br and np.array_equal(a, b))
        original_rows.append(dict(singer=source['source']['singer_id'], input=str(path), output=str(output),
            directory=str(directory), outcome=report['outcome'], preserved=preserved,
            intervention_exclusion=source.get('excluded')))
        print('ORIGINAL_REPLAY', source['source']['singer_id'], preserved, report['outcome'], flush=True)
    for pair in pairs:
        singer = pair.get('group') or pair['source']['singer_id']
        if 'excluded' in pair:
            shifted_rows.append(dict(singer=singer, excluded=pair['excluded'], cents=pair['cents']))
            continue
        # Only input waveform and fitted model enter the processing adapter.
        directory, report = run_candidate(pair['path_C'], model, payload, meta, output_root=folder / 'rendered')
        output, label = selected_output(report)
        reference, sr = sf.read(pair['path_A'], dtype='float32')
        before, before_sr = sf.read(pair['path_C'], dtype='float32')
        after, after_sr = sf.read(output, dtype='float32') if output else (np.empty(0), 0)
        if sr != before_sr or sr != after_sr or before.shape != reference.shape or after.shape != reference.shape:
            raise ValueError('Delivered pilot waveform changed its sample grid')
        with np.load(pair['target_path']) as targets:
            times = targets['times']
            mask = targets['mask'].astype(bool) & (targets['target'] != 0)
        ref_pitch = fine_pitch(reference, sr, times)
        before_error = fine_pitch(before, sr, times) - ref_pitch
        after_error = fine_pitch(after, sr, times) - ref_pitch
        if not mask.any() or not np.isfinite(before_error[mask]).all():
            raise ValueError('Declared intervention score frames lack valid input pitch')
        # The delivered signal cannot choose an easier scoring denominator.
        missing_after = int(np.count_nonzero(~np.isfinite(after_error[mask])))
        input_mae = float(np.abs(before_error[mask]).mean())
        measured_output_mae = float(np.abs(after_error[mask]).mean()) if not missing_after else None
        output_mae = measured_output_mae if measured_output_mae is not None else input_mae
        row = dict(singer=singer, input=pair['path_C'], reference=pair['path_A'],
            target_path=pair['target_path'], output=str(output), directory=str(directory), cents=pair['cents'],
            outcome=report['outcome'], delivered_preview=label == 'Processed preview',
            scored_frames=int(mask.sum()), input_mae_cents=input_mae, output_mae_cents=output_mae,
            missing_output_scored_frames=missing_after, measured_output_mae_cents=measured_output_mae,
            error_reduction_cents=input_mae-output_mae, verified_improvement=bool(
                label == 'Processed preview' and not missing_after and output_mae < input_mae),
            controller_reasons=report.get('permission', {}).get('reasons', []))
        shifted_rows.append(row)
        write(folder / 'rendered_progress.json', dict(originals=original_rows, shifts=shifted_rows))
        print('SHIFT_REPLAY', singer, pair['cents'], report['outcome'], round(input_mae, 3), round(output_mae, 3), flush=True)
    by_singer = {}
    for singer in expected_singers:
        all_rows = [r for r in shifted_rows if r['singer'] == singer]
        valid = [r for r in all_rows if 'excluded' not in r]
        before = float(np.mean([r['input_mae_cents'] for r in valid])) if valid else None
        after = float(np.mean([r['output_mae_cents'] for r in valid])) if valid else None
        by_singer[singer] = dict(attempted=len(all_rows), audited=len(valid),
            delivered_previews=sum(r['delivered_preview'] for r in valid),
            verified_improvements=sum(r['verified_improvement'] for r in valid),
            input_mae_cents=before, output_mae_cents=after,
            error_reduction_fraction=(1-after/before) if before else None)
    rendered_passes = (bool(evidence['outer_proposal_gate_passes']) and len(by_singer) == 2
        and all(r['preserved'] for r in original_rows)
        and all(not r.get('missing_output_scored_frames') for r in shifted_rows)
        and all(g['verified_improvements'] >= 1 and g['error_reduction_fraction'] > 0 for g in by_singer.values()))
    verification = dict(version='delivered-vocalset-recovery-pilot-1', checkpoint_sha256=checksum(checkpoint),
        declaration_sha256=checksum(folder / 'declaration.json'),
        originals=original_rows, shifts=shifted_rows, by_singer=by_singer,
        rendered_demo_gate_passes=rendered_passes,
        diagnostic_only=diagnostic,
        attempted_shift_count=len(pairs), audited_shift_count=sum('excluded' not in p for p in pairs),
        elapsed_seconds=round(time.perf_counter()-started, 3),
        interpretation='Audited interventions only have valid labels; every withheld/rolled-back audited input is scored unchanged. Declared scored frames are fixed; missing delivered pitch receives no gain and fails adoption. Exclusions and all four original anchors are reported. Two heldout singers; no natural-error accuracy estimate.')
    write(folder / 'rendered_verification.json', verification)
    evidence.update(rendered_demo_gate_passes=rendered_passes,
        rendered_verification=str(folder / 'rendered_verification.json'),
        rendered_verification_sha256=checksum(folder / 'rendered_verification.json'),
        actual_render_verification=verification['interpretation'], model_adopted=rendered_passes,
        adoption_scope='Separate scoped demo only; historical research model and handoff policy preserved')
    write(folder / 'evidence.json', evidence)
    if rendered_passes:
        valid = [r for r in shifted_rows if 'excluded' not in r]
        metrics = dict(model_type=payload['model_type'],
            heldout_singers=len(expected_singers), attempted_shifts=len(pairs), audited_shifts=len(valid),
            delivered_previews=sum(r['delivered_preview'] for r in valid),
            verified_improvements=sum(r['verified_improvement'] for r in valid),
            originals_preserved=sum(r['preserved'] for r in original_rows), originals_tested=len(original_rows),
            input_mae_cents=float(np.mean([g['input_mae_cents'] for g in by_singer.values()])),
            output_mae_cents=float(np.mean([g['output_mae_cents'] for g in by_singer.values()])),
            by_singer=by_singer)
        metrics['error_reduction_fraction'] = 1-metrics['output_mae_cents']/metrics['input_mae_cents']
        # Deterministic first successful example in locked source/signed-shift order;
        # whole-pilot outcomes remain visible beside the selected listening example.
        shifted = next(r for r in valid if r['verified_improvement'])
        original = next(r for r in original_rows if r['input'] == shifted['reference'])
        cases = [dict(id='learned_pitch', label='Learned repair: real held-out singer with a controlled local shift',
            input=shifted['input'], reference=shifted['reference'], score_mask_path=shifted['target_path'],
            source_kind='VocalSet real human recording; known shift added for this selected verified listening example. Reference and mask are used only for post-inference scoring.'),
            dict(id='learned_original', label='Preservation: unchanged real held-out VocalSet singer',
            input=original['input'], reference=None,
            source_kind='Unshifted working derivative of a real VocalSet recording, from the same held-out singer. The model should preserve this performance.')]
        write(CARD, dict(version='vocalset-demo-model-1', eligible_for_demo=True, **{k:v for k,v in meta.items() if k != 'metrics'},
            metrics=metrics, operating_policy=payload['operating_policy'],
            demo_cases=cases, example_selection='First verified improvement in prospectively locked source and signed-shift order',
            evidence=str(folder / 'evidence.json'), verification_sha256=checksum(folder / 'rendered_verification.json')))
        print('DEMO_MODEL_ENABLED', CARD, flush=True)
    print(json.dumps(verification, indent=2), flush=True)
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--diagnostic', action='store_true', help='Replay a failed locked candidate honestly; does not override adoption gates')
    args = parser.parse_args()
    run(args.folder, diagnostic=args.diagnostic)
