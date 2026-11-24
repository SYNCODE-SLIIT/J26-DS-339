"""Supplementary preservation outside known injected corruption, not edit permission."""
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import soundfile as sf
import train_dense_residual_transformer as residual
import compare_sequence_models as sequence
from bpdr_input.repair.rule_correction import fine_pitch

ROOT = residual.ROOT.parent / '2026-10-30_controlled_preservation'


def main():
    ex = residual.ex; d = residual.declare(); sequence.declare()
    folders = dict(reference=residual.BASE / 'arms/matched_baseline/training/median11',
                   residual=residual.ROOT / 'arm/training/median11',
                   bigru=sequence.ROOT / 'arms/bigru/training/median11',
                   tcn=sequence.ROOT / 'arms/tcn/training/median11')
    ROOT.mkdir(exist_ok=True)
    declaration = ROOT / 'declaration.json'
    if not declaration.exists():
        ex.write(declaration, dict(created_at=datetime.now(timezone.utc).isoformat(),
                 source_sha256=ex.sha(Path(__file__)), input_hashes=d['input_hashes'],
                 evidence_hashes={str(folder / 'evidence.json'): ex.sha(folder / 'evidence.json') for folder in folders.values()},
                 protocol='For each existing verified C output, compare samples outside full known injected support (including ramps) against C and A. Require C==A there before scoring. Measure supplementary fine-pitch deviation on known-zero reliable frames beyond100ms analysis guard. Identity/rollback deliver C unchanged; exclusions retained.',
                 meaning='Retrospective controlled-preservation diagnostic. Exact sample change outside a known intervention is an alteration count, not independent proof of audible harm. Pitch measures are descriptive, not a new tuned threshold or human approval. Historical fixture verification guarantees outside PROPOSED edits, a different region.',
                 selection_eligible=False, demo_activated=False, research_validation_passed=False))
    locked = ex.read(declaration)
    assert locked['source_sha256'] == ex.sha(Path(__file__))
    for group in ('input_hashes', 'evidence_hashes'):
        for path, digest in locked[group].items():
            assert ex.sha(path) == digest, path
    if (ROOT / 'results.json').exists():
        raise ValueError('Retain completed diagnostic')
    rows = ex.read(residual.ROOT / 'train_pairs.json'); pairs = residual.training.held.prepared(rows)
    results = {}
    for name, folder in folders.items():
        accuracy = ex.read(folder / 'accuracy.json')
        files = {ex.sha(p): p for p in (folder / 'audio').rglob('*_candidate.wav')}
        cases, count, alterations = [], 0, 0
        for index, pair in enumerate(pairs):
            if 'excluded' in pair:
                cases.append(dict(index=index, excluded=pair['excluded']))
                continue
            before, clean = pair['audio_C'], pair['audio_A']
            lo, hi = pair['support_seconds']; sample_times = np.arange(len(before)) / 22050.
            outside = (sample_times < lo) | (sample_times > hi)
            np.testing.assert_array_equal(before[outside], clean[outside])
            trace = accuracy['rows'][index]['trace']['C']
            output = before; pitch = pair['C']['fine']; path = None
            if trace['outcome'] == 'VERIFIED_FIXTURE_CANDIDATE':
                digest = trace['transaction']['delivered_wav_audit']['sha256']; path = files[digest]
                output, rate = sf.read(path, dtype='float32')
                assert rate == 22050 and output.shape == before.shape
                pitch = fine_pitch(output, rate, pair['C']['features']['times_seconds'])
                count += 1
            changed = (output != before) & outside
            alterations += int(changed.any())
            times = pair['C']['features']['times_seconds']
            frame_mask = (pair['mask'] & (pair['target'] == 0) & pair['A']['reliable'] & pair['C']['reliable']
                          & ((times < lo - .1) | (times > hi + .1)))
            if not frame_mask.any():
                raise ValueError('No known reliable outside reference; do not silently drop a case')
            before_error = np.abs(pair['C']['fine'][frame_mask] - pair['A']['fine'][frame_mask])
            after_error = np.abs(pitch[frame_mask] - pair['A']['fine'][frame_mask])
            assert np.isfinite(before_error).all() and np.isfinite(after_error).all()
            delta = output[outside].astype(float) - before[outside].astype(float)
            cases.append(dict(index=index, group=pair['group'], outcome=trace['outcome'],
                              output_path=str(path) if path else None,
                              outside_sample_count=int(outside.sum()), changed_outside_samples=int(changed.sum()),
                              changed_outside_seconds=float(changed.sum() / 22050.),
                              outside_waveform_rmse=float(np.sqrt(np.mean(delta ** 2))),
                              outside_waveform_max_abs=float(np.max(abs(delta))),
                              outside_pitch_before_mae_cents=float(before_error.mean()),
                              outside_pitch_after_mae_cents=float(after_error.mean()),
                              outside_pitch_after_p90_cents=float(np.quantile(after_error, .9)),
                              outside_pitch_reference_frames=int(frame_mask.sum())))
        assessed = [r for r in cases if 'excluded' not in r]
        groups = sorted({r['group'] for r in assessed})
        macro = lambda key: float(np.mean([np.mean([r[key] for r in assessed if r['group'] == g]) for g in groups]))
        results[name] = dict(attempted=len(rows), scoreable=len(assessed), excluded=len(rows) - len(assessed),
                             verified_outputs=count, cases_with_changed_samples_outside_injection=alterations,
                             singer_macro_outside_pitch_before_mae_cents=macro('outside_pitch_before_mae_cents'),
                             singer_macro_outside_pitch_after_mae_cents=macro('outside_pitch_after_mae_cents'),
                             cases=cases)
        ex.write(ROOT / (name + '.json'), results[name])
        print('CONTROLLED_PRESERVATION', name, {k: v for k, v in results[name].items() if k != 'cases'}, flush=True)
    ex.write(ROOT / 'results.json', dict(results=results, declaration_sha256=ex.sha(declaration),
             meaning=locked['meaning'], research_validation_passed=False))


if __name__ == '__main__':
    main()
