"""Recreate canonical working PCM and assess retained training-only short support."""
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

import probe_training_frontend_parity as run
from bpdr_input.experiments.learning import model_recovery as recovery
from bpdr_input.repair.rule_correction import segment_notes, RulePolicy
from prepare_short_phrase_training import save_analysis


def main():
    torch.set_num_threads(2)
    run.check_lock()
    root = run.ROOT
    d = run.read(root / 'declaration.json')
    results = run.read(root / 'results.json')
    assert d['source_sha256'] == run.sha(Path(run.__file__))
    assert d['frontend_sha256'] == run.sha(run.pipeline.__file__)
    assert results['declaration_sha256'] == run.sha(root / 'declaration.json')
    sources = {r['source_id']: r for r in d['sources']}
    reports = []
    for record in results['records']:
        source = sources[record['source_id']]
        assert run.sha(source['path']) == source['sha256'] == record['original_source_sha256']
        native, sr = sf.read(source['path'], dtype='float32', always_2d=True)
        channel = int(np.argmax(np.mean(native.astype(np.float64)**2, axis=0)))
        crop = native[:min(len(native), 10 * sr), channel]
        clipped = float(np.mean(abs(crop) >= .999))
        assert clipped == record['native_at_or_above_0999_fraction']
        assert clipped <= d['policy']['max_clipped_fraction'] and abs(crop).max() <= 1
        div = gcd(sr, 22050)
        expected = resample_poly(crop, 22050 // div, sr // div) if sr != 22050 else crop.copy()
        length = round(len(crop) * 22050 / sr)
        expected = expected[:length]
        if len(expected) < length:
            expected = np.pad(expected, (0, length - len(expected)))
        peak = float(abs(expected).max())
        gain = .999 / peak if peak > 1 else 1.
        expected *= gain
        folder = root / record['source_id']
        recreated = folder / 'independently_recreated_working.wav'
        sf.write(recreated, expected, 22050, subtype='PCM_24')
        assert run.sha(recreated) == record['accepted_working_sha256'] == run.sha(record['accepted_working_path'])
        diagnostics = run.read(record['diagnostics_path'])
        assert run.sha(diagnostics['source']['preserved_path']) == diagnostics['source']['sha256']
        operations = [op for op in diagnostics['operations'] if op['operation'] == 'peak_protection_gain']
        assert len(operations) == int(gain != 1.)
        if operations:
            assert operations[0]['gain'] == gain
        accepted, rate = sf.read(record['accepted_working_path'], dtype='float32')
        assert rate == 22050 and len(accepted) == length and abs(accepted).max() < 1
        target_duration = len(crop) / sr >= 5.
        assert target_duration == record['meets_five_second_target']
        row = dict(source_id=record['source_id'], domain=record['domain'], native_clipped_fraction=clipped,
            existing_frontend_gain=gain, sample_count=length, target_duration_pass=target_duration,
            working_path=record['accepted_working_path'], working_sha256=record['accepted_working_sha256'])
        if target_duration:
            analysis = recovery.measured(accepted)
            path = folder / 'canonical_repair_analysis.npz'; save_analysis(path, analysis)
            regions = segment_notes(analysis['views'][0][:, 64], analysis['reliable'], RulePolicy(note_split_cents=70.))
            region = max(regions, key=lambda r: r[1] - r[0]) if regions else None
            row.update(analysis_path=str(path), analysis_sha256=run.sha(path), longest_region_frames=region,
                widths_with_150ms_context=[width for width in (.24, .28, .36, .50)
                    if region is not None and (region[1] - region[0]) * .02 >= width + .30 - 1e-12])
        reports.append(row)
        print('FRONTEND_PARITY_REPLAY_SOURCE', row['source_id'], row.get('widths_with_150ms_context'), flush=True)
    run.write(root / 'integrity_receipt.json', dict(working_files_exactly_recreated=len(reports),
        full_source_and_preserved_crop_hashes_checked=True, same_existing_policy=True,
        training_target_duration_passes=sum(r['target_duration_pass'] for r in reports),
        records=reports, results_sha256=run.sha(root / 'results.json'), verifier_source_sha256=run.sha(__file__),
        development_audio_accessed=False, unseen_audio_accessed=False, new_model_fits=0,
        research_training_eligible=False, production_eligible=False))
    print('FRONTEND_PARITY_INDEPENDENT_REPLAY_PASS', len(reports), flush=True)


if __name__ == '__main__':
    main()
