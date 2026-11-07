"""Replay all33 canonical working copies and all32 eligible acoustic contexts."""
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

import prepare_canonical_training_inputs as run
from verify_short_phrase_training import check_arrays


def main():
    torch.set_num_threads(2)
    run.check_lock()
    root = run.ROOT
    d = run.read(root / 'declaration.json')
    assert d['source_sha256'] == run.sha(run.__file__)
    assert d['frontend_sha256'] == run.sha(run.pipeline.__file__)
    assert d['context_source_sha256'] == run.sha(run.recovery.__file__)
    for path, expected in d['parent_hashes'].items():
        assert run.sha(path) == expected
    for path, expected in run.read(root / 'input_hashes.json').items():
        assert run.sha(path) == expected
    originals = run.read(root / 'train_originals.json')
    assert len(originals) == 33
    by_id = {s['source_id']: s for s in d['sources']}
    recreated = root / 'verification_recreated'; recreated.mkdir(exist_ok=True)
    changes, counts, arrays, eligible = [], 0, 0, 0
    for original in originals:
        source = by_id[original['source_id']]
        assert original['source'] == source and original['split'] == 'train'
        assert run.sha(source['native_path']) == source['native_sha256']
        native, sr = sf.read(source['native_path'], dtype='float32', always_2d=True)
        channel = int(np.argmax(np.mean(native.astype(np.float64)**2, axis=0)))
        crop = native[:min(len(native), 10 * sr), channel]
        assert original['source_channel'] == channel and original['crop_native_samples'] == len(crop)
        assert original['native_duration_seconds'] == len(crop) / sr
        assert np.isfinite(crop).all() and abs(crop).max() <= 1
        assert float(np.mean(abs(crop) >= .999)) <= d['policy']['max_clipped_fraction']
        divisor = gcd(sr, 22050)
        working = resample_poly(crop, 22050 // divisor, sr // divisor) if sr != 22050 else crop.copy()
        length = round(len(crop) * 22050 / sr)
        working = working[:length]
        if len(working) < length:
            working = np.pad(working, (0, length - len(working)))
        peak = float(abs(working).max()); gain = .999 / peak if peak > 1 else 1.
        working *= gain
        path = recreated / (original['source_id'] + '.wav')
        sf.write(path, working, 22050, subtype='PCM_24')
        assert run.sha(path) == original['canonical_working_sha256'] == run.sha(original['path_A'])
        diagnostics = run.read(original['canonical_diagnostics'])
        assert diagnostics['operations'] == original['canonical_operations']
        assert run.sha(diagnostics['source']['preserved_path']) == diagnostics['source']['sha256']
        preserved, preserved_rate = sf.read(diagnostics['source']['preserved_path'], dtype='float32')
        assert preserved_rate == sr
        np.testing.assert_array_equal(preserved, crop)
        gains = [op['gain'] for op in diagnostics['operations'] if op['operation'] == 'peak_protection_gain']
        assert gains == ([gain] if gain != 1. else [])
        counts += 1
        if len(crop) / sr < 5.:
            assert original['excluded'] == 'SOURCE_SHORTER_THAN_FIVE_SECOND_TARGET'
            assert 'analysis_A' not in original
            continue
        assert 'excluded' not in original
        audio, rate = sf.read(path, dtype='float32'); assert rate == 22050
        old = source['prior_original']
        same = 'excluded' not in old and run.sha(old['path_A']) == run.sha(path)
        assert original['identical_to_historical_working_pcm'] == same
        if not same:
            change = dict(source_id=original['source_id'], domain=source['domain'], gain=gain,
                prior_exclusion=old.get('excluded'), new_sample_count=len(audio))
            if 'excluded' not in old:
                previous, _ = sf.read(old['path_A'], dtype='float32')
                n = min(len(previous), len(audio))
                change.update(old_sample_count=len(previous), common_prefix_identical=bool(np.array_equal(previous[:n],audio[:n])))
            changes.append(change)
        a = run.recovery.measured(audio)
        arrays += check_arrays(original['analysis_A'], a)
        regions = run.segment_notes(a['views'][0][:,64],a['reliable'],run.RulePolicy(note_split_cents=70.))
        region = max(regions,key=lambda r:r[1]-r[0]) if regions else None
        assert run.norm(region) == original['longest_region_frames']
        widths = [w for w in (.24,.28,.36,.50) if region is not None and (region[1]-region[0])*.02>=w+.30-1e-12]
        assert widths == original['widths_with_150ms_context']
        eligible += 1
        print('CANONICAL_INPUT_REPLAY_SOURCE',original['source_id'],same,widths,flush=True)
    summary = run.read(root / 'summary.json')
    assert summary['valid'] == eligible == 32 and counts == 33 and len(changes) == 5
    assert summary['working_pcm_identical_to_prior'] == eligible-len(changes) == 27
    run.write(root / 'integrity_receipt.json',dict(canonical_working_copies_recreated=counts,
        eligible_contexts_reinferred=eligible,acoustic_arrays_reinferred=arrays,
        changed_or_new_working_copy_details=changes,original_native_and_crop_preservation_rechecked=True,
        declaration_sha256=run.sha(root/'declaration.json'),train_originals_sha256=run.sha(root/'train_originals.json'),
        input_hashes_sha256=run.sha(root/'input_hashes.json'),summary_sha256=run.sha(root/'summary.json'),
        verifier_source_sha256=run.sha(__file__),nontraining_audio_accessed=False,new_interventions=0,new_model_fits=0,
        research_training_eligible=False,production_eligible=False))
    print('CANONICAL_TRAINING_INPUT_INDEPENDENT_REPLAY_PASS',counts,eligible,arrays,flush=True)


if __name__=='__main__':
    main()
