"""Independently replay privileged-label and outside-injection diagnostics."""
from collections import Counter
from pathlib import Path
import hashlib
import numpy as np
import soundfile as sf
import train_dense_residual_transformer as study
from bpdr_input.repair.repair_controller import decision_hash
from bpdr_input.repair.rule_correction import fine_pitch


def main():
    ex = study.ex; study.declare(); root = study.ROOT.parent
    pairs = study.training.held.prepared(ex.read(study.ROOT / 'train_pairs.json'))
    oracle_root = root / '2026-10-30_known_label_wav_diagnostic'
    oracle = ex.read(oracle_root / 'results.json'); counts = Counter()
    assert oracle['NOT_MODEL_PERFORMANCE'] and not oracle['selection_eligible']
    assert oracle['declaration_sha256'] == ex.sha(oracle_root / 'declaration.json')
    declaration = ex.read(oracle_root / 'declaration.json')
    assert declaration['source_sha256'] == ex.sha(study.WORKSPACE / 'tools/render_known_label_diagnostic.py')
    assert declaration['preflight_sha256'] == ex.sha(root / '2026-10-30_known_label_diagnostic/results.json')
    for path, digest in declaration['input_hashes'].items():
        assert ex.sha(path) == digest
    files = {ex.sha(p): p for p in (oracle_root / 'audio').rglob('*_candidate.wav')}
    for row, pair in zip(oracle['cases'], pairs, strict=True):
        if 'excluded' in pair:
            assert 'excluded' in row
            continue
        plan, transaction = row['permission'], row['transaction']
        assert decision_hash(plan) == plan['decision_sha256']
        counts[transaction['outcome']] += 1
        if transaction['outcome'] != 'VERIFIED_FIXTURE_CANDIDATE':
            assert row['after_mae_cents'] == row['before_mae_cents'] and not row['improved']
            continue
        audit = transaction['delivered_wav_audit']; acoustic = audit['acoustic_audit']
        assert audit['passed'] and audit['outside_support_unchanged'] and acoustic['passed']
        assert acoustic['fine_wrong_direction_frames'] == acoustic['coarse_wrong_direction_frames'] == 0
        output, rate = sf.read(files[audit['sha256']], dtype='float32')
        assert rate == 22050 and output.shape == pair['audio_C'].shape
        assert hashlib.sha256(output.tobytes()).hexdigest() == audit['decoded_waveform_sha256']
        outside = np.ones(len(output), bool)
        for lo, hi in plan['intervals_seconds']:
            outside[int(round(lo * rate)):int(round(hi * rate))] = False
        np.testing.assert_array_equal(output[outside], pair['audio_C'][outside])
        mask = pair['mask'] & (pair['target'] != 0)
        pitch = fine_pitch(output, rate, pair['C']['features']['times_seconds'])
        after = float(abs(pitch[mask] - pair['A']['fine'][mask]).mean())
        np.testing.assert_allclose(after, row['after_mae_cents'], atol=1e-9, rtol=1e-9)
        improved = after < row['before_mae_cents'] - 1e-6
        assert improved == row['improved']
        counts['delivered_reference_improvement'] += int(improved)
    assert dict(counts) == oracle['outcomes']
    preservation_root = root / '2026-10-30_controlled_preservation'
    preservation = ex.read(preservation_root / 'results.json')
    locked = ex.read(preservation_root / 'declaration.json')
    assert preservation['declaration_sha256'] == ex.sha(preservation_root / 'declaration.json')
    assert locked['source_sha256'] == ex.sha(study.WORKSPACE / 'tools/audit_controlled_preservation.py')
    for group in ('input_hashes', 'evidence_hashes'):
        for path, digest in locked[group].items():
            assert ex.sha(path) == digest
    reports = {}
    for name, result in preservation['results'].items():
        altered = verified = 0
        for row, pair in zip(result['cases'], pairs, strict=True):
            if 'excluded' in pair:
                assert 'excluded' in row
                continue
            lo, hi = pair['support_seconds']; times = np.arange(len(pair['audio_C'])) / 22050.
            outside = (times < lo) | (times > hi)
            np.testing.assert_array_equal(pair['audio_C'][outside], pair['audio_A'][outside])
            if row['output_path'] is None:
                assert row['changed_outside_samples'] == 0
                continue
            output, rate = sf.read(row['output_path'], dtype='float32')
            assert rate == 22050 and output.shape == pair['audio_C'].shape
            changed = int(np.sum((output != pair['audio_C']) & outside)); verified += 1
            assert changed == row['changed_outside_samples']
            assert int(outside.sum()) == row['outside_sample_count']
            np.testing.assert_allclose(changed / rate, row['changed_outside_seconds'], atol=1e-12)
            altered += int(changed > 0)
        assert altered == result['cases_with_changed_samples_outside_injection']
        assert verified == result['verified_outputs']
        reports[name] = dict(verified_outputs=verified, changed_outside_injection=altered)
    protected = ex.read(root / '2026-10-28_v7_guard/declaration.json')['protected_files']
    for path, digest in protected.items():
        assert ex.sha(path) == digest
    assert not (root / 'active_model.json').exists()
    receipt = dict(known_label_encoded_wavs_verified=counts['VERIFIED_FIXTURE_CANDIDATE'],
                   controlled_preservation=reports, protected_files_unchanged=len(protected),
                   NOT_MODEL_PERFORMANCE=True, selection_eligible=False, activation_absent=True,
                   oracle_results_sha256=ex.sha(oracle_root / 'results.json'),
                   preservation_results_sha256=ex.sha(preservation_root / 'results.json'),
                   verifier_sha256=ex.sha(Path(__file__)), research_validation_passed=False)
    ex.write(oracle_root / 'integrity_receipt.json', receipt)
    ex.write(preservation_root / 'integrity_receipt.json', receipt)
    print('FEASIBILITY_DIAGNOSTICS_VERIFIED', receipt, flush=True)


if __name__ == '__main__':
    main()
