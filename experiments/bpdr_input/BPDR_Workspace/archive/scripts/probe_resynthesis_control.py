"""Measure forced zero-shift overlap-add on a fixed sample of training cases."""

from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf

import compare_preservation_sampling as data_source
from bpdr_input.repair.interventions import envelope
from bpdr_input.repair.rule_correction import fine_pitch


ROOT = data_source.DATA.parent / '2026-11-06_resynthesis_control'
read, write, sha = data_source.read, data_source.write, data_source.sha


def compare(row, policy):
    clean, rate = sf.read(row['path_A'], dtype='float32')
    shifted, shifted_rate = sf.read(row['path_C'], dtype='float32')
    assert rate == shifted_rate == 22050 and clean.shape == shifted.shape
    assert parselmouth.__version__ == policy['parselmouth_version']
    sound = parselmouth.Sound(clean.astype(np.float64), sampling_frequency=rate, start_time=0.)
    manipulation = call(sound, 'To Manipulation', policy['time_step_seconds'],
                        policy['pitch_floor_hz'], policy['pitch_ceiling_hz'])
    forced = call(manipulation, 'Get resynthesis (overlap-add)')
    assert (forced.values.shape == (1, len(clean)) and forced.sampling_frequency == rate
            and (forced.xmin, forced.xmax, forced.x1) == (sound.xmin, sound.xmax, sound.x1))
    blend = envelope(np.arange(len(clean)) / rate, row['support_seconds'], policy['ramp_seconds'])
    sham = (clean * (1 - blend) + forced.values[0] * blend).astype(np.float32)
    assert np.isfinite(sham).all() and np.array_equal(clean[blend == 0], sham[blend == 0])
    times = np.arange(len(clean) // 441 + 1) * .02
    with np.load(row['target_path']) as archive:
        positive = archive['mask'] & (archive['target'] != 0)
    base = fine_pitch(clean, rate, times)
    support = blend > 0
    metrics = {}
    for label, candidate in [('zero_shift_resynthesis', sham), ('registered_shift', shifted)]:
        observed = fine_pitch(candidate, rate, times)
        common = positive & np.isfinite(base) & np.isfinite(observed)
        change = observed[common] - base[common]
        metrics[label] = dict(
            waveform_residual_rms=float(np.sqrt(np.mean((candidate[support] - clean[support]) ** 2))),
            exact_outside_support=bool(np.array_equal(candidate[~support], clean[~support])),
            common_pitch_frames=int(common.sum()),
            median_abs_pitch_change_cents=float(np.median(abs(change))) if len(change) else None,
        )
    return dict(positive_frames=int(positive.sum()), **metrics)


def main():
    manifest = data_source.DATA / 'train_pairs.json'
    receipt = data_source.DATA / 'integrity_receipt.json'
    assert read(receipt)['train_pairs_sha256'] == sha(manifest)
    selected = {}
    for index, row in enumerate(read(manifest)):
        if 'excluded' in row:
            continue
        assert row['split'] == 'train'
        domain = 'humming' if row['group'].startswith('HumTrans:') else 'vocadito'
        groups = selected.setdefault(domain, {})
        if len(groups) < 4:
            groups.setdefault(row['group'], (index, row))
    cases = [case for groups in selected.values() for case in groups.values()]
    assert 0 < len(cases) <= 8
    input_hashes = read(data_source.DATA / 'input_hashes.json')
    assert read(receipt)['input_hashes_sha256'] == sha(data_source.DATA / 'input_hashes.json')
    paths = {row[key] for _, row in cases for key in ('path_A', 'path_C', 'target_path')}
    for path in paths:
        assert sha(path) == input_hashes[path]
    renderer_declarations = {}
    for index, row in cases:
        if 'render_receipt' in row:
            renderer_declarations[index] = row['render_receipt']['policy']
        else:
            historical = Path(row['path_C']).parents[2] / 'declaration.json'
            renderer_declarations[index] = read(historical)['renderer']
    renderer = read(data_source.DATA / 'declaration.json')['renderer']
    assert all(policy == renderer for policy in renderer_declarations.values())
    declaration = dict(case_indices=[index for index, _ in cases], manifest_sha256=sha(manifest),
                       source_sha256=sha(Path(__file__)), renderer=renderer,
                       no_fit=True, production_eligible=False)
    ROOT.mkdir(exist_ok=True)
    declaration_path = ROOT / 'declaration.json'
    if declaration_path.exists():
        assert read(declaration_path) == declaration
    else:
        write(declaration_path, declaration)
    results = dict(declaration_sha256=sha(declaration_path), cases=[
        dict(index=index, group=row['group'], injected_cents=row['cents'], **compare(row, renderer))
        for index, row in cases
    ])
    path = ROOT / 'results.json'
    if path.exists():
        assert read(path) == results
        print('RESYNTHESIS_CONTROL_REPLAY_MATCH', len(cases), flush=True)
    else:
        write(path, results)
        print('RESYNTHESIS_CONTROL_COMPLETE', len(cases), flush=True)


if __name__ == '__main__':
    main()
