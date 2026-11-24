"""Trace the outlying held-source V68 response with fixed feature substitutions."""

from pathlib import Path

import numpy as np
import torch

import compare_tail_preservation as study
import probe_resynthesis_control as control
from bpdr_input.experiments.learning.model_recovery import measured


def main():
    torch.set_num_threads(2)
    read, write, sha = control.read, control.write, control.sha
    prior = control.ROOT / 'prediction_results.json'
    case, = [row for row in read(prior)['cases'] if row['group'] == 'S7']
    index = case['index']
    manifest = study.DATA / 'train_pairs.json'
    protocol = read(control.ROOT / 'declaration.json')
    assert sha(manifest) == protocol['manifest_sha256']
    row = read(manifest)[index]
    fold, = [i for i, groups in enumerate(read(study.ROOT / 'declaration.json')['folds'], 1)
             if row['group'] in groups]
    folder = study.ROOT / f'fold_{fold}/upper_tail_preservation'
    checkpoint = folder / 'checkpoint.pt'
    receipt = read(folder / 'fit_receipt.json')
    assert row['source_id'] in receipt['heldout_sources']
    assert sha(checkpoint) == receipt['checkpoint_sha256']
    input_hashes = read(study.DATA / 'input_hashes.json')
    for key in ('path_A', 'target_path'):
        assert sha(row[key]) == input_hashes[row[key]]
    groups = {'pitch_voicing_delta': [64, 66, 67], 'confidence_energy': [65, 68]}
    declaration = dict(case=index, groups=groups, hashes={
        str(path): sha(path) for path in (Path(__file__), prior, manifest, checkpoint)
    })
    path = control.ROOT / 'feature_trace_declaration.json'
    if path.exists():
        assert read(path) == declaration
    else:
        write(path, declaration)
    clean, rate = control.sf.read(row['path_A'], dtype='float32')
    renderer = protocol['renderer']
    sound = control.parselmouth.Sound(clean.astype(np.float64), sampling_frequency=rate)
    manipulation = control.call(sound, 'To Manipulation', renderer['time_step_seconds'],
                                renderer['pitch_floor_hz'], renderer['pitch_ceiling_hz'])
    forced = control.call(manipulation, 'Get resynthesis (overlap-add)')
    assert forced.values.shape == (1, len(clean)) and forced.x1 == sound.x1
    blend = control.envelope(np.arange(len(clean)) / rate, row['support_seconds'],
                             renderer['ramp_seconds'])
    sham = (clean * (1 - blend) + forced.values[0] * blend).astype(np.float32)
    assert np.array_equal(sham[blend == 0], clean[blend == 0])
    base, changed = [measured(audio)['views'][0] for audio in (clean, sham)]
    model, _ = study.model_module.load(checkpoint)
    with np.load(row['target_path']) as archive:
        mask = archive['mask'] & (archive['target'] != 0)
    conditions = {'clean': base, 'zero_shift': changed}
    for label, columns in groups.items():
        hybrid = base.copy()
        hybrid[:, columns] = changed[:, columns]
        conditions[label] = hybrid
    requests = {label: study.model_module.predict_raw(model, raw)[0][mask]
                for label, raw in conditions.items()}
    deltas = {label: float(np.mean(abs(values - requests['clean'])))
              for label, values in requests.items()}
    assert deltas['zero_shift'] == case['zero_shift']['mean_abs_change_from_clean']
    report = dict(declaration_sha256=sha(path), mean_abs_request_change_cents=deltas,
                  requests_cents={label: values.tolist() for label, values in requests.items()})
    output = control.ROOT / 'feature_trace_results.json'
    if output.exists():
        assert read(output) == report
    else:
        write(output, report)
    print('RESYNTHESIS_FEATURE_TRACE', deltas, flush=True)


if __name__ == '__main__':
    main()
