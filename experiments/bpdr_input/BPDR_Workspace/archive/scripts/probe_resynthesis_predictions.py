"""Compare held-out V68 requests on clean, zero-shift, and shifted waveforms."""

from pathlib import Path

import numpy as np
import torch

import compare_tail_preservation as study
import probe_resynthesis_control as control
from bpdr_input.experiments.learning.model_recovery import measured


def main():
    torch.set_num_threads(2)
    read, write, sha = control.read, control.write, control.sha
    protocol_path = control.ROOT / 'declaration.json'
    protocol = read(protocol_path)
    assert protocol['source_sha256'] == sha(Path(control.__file__))
    manifest = study.DATA / 'train_pairs.json'
    assert protocol['manifest_sha256'] == sha(manifest)
    rows = read(manifest)
    data_hashes = read(study.DATA / 'input_hashes.json')
    assert read(study.DATA / 'integrity_receipt.json')['input_hashes_sha256'] == sha(
        study.DATA / 'input_hashes.json'
    )
    fit_plan = study.ROOT / 'declaration.json'
    folds = read(fit_plan)['folds']
    archived_path = study.ROOT / 'upper_tail_preservation_all_predictions.npz'
    paths = [Path(__file__), protocol_path, fit_plan, archived_path]
    selected = []
    for index in protocol['case_indices']:
        row = rows[index]
        fold, = [i for i, groups in enumerate(folds, start=1) if row['group'] in groups]
        folder = study.ROOT / f'fold_{fold}/upper_tail_preservation'
        checkpoint = folder / 'checkpoint.pt'
        receipt = read(folder / 'fit_receipt.json')
        assert receipt['checkpoint_sha256'] == sha(checkpoint)
        assert row['source_id'] in receipt['heldout_sources']
        assert row['source_id'] not in receipt['training_sources']
        paths.extend([checkpoint, folder / 'fit_receipt.json'])
        for key in ('path_A', 'path_C', 'target_path'):
            assert sha(row[key]) == data_hashes[row[key]]
        selected.append((index, row, checkpoint))
    declaration = dict(source_hashes={str(path): sha(path) for path in paths},
                       selected=protocol['case_indices'], production_eligible=False)
    output = control.ROOT / 'prediction_declaration.json'
    if output.exists():
        assert read(output) == declaration
    else:
        write(output, declaration)
    results = []
    with np.load(archived_path) as archived:
        for index, row, checkpoint in selected:
            model, _ = study.model_module.load(checkpoint)
            clean, rate = control.sf.read(row['path_A'], dtype='float32')
            shifted, shifted_rate = control.sf.read(row['path_C'], dtype='float32')
            assert rate == shifted_rate == 22050
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
            with np.load(row['target_path']) as archive:
                positive = archive['mask'] & (archive['target'] != 0)
            clean_request, sham_request, shifted_request = [
                study.model_module.predict_raw(model, measured(audio)['views'][0])[0]
                for audio in (clean, sham, shifted)
            ]
            condition_metrics = {}
            for label, request in [('clean', clean_request), ('zero_shift', sham_request),
                                   ('shifted', shifted_request)]:
                local = request[positive]
                condition_metrics[label] = dict(
                    mean_abs_change_from_clean=float(np.mean(abs(local - clean_request[positive]))),
                    peak_abs_request=float(np.max(abs(local))),
                    frames_at_least_12_cents=int(np.count_nonzero(abs(local) >= 12)),
                )
            results.append(dict(
                index=index, group=row['group'], frames=int(positive.sum()),
                archived_clean_max_difference=float(np.max(abs(clean_request - archived['A_' + row['source_id']]))),
                archived_shifted_max_difference=float(np.max(abs(shifted_request - archived[f'C_{index}']))),
                **condition_metrics,
            ))
            print('RESYNTHESIS_PREDICTION_CASE', index, flush=True)
    result_path = control.ROOT / 'prediction_results.json'
    report = dict(declaration_sha256=sha(output), cases=results)
    if result_path.exists():
        assert read(result_path) == report
        print('RESYNTHESIS_PREDICTION_REPLAY_MATCH', len(results), flush=True)
    else:
        write(result_path, report)
        print('RESYNTHESIS_PREDICTION_COMPLETE', len(results), flush=True)


if __name__ == '__main__':
    main()
