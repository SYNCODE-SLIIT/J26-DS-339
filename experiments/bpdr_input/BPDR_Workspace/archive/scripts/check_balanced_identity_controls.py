"""Held-checkpoint identity-resynthesis controls for the balanced comparison.

Uses the entire existing eight-case development control cohort. No fit,
reference-conditioned inference, new source selection or model promotion.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import compare_balanced_short_objectives as study
import probe_resynthesis_control as control
from bpdr_input.experiments.learning.model_recovery import measured


ROOT = study.ROOT / 'identity_controls'


def declare():
    declaration, rows, originals = study.declare()
    old_protocol = control.ROOT / 'declaration.json'
    protocol = control.read(old_protocol)
    manifest = study.previous.parent.DATA / 'train_pairs.json'
    assert protocol['manifest_sha256'] == study.sha(manifest)
    assert protocol['source_sha256'] == study.sha(Path(control.__file__))
    assert control.read(control.ROOT / 'results.json')['declaration_sha256'] == study.sha(old_protocol)
    cases = [(index, rows[index]) for index in protocol['case_indices']]
    assert len(cases) == 8 and all('excluded' not in row and row['split'] == 'train' for _, row in cases)
    input_hashes = study.read(study.previous.parent.DATA / 'input_hashes.json')
    inputs = {}
    for _, row in cases:
        for field in ['path_A', 'path_C', 'target_path']:
            path = Path(row[field])
            assert study.sha(path) == input_hashes[str(path)]
            inputs[str(path)] = study.sha(path)
    source_paths = [Path(__file__), Path(study.__file__), Path(study.previous.__file__),
        Path(control.__file__), Path(measured.__code__.co_filename),
        Path(study.boundaries.__file__), Path(study.previous.short.__file__)]
    record = dict(source_hashes={str(p): study.sha(p) for p in source_paths}, input_hashes=inputs,
        parent_declaration_sha256=study.sha(study.ROOT / 'declaration.json'),
        old_control_declaration_sha256=study.sha(old_protocol), selected_indices=protocol['case_indices'],
        hypothesis='Full BPDR must not confuse forced identity resynthesis with genuine local detuning; compare matched controls on the same held sources.',
        scope='Eight previously inspected development cases; supervised no-change/physical-detune controls, not human intended-note correctness or fresh confirmation.',
        budget='No fitting. Twenty-four waveform feature extractions and up to 216 two-view predictions per create/verify process, all three arms and perturbations.',
        metrics='Registered physical mask plus common reliable frames: identity sensitivity, clean/identity requests, actual-detune residual response, boundary sensitivity, voiced coverage; 12-cent flags are descriptive existing activation markers.',
        criteria='All sources must be absent from checkpoint fitting; exact inference/input replay. Retain unsupported-boundary exclusions. No promotion based on these controls.',
        stopping='Use all eight cases once and fresh-process replay after parent twelve-checkpoint replay. No weights, thresholds, seeds or cohort changes.',
        production_eligible=False)
    ROOT.mkdir(exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert study.read(path) == record
    else:
        study.write(path, record)
        snapshots = ROOT / 'source_snapshots'
        snapshots.mkdir(exist_ok=True)
        for source in source_paths:
            (snapshots / source.name).write_bytes(source.read_bytes())
    return declaration, rows, originals, cases, protocol


def make_record(raw):
    tokens, padding = study.previous.short.tokens_from_features(
        study.previous.short.frame_features(raw), np.arange(len(raw)))
    bank = study.boundaries.variants(raw)
    return dict(raw=raw, short_tokens=tokens, short_pad=padding), bank


def main(mode):
    torch.set_num_threads(2)
    declaration, rows, originals, cases, old = declare()
    if mode == 'declare':
        print('BALANCED_IDENTITY_DECLARED', len(cases), flush=True)
        return
    parent_result = study.ROOT / 'results.json'
    parent_receipt = study.ROOT / 'integrity_receipt.json'
    assert parent_receipt.exists(), 'Finish parent twelve-checkpoint replay before this evaluation.'
    assert study.read(parent_receipt)['results_sha256'] == study.sha(parent_result)
    assert study.read(parent_receipt)['checkpoint_reloads'] == 12
    artifact_hashes = {str(p): study.sha(p) for p in [parent_result, parent_receipt]}
    for path in study.ROOT.glob('fold_*/*/fit_receipt.json'):
        checkpoint = path.parent / 'checkpoint.pt'
        assert study.read(path)['checkpoint_sha256'] == study.sha(checkpoint)
        artifact_hashes.update({str(path): study.sha(path), str(checkpoint): study.sha(checkpoint)})
    assert len(list(study.ROOT.glob('fold_*/*/fit_receipt.json'))) == 12
    binding = dict(declaration_sha256=study.sha(ROOT / 'declaration.json'), parent_artifacts=artifact_hashes)
    binding_path = ROOT / 'execution_binding.json'
    if binding_path.exists():
        assert study.read(binding_path) == binding
    else:
        assert mode == 'create'
        study.write(binding_path, binding)
    if mode == 'create' and (ROOT / 'results.json').exists():
        raise FileExistsError('Identity results already exist; use --mode verify.')
    all_records = study.records(rows, originals)
    source_for_id = {r['id']: r['source_id'] for r in all_records}
    predictions, results = {}, []
    models = {}
    for index, row in cases:
        fold, = [i for i, groups in enumerate(declaration['folds'], 1) if row['group'] in groups]
        clean, rate = control.sf.read(row['path_A'], dtype='float32')
        altered, altered_rate = control.sf.read(row['path_C'], dtype='float32')
        assert rate == altered_rate == 22050 and clean.shape == altered.shape
        assert control.parselmouth.__version__ == old['renderer']['parselmouth_version']
        sound = control.parselmouth.Sound(clean.astype(np.float64), sampling_frequency=rate)
        manipulation = control.call(sound, 'To Manipulation', old['renderer']['time_step_seconds'],
            old['renderer']['pitch_floor_hz'], old['renderer']['pitch_ceiling_hz'])
        resynthesis = control.call(manipulation, 'Get resynthesis (overlap-add)')
        assert resynthesis.values.shape == (1, len(clean)) and resynthesis.x1 == sound.x1
        blend = control.envelope(np.arange(len(clean)) / rate, row['support_seconds'], old['renderer']['ramp_seconds'])
        identity = (clean * (1 - blend) + resynthesis.values[0] * blend).astype(np.float32)
        assert np.array_equal(identity[blend == 0], clean[blend == 0])
        analyses = {label: measured(wave) for label, wave in [('clean', clean), ('identity', identity), ('detuned', altered)]}
        with np.load(row['target_path']) as saved:
            mask = saved['mask'].astype(bool) & (saved['target'] != 0)
            target = saved['target'].copy()
        assert mask.any()
        prepared = {label: make_record(value['views'][0]) for label, value in analyses.items()}
        common = analyses['clean']['reliable'] & analyses['identity']['reliable']
        for arm in declaration['conditions']:
            key = (arm, fold)
            folder = study.ROOT / f'fold_{fold}' / arm
            receipt = study.read(folder / 'fit_receipt.json')
            assert 'A_' + row['source_id'] in receipt['held_ids']
            assert row['source_id'] not in {source_for_id[x] for x in receipt['training_ids']}
            if key not in models:
                model = study.previous.short_experiment.new_model(declaration)
                model.load_state_dict(torch.load(folder / 'checkpoint.pt', weights_only=True, map_location='cpu'))
                models[key] = model.eval()
            for kind in study.KINDS:
                unsupported = {label: bank[kind].get('excluded') for label, (_, bank) in prepared.items() if 'excluded' in bank[kind]}
                result = dict(case_index=index, group=row['group'], source_id=row['source_id'],
                    domain='humming' if row['group'].startswith('HumTrans:') else 'vocadito', arm=arm, kind=kind)
                if unsupported:
                    results.append(dict(result, excluded_boundary_views=unsupported))
                    continue
                values = {}
                for label, (record, bank) in prepared.items():
                    record['boundaries'] = np.stack((record['raw'][:, 70], bank[kind]['raw'][:, 70]))
                    if arm == 'frame_only':
                        token, _ = study.previous.select_tokens(record, np.arange(len(record['raw'])), arm)
                        np.testing.assert_array_equal(token[0], token[1])
                    values[label] = study.previous.predict(models[key], record, arm)
                    predictions[f'{index}_{arm}_{kind}_{label}'] = values[label]
                clean_request, identity_request, detuned_request = [values[label] for label in ['clean', 'identity', 'detuned']]
                results.append(dict(result, registered_frames=int(mask.sum()), common_reliable_frames=int(common.sum()),
                    identity_registered_voiced_frames=int(np.count_nonzero(mask & (prepared['identity'][0]['raw'][:, 66] > .5))),
                    clean_mean_abs_request=float(abs(clean_request[:, mask]).mean()),
                    identity_mean_abs_request=float(abs(identity_request[:, mask]).mean()),
                    identity_mean_delta_cents=float(abs(identity_request[:, mask] - clean_request[:, mask]).mean()),
                    identity_max_delta_cents=float(abs(identity_request[:, mask] - clean_request[:, mask]).max()),
                    identity_common_reliable_mean_delta=float(abs(identity_request[:, common] - clean_request[:, common]).mean()) if common.any() else None,
                    identity_frames_at_least_12_cents=int(np.count_nonzero(abs(identity_request[0, mask]) >= 12)),
                    clean_frames_at_least_12_cents=int(np.count_nonzero(abs(clean_request[0, mask]) >= 12)),
                    detuned_counter_response_error_cents=float(abs(detuned_request[:, mask] - clean_request[:, mask] - target[mask]).mean()),
                    identity_boundary_sensitivity_cents=float(abs(identity_request[0, mask] - identity_request[1, mask]).mean()),
                    identity_measured_pitch_delta_cents=float(np.median(abs(analyses['identity']['fine'][mask] - analyses['clean']['fine'][mask])))))
        print('BALANCED_IDENTITY_CASE', mode, index, flush=True)
    report = dict(execution_binding_sha256=study.sha(binding_path), cases=8, planned_arm_kind_rows=72,
                  records=results, prediction_arrays=len(predictions), production_eligible=False)
    arrays_path, path = ROOT / 'predictions.npz', ROOT / 'results.json'
    if mode == 'create':
        np.savez_compressed(arrays_path, **predictions)
        study.write(path, report)
    else:
        with np.load(arrays_path) as saved:
            assert set(saved.files) == set(predictions)
            for key, value in predictions.items():
                np.testing.assert_array_equal(value, saved[key])
        assert study.read(path) == report
        study.write(ROOT / 'integrity_receipt.json', dict(results_sha256=study.sha(path),
            execution_binding_sha256=study.sha(binding_path), prediction_arrays_replayed=len(predictions),
            cases_replayed=8, production_eligible=False))
    print('BALANCED_IDENTITY_COMPLETE', mode, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['declare', 'create', 'verify'], required=True)
    main(parser.parse_args().mode)
