"""No-fit diagnosis of identical-view numerical residuals in a saved control."""

from pathlib import Path

import numpy as np
import torch

import compare_balanced_short_objectives as study


ROOT = study.ROOT.parent / '2026-11-08_frame_duplicate_diagnosis'


def main():
    torch.set_num_threads(2)
    declaration, rows, originals = study.declare()
    checkpoint = study.ROOT / 'fold_1/frame_only/checkpoint.pt'
    receipt_path = checkpoint.with_name('fit_receipt.json')
    receipt = study.read(receipt_path)
    assert receipt['checkpoint_sha256'] == study.sha(checkpoint)
    assert receipt['declaration_sha256'] == study.sha(study.ROOT / 'declaration.json')
    protocol = dict(source_sha256=study.sha(Path(__file__)),
        checkpoint_sha256=study.sha(checkpoint), fit_receipt_sha256=study.sha(receipt_path),
        parent_declaration_sha256=study.sha(study.ROOT / 'declaration.json'),
        hypothesis='Frame-only tokens are identical; batched float32 arithmetic may produce tiny within-batch residuals.',
        budget='First held fold, split view only, stop at first mismatch; two repeats and one single-view pass. No fit.',
        stopping='Record the mismatch and exact input/repeat checks. No acceptance-gate or source changes.',
        production_eligible=False)
    ROOT.mkdir(exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert study.read(path) == protocol
    else:
        study.write(path, protocol)
    records = study.records(rows, originals)
    held = [r for r in records if r['group'] in declaration['folds'][0]]
    assert [r['id'] for r in held] == receipt['held_ids']
    study.set_kind(held, study.KINDS[0])
    model = study.previous.short_experiment.new_model(declaration)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
    model.eval()
    for number, record in enumerate(held, 1):
        prediction = study.previous.predict(model, record, 'frame_only')
        if np.array_equal(prediction[0], prediction[1]):
            continue
        for _ in range(2):
            repeated = study.previous.predict(model, record, 'frame_only')
            np.testing.assert_array_equal(prediction, repeated)
        single = []
        for start in range(0, len(record['raw']), 128):
            centers = np.arange(start, min(start + 128, len(record['raw'])))
            views, padding = study.previous.select_tokens(record, centers, 'frame_only')
            np.testing.assert_array_equal(views[0], views[1])
            assert np.count_nonzero(views[..., 8:]) == 0
            with torch.inference_mode():
                single.append(model(torch.from_numpy(views[0]), torch.from_numpy(padding)).numpy())
        single = np.concatenate(single)
        single[record['raw'][:, 66] <= .5] = 0.
        difference = np.abs(prediction[0].astype(np.float64) - prediction[1].astype(np.float64))
        spacing = np.spacing(np.maximum(np.abs(prediction[0]), np.abs(prediction[1])))
        ulps = difference / np.maximum(spacing.astype(np.float64), np.finfo(np.float32).smallest_subnormal)
        output = dict(declaration_sha256=study.sha(path), held_records_visited=number,
            held_records_total=len(held), record_id=record['id'], kind=study.KINDS[0],
            duplicate_tokens_exact=True, repeats_exact=True,
            mismatch_frames=np.flatnonzero(difference).tolist(),
            maximum_difference_cents=float(difference.max()), maximum_float32_ulps=float(ulps.max()),
            single_view_difference_cents=float(np.max(abs(prediction[0] - single))),
            interpretation='Exact repeated batch outputs and identical input rows establish a numerical within-batch residual on this saved control; no model accuracy result.',
            production_eligible=False)
        study.write(ROOT / 'results.json', output)
        np.savez_compressed(ROOT / 'offending_predictions.npz', duplicate=prediction, single=single)
        print('FRAME_DUPLICATE_DIAGNOSIS', output, flush=True)
        return
    raise RuntimeError('No mismatch reproduced; do not change the assertion without further evidence.')


if __name__ == '__main__':
    main()
