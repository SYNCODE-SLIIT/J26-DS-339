"""Report domain, direction and preservation metrics from replayed objective fits."""

from pathlib import Path

import numpy as np

import compare_short_objectives as run
from report_coverage_raw import summarize


def diagnostic(record, prediction, row=None):
    mask, target = record['mask'], record['target']
    main, alternate = prediction
    assert main.shape == alternate.shape == target.shape and np.isfinite(prediction).all()
    entry = dict(id=record['id'], group=record['group'], kind=record['kind'],
                 domain='humming' if record['group'].startswith('HumTrans:') else 'vocadito')
    if record['kind'] == 'original':
        values = abs(main[mask])
        entry.update(reliable_frames=int(mask.sum()), max_abs_cents=float(values.max()) if len(values) else None,
                     activation_frames=int(np.count_nonzero(values >= 12)))
        return entry
    positive, known_zero = mask & (target != 0), mask & (target == 0)
    assert positive.any() and known_zero.any()
    correction, intended = main[positive], target[positive]
    error = abs(correction - intended)
    entry.update(
        width_seconds=row['width_seconds'], injected_cents=row['cents'],
        positive_frames=int(positive.sum()), mae_cents=float(error.mean()),
        within10=float((error <= 10).mean()), within25=float((error <= 25).mean()),
        correct_direction=float((correction * intended > 0).mean()),
        signed_response=float(np.mean(correction * intended) / np.mean(intended ** 2)),
        wrong_direction_frames=int(np.count_nonzero(correction * intended < 0)),
        zero_command_mae_cents=float(abs(intended).mean()),
        zero_command_within25=float((abs(intended) <= 25).mean()),
        known_zero_mean_abs_cents=float(abs(main[known_zero]).mean()),
        known_zero_activation_frames=int(np.count_nonzero(abs(main[known_zero]) >= 12)),
    )
    return entry


def report(records, rows, predictions):
    assert set(predictions) == {record['id'] for record in records}
    entries = [diagnostic(record, predictions[record['id']],
                          rows[record['pair_index']] if record['kind'] == 'corrupted' else None)
               for record in records]
    return dict(
        all=summarize(entries),
        by_domain={domain: summarize([row for row in entries if row['domain'] == domain])
                   for domain in ('humming', 'vocadito')},
        by_injected_direction={
            str(sign): summarize([row for row in entries if row.get('injected_cents', 0) * sign > 0])
            for sign in (-1, 1)
        },
        records=entries,
    )


def main():
    read, write, sha = run.read, run.write, run.sha
    root, data = run.ROOT, run.parent.DATA
    proof = root / 'integrity_receipt.json'
    results = root / 'results.json'
    assert read(proof)['checkpoint_reloads'] == 12 and read(proof)['results_sha256'] == sha(results)
    declaration = root / 'declaration.json'
    saved = read(results)
    assert saved['declaration_sha256'] == sha(declaration)
    manifest, originals = data / 'train_pairs.json', data / 'train_originals.json'
    data_proof = read(data / 'integrity_receipt.json')
    assert sha(manifest) == data_proof['train_pairs_sha256']
    assert sha(originals) == data_proof['train_originals_sha256']
    rows = read(manifest)
    records = run.parent.paired.records(rows, read(originals))
    assert len(records) == 498
    paths, arms = {}, {}
    for condition in run.CONDITIONS:
        path = root / f'{condition}_predictions.npz'
        paths[condition] = sha(path)
        with np.load(path) as predictions:
            arrays = {key: predictions[key] for key in predictions.files}
        assert run.parent.norm(run.score(records, arrays)) == saved['arms'][condition]
        arms[condition] = report(records, rows, arrays)
        assert arms[condition]['all']['scoreable_pairs'] == 445
    write(root / 'raw_diagnostics.json', dict(
        arms=arms, replay_receipt_sha256=sha(proof), reporter_sha256=sha(Path(__file__)),
        prediction_hashes=paths, raw_requests_only=True, production_eligible=False,
    ))


if __name__ == '__main__':
    main()
