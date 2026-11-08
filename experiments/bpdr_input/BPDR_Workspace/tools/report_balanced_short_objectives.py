"""Read-only stratified diagnostics for the replayed balanced objective study.

No fitting, inference, waveform rendering, selection changes or fresh data.
Create once, then recompute in a fresh process with --mode verify.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import compare_balanced_short_objectives as study
from report_short_objective_comparison import diagnostic
from report_coverage_raw import summarize


def strata(entries, field):
    values = sorted({r[field] for r in entries if field in r}, key=str)
    return {str(value): summarize([r for r in entries if r.get(field) == value])
            for value in values}


def attempts(rows):
    excluded = [r for r in rows if 'excluded' in r]
    return dict(attempts=len(rows), scoreable=len(rows)-len(excluded),
                excluded=len(excluded),
                exclusion_reasons=dict(Counter(str(r['excluded']) for r in excluded)))


def build():
    root = study.ROOT
    receipt = root / 'integrity_receipt.json'
    result_path = root / 'results.json'
    proof = study.read(receipt)
    assert proof['checkpoint_reloads'] == 12
    assert proof['prediction_arrays_replayed'] == 4482
    assert proof['results_sha256'] == study.sha(result_path)
    declaration, rows, originals = study.declare()
    assert len(rows) == 1152 and len(originals) == 53
    saved = study.read(result_path)
    assert saved['declaration_sha256'] == study.sha(root / 'declaration.json')
    data = study.records(rows, originals)
    assert len(data) == 498
    duration = {r['source_id']: r['duration_seconds'] for r in originals if 'excluded' not in r}
    paths = [receipt, result_path, root / 'declaration.json', Path(__file__),
             Path(diagnostic.__code__.co_filename), Path(summarize.__code__.co_filename),
             study.previous.parent.DATA / 'train_pairs.json',
             study.previous.parent.DATA / 'train_originals.json']
    arms = {}
    for arm in declaration['conditions']:
        arms[arm] = {}
        for kind in study.KINDS:
            path = root / f'{arm}_{kind}_predictions.npz'
            paths.append(path)
            with np.load(path) as archive:
                predictions = {key: archive[key].copy() for key in archive.files}
            assert set(predictions) == {r['id'] for r in data}
            # Reconstruct the entire parent summary, including response and
            # boundary metrics, before deriving any additional breakdown.
            score = study.previous.parent.norm(study.previous.score(data, predictions))
            assert score == saved['arms'][arm][kind]
            entries = []
            for record in data:
                row = rows[record['pair_index']] if record['kind'] == 'corrupted' else None
                item = diagnostic(record, predictions[record['id']], row)
                item.update(source_id=record['source_id'],
                            duration_seconds=duration[record['source_id']])
                if row is None:
                    values = abs(predictions[record['id']][0][record['mask']])
                    item['mean_abs_cents'] = float(values.mean()) if len(values) else None
                entries.append(item)
            summary = summarize(entries)
            assert summary['scoreable_pairs'] == 445 and summary['original_windows'] == 53
            assert summary['positive_singer_macro_mae_cents'] == score['raw']['positive_singer_macro_mae_cents']
            # Independently derive the raw gate from per-record diagnostics.
            failed = []
            for group in score['raw']['groups']:
                own = [r for r in entries if r['group'] == group['group']]
                positive = [r for r in own if r['kind'] == 'corrupted']
                original = [r['mean_abs_cents'] for r in own
                            if r['kind'] == 'original' and r['mean_abs_cents'] is not None]
                pmean = float(np.mean([r['mae_cents'] for r in positive])) if positive else None
                response = float(np.mean([r['signed_response'] for r in positive])) if positive else None
                omean = float(np.mean(original)) if original else None
                assert pmean == group['positive_mae_cents']
                assert response == group['signed_response']
                assert omean == group['original_mae_cents']
                reasons = []
                if positive and not pmean < 25:
                    reasons.append('positive_group_mae_not_below_25_cents')
                if positive and not response >= .25:
                    reasons.append('positive_group_signed_response_below_0.25')
                if original and not omean <= 3:
                    reasons.append('original_group_mean_request_above_3_cents')
                if reasons:
                    failed.append(dict(group=group['group'], reasons=reasons))
            assert (len(failed) == 0) == score['raw']['raw_gate_passed']
            arms[arm][kind] = dict(all=summary,
                by_domain=strata(entries, 'domain'), by_singer=strata(entries, 'group'),
                by_duration_seconds=strata(entries, 'duration_seconds'),
                by_edit_width_seconds=strata(entries, 'width_seconds'),
                by_injected_direction={str(sign): summarize([r for r in entries
                    if r.get('injected_cents', 0) * sign > 0]) for sign in (-1, 1)},
                raw_gate_failed_groups=failed, records=entries)
    # Nominal predictions are shared across the three boundary contrasts;
    # do not count them as three independent cohorts in paired summaries.
    names = sorted(g for g, r in arms['bpdr']['split']['by_singer'].items() if r['scoreable_pairs'])
    paired = {}
    for control in ('augmentation_only', 'frame_only'):
        differences = np.array([
            arms['bpdr']['split']['by_singer'][g]['positive_singer_macro_mae_cents'] -
            arms[control]['split']['by_singer'][g]['positive_singer_macro_mae_cents'] for g in names])
        rng = np.random.default_rng(339)
        means = differences[rng.integers(0, len(names), size=(2000, len(names)))].mean(axis=1)
        paired[control] = dict(bpdr_minus_control_mae_cents=float(differences.mean()),
            descriptive_singer_bootstrap_percentile95_cents=np.quantile(means, [.025, .975]).tolist(),
            singer_groups=len(names), seed=339, replicates=2000)
    excluded = [r for r in rows if 'excluded' in r]
    return dict(arms=arms, paired_nominal_comparisons=paired,
        accounting=dict(physical_attempts=len(rows), construction_excluded=len(excluded),
            scoreable_physical_pairs=len(rows)-len(excluded), original_controls=len(originals),
            exclusion_reasons=dict(Counter(str(r['excluded']) for r in excluded)),
            by_singer={g: attempts([r for r in rows if r['group'] == g])
                       for g in sorted({r['group'] for r in rows})},
            by_domain={domain: attempts([r for r in rows
                if r['group'].startswith('HumTrans:') == (domain == 'humming')])
                for domain in ('humming', 'vocadito')}),
        artifact_hashes={str(p): study.sha(p) for p in paths},
        advance_to_delivery=saved['advance_to_delivery'],
        delivered_audio_evaluation='Not performed for this candidate: frozen raw prerequisites failed.',
        scope='Previously inspected training-source grouped cross-validation; bootstrap is descriptive, not fresh population inference. Frames, derivatives and boundary views are not independent participants. Tolerance fractions are raw requests, not musical accuracy.',
        raw_predictions_only=True, production_eligible=False, research_validation_passed=False)


def main(mode):
    result = build()
    path = study.ROOT / 'stratified_raw_diagnostics.json'
    if mode == 'create':
        assert not path.exists(), 'Report already exists; use --mode verify.'
        study.write(path, result)
    else:
        assert study.read(path) == result
        study.write(study.ROOT / 'stratified_raw_diagnostics_receipt.json', dict(
            report_sha256=study.sha(path), source_sha256=study.sha(Path(__file__)),
            parent_receipt_sha256=study.sha(study.ROOT / 'integrity_receipt.json'),
            prediction_arrays_rescored=4482, fresh_process_recomputation=True))
    print('BALANCED_STRATIFIED_DIAGNOSTICS', mode, result['accounting'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('create', 'verify'), required=True)
    main(parser.parse_args().mode)
