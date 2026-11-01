"""Reconstruct authenticated training draws to measure nearby-zero supervision."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
import train_absolute_response_transformer as study
from bpdr_input.experiments.learning.context_tree_training import sampled_queries

ROOT = study.ROOT.parent / '2026-11-01_negative_sampling_diagnostic'


def main():
    ex = study.ex; d = study.declare(); ROOT.mkdir(exist_ok=True)
    declaration = ROOT / 'declaration.json'
    if not declaration.exists():
        ex.write(declaration, dict(created_at=datetime.now(timezone.utc).isoformat(),
                 source_sha256=ex.sha(Path(__file__)), sampler_source_sha256=ex.sha(Path(sampled_queries.__globals__['__file__'])),
                 parent_declaration_sha256=ex.sha(study.ROOT / 'declaration.json'),
                 protocol='Reconstruct original1200x12x3 draws per fold and require actual sampler/source-group digest match to V12. Classify real C positive/known-zero queries and distance to nearest known-positive frame. Bins10/25/43/100 frames are descriptive, matching short context, minimum support and model radius; not permission thresholds. Feature-only draws and original queries remain separate.',
                 input_hashes=d['input_hashes'], diagnostic_only=True, model_activated=False,
                 no_new_fitting=True, research_validation_passed=False))
    locked = ex.read(declaration)
    assert locked['source_sha256'] == ex.sha(Path(__file__))
    assert locked['sampler_source_sha256'] == ex.sha(Path(sampled_queries.__globals__['__file__']))
    assert locked['parent_declaration_sha256'] == ex.sha(study.ROOT / 'declaration.json')
    if (ROOT / 'results.json').exists():
        raise ValueError('Retain completed sampling diagnostic')
    rows = ex.read(study.ROOT / 'train_pairs.json'); pairs = study.training.held.prepared(rows)
    anchors = study.training.held.anchor_pool(ex.read(study.ROOT / 'anchor_sources.json'))
    reports = []
    for fold, groups in enumerate(d['folds'], 1):
        train = [p for p in pairs if ex.group(p) not in groups]
        originals = [p for p in anchors if p['group'] not in groups]
        _, queries, target, weights, digest, augmentations = sampled_queries(train, originals, 339, 1200)
        old = ex.read(study.BASE / f'cross_validation/fold_{fold}/matched_baseline/fit_receipt.json')
        assert digest == old['sampling_sha256']
        by_path = {p['path_C']: p for p in train if 'excluded' not in p}
        distances = {path: distance_transform_edt(~(p['mask'] & (p['target'] != 0))) for path, p in by_path.items()}
        counts = Counter(); nearby_weight = positive_weight = 0.
        for position, ((key, index), value, weight) in enumerate(zip(queries, target, weights, strict=True)):
            slot = position % 3
            if slot == 2:
                assert value == 0. and weight == 4.
                counts['uniform_original_anchor'] += 1
            elif not isinstance(key, str):
                counts['feature_only_' + ('original' if slot == 0 else 'positive' if value else 'outside')] += 1
            elif slot == 0:
                assert value == 0. and weight == 4.
                counts['paired_real_original'] += 1
            else:
                p = by_path[key]; assert p['mask'][index]
                assert np.isclose(value, p['target'][index] / 30.)
                if value:
                    counts['real_C_positive'] += 1; positive_weight += float(weight)
                else:
                    counts['real_C_known_zero'] += 1
                    distance = float(distances[key][index])
                    for bound in (10, 25, 43, 100):
                        if distance <= bound:
                            counts[f'real_C_known_zero_within_{bound}_frames'] += 1
                    if distance <= 43:
                        nearby_weight += float(weight)
        nearby = counts['real_C_known_zero_within_43_frames']; total = counts['real_C_known_zero']
        report = dict(fold=fold, counts=dict(counts), sampled_rows=len(queries), sampling_sha256=digest,
                      feature_only_examples=augmentations, original_anchor_recordings=len(originals),
                      nearby_zero_fraction_of_real_C_zero=nearby / total,
                      nearby_zero_fraction_of_all_rows=nearby / len(queries),
                      positive_to_nearby_zero_draw_ratio=counts['real_C_positive'] / nearby,
                      positive_to_nearby_zero_weight_ratio=positive_weight / nearby_weight)
        reports.append(report)
        print('NEGATIVE_SAMPLING_FOLD', report, flush=True)
    protected = ex.read(study.ROOT.parent / '2026-10-28_v7_guard/declaration.json')['protected_files']
    for path, digest in protected.items():
        assert ex.sha(path) == digest
    result = dict(folds=reports, reconstructed_sampled_rows=sum(r['sampled_rows'] for r in reports),
                  source_singer_folds_retained=True, all_reference_sampling_hashes_exact=True,
                  declaration_sha256=ex.sha(declaration), protected_files_unchanged=len(protected),
                  interpretation='Descriptive training exposure, not proof that hard negatives improve the model. Labels locate queries only for this training audit, never runtime input. Four folds reuse sources; rows are not independent people. Near-zero counts at multiple bounds overlap.',
                  model_activated=False, no_new_fitting=True, research_validation_passed=False)
    ex.write(ROOT / 'results.json', result)
    print('NEGATIVE_SAMPLING_DIAGNOSTIC_COMPLETE', result['reconstructed_sampled_rows'], flush=True)


if __name__ == '__main__':
    main()
