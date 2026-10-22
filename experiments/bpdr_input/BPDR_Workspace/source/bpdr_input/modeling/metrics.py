"""Phase 3 measurements on predicted correction curves; repair and preservation are reported together."""
import numpy as np


def _mean(values, mask):
    return float(abs(values[mask]).mean()) if mask.any() else None


def evaluate(predictions, batch, edit_threshold_cents=5.):
    """One row per pair, then singers as the independent unit. Predictions: view -> batch x frames, in cents."""
    r = {view: np.asarray(values.detach().cpu() if hasattr(values, 'detach') else values) for view, values in predictions.items()}
    mask, detuning = np.asarray(batch['mask']), np.asarray(batch['detuning'])
    rows = []
    for index, pair_id in enumerate(batch['pair_ids']):
        common = mask[index]
        plateau, outside = common & (detuning[index] != 0), common & (detuning[index] == 0)
        if not plateau.any():
            rows.append(dict(pair_id=pair_id, singer_id=batch['singer_ids'][index], excluded='empty detuned mask'))
            continue
        target = -detuning[index]
        rows.append(dict(
            pair_id=pair_id, singer_id=batch['singer_ids'][index],
            # A model that outputs zero scores the injected size here, so flat invariance cannot pass as repair.
            repair_error_cents=np.mean([_mean(r[view][index]-target, plateau) for view in 'CD']),
            boundary_sensitivity_cents=np.mean([_mean(r[a][index]-r[b][index], common) for a, b in ('AB', 'CD')]),
            unchanged_alteration_cents=np.mean([_mean(r[view][index], common) for view in 'AB']),
            outside_alteration_cents=None if not outside.any() else np.mean([_mean(r[view][index], outside) for view in 'CD']),
            edit_coverage=float((abs(r['C'][index][common]) >= edit_threshold_cents).mean()),
            unnecessary_edit_coverage=float((abs(r['A'][index][common]) >= edit_threshold_cents).mean())))
    used = [row for row in rows if 'excluded' not in row]
    singers = sorted({row['singer_id'] for row in used})
    summary = dict(pairs=len(used), singers=len(singers), excluded_pairs=len(rows)-len(used), edit_threshold_cents=edit_threshold_cents)
    for key in ('repair_error_cents', 'boundary_sensitivity_cents', 'unchanged_alteration_cents', 'outside_alteration_cents',
                'edit_coverage', 'unnecessary_edit_coverage'):
        per_singer = [np.mean([row[key] for row in used if row['singer_id'] == singer and row[key] is not None] or [np.nan]) for singer in singers]
        summary[key] = None if not per_singer or np.all(np.isnan(per_singer)) else float(np.nanmean(per_singer))
    return dict(summary=summary, pairs=[{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in row.items()} for row in rows])
