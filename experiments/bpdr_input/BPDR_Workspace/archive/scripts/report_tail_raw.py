"""Prediction metrics with fixed source roles; not delivered/intended-note accuracy."""
import numpy as np
import compare_tail_preservation as run


def summarize(entries):
    positives=[r for r in entries if r['kind']=='corrupted'];originals=[r for r in entries if r['kind']=='original']
    groups=sorted({r['group'] for r in positives})
    macro=lambda field:float(np.mean([np.mean([r[field] for r in positives if r['group']==g]) for g in groups])) if groups else None
    return dict(scoreable_pairs=len(positives),positive_frames=sum(r['positive_frames'] for r in positives),
        positive_singer_macro_mae_cents=macro('mae_cents'),within10_singer_macro_fraction=macro('within10'),
        within25_singer_macro_fraction=macro('within25'),correct_direction_singer_macro_fraction=macro('correct_direction'),signed_response_singer_macro=macro('signed_response'),
        zero_command_singer_macro_mae_cents=macro('zero_command_mae_cents'),zero_command_within25_singer_macro_fraction=macro('zero_command_within25'),
        known_zero_singer_macro_mae_cents=macro('known_zero_mean_abs_cents'),known_zero_activation_frames=sum(r['known_zero_activation_frames'] for r in positives),
        wrong_direction_prediction_frames=sum(r['wrong_direction_frames'] for r in positives),
        original_windows=len(originals),original_windows_with_reliable_frames=sum(r.get('reliable_frames',0)>0 for r in originals),original_windows_with_activation_tail=sum(r['activation_frames']>0 for r in originals),
        maximum_original_request_cents=max((r['max_abs_cents'] for r in originals if r['max_abs_cents'] is not None),default=None))


def main():
    d,rows,originals=run.declare();proof=run.read(run.ROOT/'integrity_receipt.json')
    assert proof['results_sha256']==run.sha(run.ROOT/'results.json')
    data=run.paired.records(rows,originals);arms={};records={}
    for arm in run.ARMS:
        with np.load(run.ROOT/(arm+'_all_predictions.npz')) as z:pred={k:z[k].copy() for k in z.files}
        entries=[]
        for record in data:
            prediction=pred[record['id']];mask=record['mask'];target=record['target']
            entry=dict(id=record['id'],source_id=record['source_id'],group=record['group'],kind=record['kind'],
                domain='humming' if record['group'].startswith('HumTrans:') else 'vocadito')
            if record['kind']=='original':
                p=prediction[mask];entry.update(reliable_frames=int(mask.sum()),mean_abs_cents=float(np.mean(abs(p))) if len(p) else None,max_abs_cents=float(np.max(abs(p))) if len(p) else None,activation_frames=int(np.sum(abs(p)>=12.)))
            else:
                row=rows[record['pair_index']];positive=mask&(target!=0);zero=mask&(target==0);p=prediction[positive];t=target[positive];error=abs(p-t)
                entry.update(width_seconds=row['width_seconds'],injected_cents=row['cents'],positive_frames=int(positive.sum()),
                    mae_cents=float(np.mean(error)),within10=float(np.mean(error<=10)),within25=float(np.mean(error<=25)),
                    correct_direction=float(np.mean(p*t>0)),signed_response=float(np.mean(p*t)/np.mean(t*t)),
                    zero_command_mae_cents=float(np.mean(abs(t))),zero_command_within25=float(np.mean(abs(t)<=25)),wrong_direction_frames=int(np.sum(p*t<0)),zero_frames=int(zero.sum()),
                    known_zero_mean_abs_cents=float(np.mean(abs(prediction[zero]))),known_zero_activation_frames=int(np.sum(abs(prediction[zero])>=12)))
            entries.append(entry)
        assert sum(r['kind']=='corrupted' for r in entries)==445 and sum(r['kind']=='original' for r in entries)==53
        records[arm]=entries
        arms[arm]=dict(all=summarize(entries),
            by_domain={g:summarize([r for r in entries if r['domain']==g]) for g in ['humming','vocadito']},
            by_singer={g:summarize([r for r in entries if r['group']==g]) for g in sorted({r['group'] for r in entries})},
            by_width={str(w):summarize([r for r in entries if r.get('width_seconds')==w]) for w in [.24,.28,.36,.50]},
            by_injected_direction={str(sign):summarize([r for r in entries if r.get('injected_cents',0)*sign>0]) for sign in [-1,1]})
    # Paired resampling of singer summaries, not overlapping frame observations.
    names=sorted(g for g,s in arms['frozen_uniform']['by_singer'].items() if s['scoreable_pairs'])
    difference=np.array([arms['upper_tail_preservation']['by_singer'][g]['positive_singer_macro_mae_cents']-arms['frozen_uniform']['by_singer'][g]['positive_singer_macro_mae_cents'] for g in names])
    rng=np.random.default_rng(339);means=difference[rng.integers(0,len(difference),size=(2000,len(difference)))].mean(axis=1)
    run.write(run.ROOT/'raw_metric_report.json',dict(arms=arms,records=records,
        paired_singer_macro_mae_difference_tail_minus_uniform=float(difference.mean()),
        paired_singer_bootstrap_percentile95_cents=np.quantile(means,[.025,.975]).tolist(),bootstrap_replicates=2000,bootstrap_seed=339,
        bootstrap_scope='Descriptive resampling of these inspected training-source singer-group summaries, not fresh population inference; derived windows and frames are not independent people.',
        model_integrity_sha256=run.sha(run.ROOT/'integrity_receipt.json'),reporter_source_sha256=run.sha(__file__),
        raw_predictions_only=True,production_eligible=False,research_validation_passed=False))
    print('PRESERVATION_RAW_METRICS', {a:s['all'] for a,s in arms.items()},flush=True)


if __name__=='__main__':main()
