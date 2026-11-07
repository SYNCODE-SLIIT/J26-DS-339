"""Report reference coverage changes without presenting them as learned accuracy."""
from collections import Counter
import build_reliable_melodic_pairs as run


def main():
    root=run.ROOT;proof=run.read(root/'integrity_receipt.json')
    assert proof['train_pairs_sha256']==run.sha(root/'train_pairs.json')
    old=run.read(run.inputs.PARENTS[0]/'train_pairs.json');new=run.read(root/'train_pairs.json')
    originals=run.read(root/'train_originals.json');valid_windows={r['source_id'] for r in originals if 'excluded' not in r}
    assert len(old)==len(new)==1152
    key=lambda r:(r['source_id'],r['width_seconds'],r['cents'])
    before={key(r):r for r in old};after={key(r):r for r in new};assert set(before)==set(after)
    groups=[]
    for group in sorted({r['group'] for r in new}):
        a=[r for r in old if r['group']==group];b=[r for r in new if r['group']==group]
        duration_valid=sum(r['source_id'] in valid_windows for r in b);passing=sum('excluded' not in r for r in b)
        groups.append(dict(group=group,attempts=len(b),duration_valid_attempts=duration_valid,
            maximum_all_attempt_repair_fraction=passing/len(b),maximum_duration_valid_repair_fraction=passing/duration_valid if duration_valid else None,
            before_valid=sum('excluded' not in r for r in a),after_valid=sum('excluded' not in r for r in b),
            before_by_direction={s:sum('excluded' not in r and (r['cents']>0)==(s=='positive') for r in a) for s in ['positive','negative']},
            after_by_direction={s:sum('excluded' not in r and (r['cents']>0)==(s=='positive') for r in b) for s in ['positive','negative']}))
    transitions=Counter(('valid' if 'excluded' not in before[k] else before[k]['excluded'], 'valid' if 'excluded' not in after[k] else after[k]['excluded']) for k in before)
    report=dict(groups=groups,groups_below_25percent_duration_valid_reference_ceiling=[r['group'] for r in groups if r['maximum_duration_valid_repair_fraction'] is None or r['maximum_duration_valid_repair_fraction']<.25],transitions=[dict(before=a,after=b,attempts=n) for (a,b),n in sorted(transitions.items())],
        previous_valid=sum('excluded' not in r for r in old),current_valid=sum('excluded' not in r for r in new),
        new_constructible_keys=sum('excluded' in before[k] and 'excluded' not in after[k] for k in before),
        lost_constructible_keys=sum('excluded' not in before[k] and 'excluded' in after[k] for k in before),
        old_and_new_valid_but_support_changed=sum('excluded' not in before[k] and 'excluded' not in after[k] and before[k]['support_seconds']!=after[k]['support_seconds'] for k in before),
        construction_integrity_sha256=run.sha(root/'integrity_receipt.json'),reporter_source_sha256=run.sha(__file__),
        scope='Reference construction only. Different support locations are not identical physical examples. No trained-model or natural-intention accuracy claim.',new_model_fits=0)
    run.write(root/'coverage_report.json',report);print('RELIABLE_REGION_COVERAGE',report['previous_valid'],report['current_valid'],flush=True)


if __name__=='__main__':main()
