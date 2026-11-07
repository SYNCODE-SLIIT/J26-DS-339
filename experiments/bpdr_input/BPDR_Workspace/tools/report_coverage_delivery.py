"""Separate real-hum and Vocadito delivered outcomes; preserve every attempt."""
from collections import Counter
import numpy as np
import soundfile as sf

import evaluate_coverage_delivery as run


def domain_of(row):
    return 'humming' if row['group'].startswith('HumTrans:') else 'vocadito'


def summarize(reports, attempts, originals):
    positive = [r for r in reports if r['kind'] == 'corrupted']
    controls = [r for r in reports if r['kind'] == 'original']
    scoreable_groups = sorted({r['group'] for r in positive})
    before = [float(np.mean([r['before_mae_cents'] for r in positive if r['group'] == group])) for group in scoreable_groups]
    after = [float(np.mean([r['after_mae_cents'] for r in positive if r['group'] == group])) for group in scoreable_groups]
    increases = [r['after_known_zero_mae_cents'] - r['before_known_zero_mae_cents'] for r in positive if r['known_zero_harm']]
    return dict(attempted=len(attempts), scoreable=len(positive), construction_excluded=len(attempts) - len(positive),
        attempted_groups=sorted({r['group'] for r in attempts}), scoreable_groups=scoreable_groups,
        groups_without_positive_pairs=sorted({r['group'] for r in attempts} - set(scoreable_groups)),
        original_attempts=len(originals), original_controls=len(controls),
        original_controls_unchanged=all(r['waveform_unchanged'] for r in controls),
        changed_original_ids=[r['source_id'] for r in controls if not r['waveform_unchanged']],
        verified_beneficial=sum(r['verified_beneficial'] for r in positive), verified_harms=sum(r['verified_harm'] for r in positive),
        target_improvements=sum(r['target_improvement'] for r in positive), known_zero_harms=sum(r['known_zero_harm'] for r in positive),
        beneficial_all_attempt_fraction=sum(r['verified_beneficial'] for r in positive) / len(attempts) if attempts else None,
        singer_macro_before_mae_cents=float(np.mean(before)) if before else None,
        singer_macro_after_mae_cents=float(np.mean(after)) if after else None,
        known_zero_error_increase_min_median_max_cents=np.quantile(increases, [0, .5, 1]).tolist() if increases else None,
        requested_wrong_target_direction_frames=sum(r.get('requested_wrong_target_direction_frames',0) for r in positive),
        measured_wrong_target_direction_frames=sum(r.get('measured_wrong_target_direction_frames',0) for r in positive),
        verified_records_with_measured_wrong_direction=sum(r.get('measured_wrong_target_direction_frames',0)>0 for r in positive),
        outcomes=dict(Counter(r['transaction']['outcome'] for r in reports)))



def target_direction_counts(arm,records,rows):
    evidence=[]
    for record in records:
        if record['kind']!='corrupted':continue
        requested=0;observed=0
        if record['verified_output']:
            row=rows[record['pair_index']]
            with np.load(row['target_path']) as z:plateau=z['plateau'].copy()
            with np.load(run.ROOT/arm/record['id']/'proposal_NOT_REFERENCE.npz') as z:curve=z['curve'].copy()
            with np.load(row['analysis_C']) as z:times=z['times_seconds'].copy()
            before,sr=sf.read(row['path_C'],dtype='float32');after,rate=sf.read(record['verified_output'],dtype='float32');assert sr==rate==22050
            target=-row['cents'];requested=int(np.sum(plateau&(curve*target<0)))
            delta=run.fine_pitch(after,sr,times)-run.fine_pitch(before,sr,times)
            # Same1e-6c numerical margin as the existing reference-harm report;
            # zero/withheld changes do not count as opposite-direction edits.
            observed=int(np.sum(plateau&(delta*np.sign(target)<-1e-6)))
        record['requested_wrong_target_direction_frames']=requested
        record['measured_wrong_target_direction_frames']=observed
        evidence.append(dict(id=record['id'],requested_wrong_target_direction_frames=requested,measured_wrong_target_direction_frames=observed))
    return evidence


def main():
    root = run.ROOT
    result = run.study.read(root / 'results.json')
    receipt = run.study.read(root / 'integrity_receipt.json')
    assert receipt['results_sha256'] == run.sha(root / 'results.json')
    _, rows, originals = run.study.declare()
    report = {};direction_evidence={}
    for arm, data in result['arms'].items():
        records = data['records']
        direction_evidence[arm]=target_direction_counts(arm,records,rows)
        by_domain = {}
        for domain in ('humming', 'vocadito'):
            r = [item for item in records if domain_of(item) == domain]
            a = [item for item in rows if domain_of(item) == domain]
            o = [item for item in originals if domain_of(item) == domain]
            summary = summarize(r, a, o)
            assert summary['attempted']==len(a) and summary['scoreable']==sum('excluded' not in item for item in a)
            by_domain[domain] = summary
        report[arm] = dict(by_domain=by_domain,
            by_width={str(w): summarize([r for r in records if r['kind'] == 'corrupted' and r['width_seconds'] == w],
                                       [r for r in rows if r['width_seconds'] == w], [])
                      for w in sorted({r['width_seconds'] for r in rows})},
            by_direction={str(sign): summarize([r for r in records if r['kind'] == 'corrupted' and np.sign(r['cents']) == sign],
                                               [r for r in rows if np.sign(r['cents']) == sign], []) for sign in (-1, 1)})
    run.study.write(root / 'domain_report.json', dict(arms=report, same_data_two_original_sampler_conditions=True,direction_evidence=direction_evidence,
        direction_scope="Post-output reference scoring on registered positive plateau frames; compare requests and delivered fine-pitch changes to the known counter-detuning target. This is separate from auditor agreement with a request and is not natural intention accuracy. No runtime threshold/policy is changed.",
        results_sha256=run.sha(root / 'results.json'), integrity_sha256=run.sha(root / 'integrity_receipt.json'),
        report_source_sha256=run.sha(__file__), production_eligible=False, research_validation_passed=False))
    print('HUMTRANS_DELIVERY_DOMAIN_REPORT', {arm: {domain: (s['verified_beneficial'], s['verified_harms'], s['changed_original_ids'])
          for domain, s in data['by_domain'].items()} for arm, data in report.items()}, flush=True)


if __name__ == '__main__':
    main()
