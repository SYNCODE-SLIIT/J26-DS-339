"""Separate real-hum and Vocadito delivered outcomes; preserve every attempt."""
from collections import Counter
import numpy as np

import evaluate_humtrans_short_delivery as run


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
        outcomes=dict(Counter(r['transaction']['outcome'] for r in reports)))


def main():
    root = run.ROOT
    result = run.study.read(root / 'results.json')
    receipt = run.study.read(root / 'integrity_receipt.json')
    assert receipt['results_sha256'] == run.sha(root / 'results.json')
    _, rows, originals = run.study.declare()
    report = {}
    for arm, data in result['arms'].items():
        records = data['records']
        by_domain = {}
        for domain in ('humming', 'vocadito'):
            r = [item for item in records if domain_of(item) == domain]
            a = [item for item in rows if domain_of(item) == domain]
            o = [item for item in originals if domain_of(item) == domain]
            summary = summarize(r, a, o)
            assert summary['attempted'] == (192 if domain == 'humming' else 336)
            assert summary['scoreable'] == (61 if domain == 'humming' else 115)
            by_domain[domain] = summary
        report[arm] = dict(by_domain=by_domain,
            by_width={str(w): summarize([r for r in records if r['kind'] == 'corrupted' and r['width_seconds'] == w],
                                       [r for r in rows if r['width_seconds'] == w], [])
                      for w in sorted({r['width_seconds'] for r in rows})},
            by_direction={str(sign): summarize([r for r in records if r['kind'] == 'corrupted' and np.sign(r['cents']) == sign],
                                               [r for r in rows if np.sign(r['cents']) == sign], []) for sign in (-1, 1)})
    # These parents and the retained Vocadito cohort are exactly V47 physical.
    # A mismatch means investigate the evaluation, not silently claim a gain.
    previous = run.study.read(root.parent / '2026-11-02_v47_paired_delivery/results.json')['arms']['paired_physical']['summary']
    baseline = report['frozen_vocadito_physical']['by_domain']['vocadito']
    for key in ('verified_beneficial', 'verified_harms', 'target_improvements', 'known_zero_harms',
                'original_controls_unchanged', 'singer_macro_before_mae_cents', 'singer_macro_after_mae_cents'):
        assert baseline[key] == previous[key], key
    run.study.write(root / 'domain_report.json', dict(arms=report, retained_v47_vocadito_baseline_reproduced=True,
        results_sha256=run.sha(root / 'results.json'), integrity_sha256=run.sha(root / 'integrity_receipt.json'),
        report_source_sha256=run.sha(__file__), production_eligible=False, research_validation_passed=False))
    print('HUMTRANS_DELIVERY_DOMAIN_REPORT', {arm: {domain: (s['verified_beneficial'], s['verified_harms'], s['changed_original_ids'])
          for domain, s in data['by_domain'].items()} for arm, data in report.items()}, flush=True)


if __name__ == '__main__':
    main()
