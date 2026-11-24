"""Explain the current reference's stage losses without fitting or relaxing gates."""
from collections import Counter
import numpy as np
import train_dense_residual_transformer as study
from bpdr_input.repair.held_voice_policy import proposal, HeldPolicy, spans


def main():
    d = study.declare(); ex = study.ex
    rows = ex.read(study.ROOT / 'train_pairs.json')
    pairs = study.training.held.prepared(rows)
    outcome = ex.read(ex.read(study.ROOT / 'declaration.json')['baseline_evidence'])
    folder = study.BASE / 'arms/matched_baseline/training/median11'
    accuracy = ex.read(folder / 'accuracy.json')
    counts, reasons, case_reasons = Counter(), Counter(), Counter()
    details = []
    with np.load(folder / 'predictions.npz') as z:
        for i, pair in enumerate(pairs):
            if 'excluded' in pair:
                continue
            analysis = pair['C']; raw = analysis['views'][0]
            detail = proposal(pair['audio_C'], analysis['features']['times_seconds'], analysis['views'],
                              z[f'{i}_C'], HeldPolicy(**d['policy']))
            trace = accuracy['rows'][i]['trace']['C']
            assert detail['plan']['identity'] == trace['permission']['identity']
            np.testing.assert_array_equal(np.asarray(detail['plan']['intervals_seconds']),
                                          np.asarray(trace['permission']['intervals_seconds']))
            counts['scoreable'] += 1
            counts['allowed_long_unprotected_input'] += int(any(b - a >= 43 for a, b in spans(detail['allowed'] & detail['reliable'] & ~detail['protected'])))
            counts['has_any_prediction_support_candidate'] += bool(detail['candidates'])
            counts['has_permitted_nonidentity_plan'] += int(detail['plan']['permitted'] and not detail['plan']['identity'])
            counts['outcome_' + trace['outcome']] += 1
            one = Counter(c['reason'] or 'ELIGIBLE' for c in detail['candidates'])
            reasons.update(one)
            case_reasons.update(one.keys())
            if not one:
                case_reasons['NO_CANDIDATE'] += 1
            details.append(dict(index=i, group=pair['group'], source_sha256=pair['source']['sha256'],
                                candidate_reason_counts=dict(one), outcome=trace['outcome']))
    result = dict(reference_evidence_sha256=ex.sha(d['baseline_evidence']),
                  attempted=len(rows), scoreable=counts['scoreable'], excluded=len(rows) - counts['scoreable'],
                  stage_counts=dict(counts), candidate_counts_by_reason=dict(reasons),
                  case_counts_by_reason_NONEXCLUSIVE=dict(case_reasons), cases=details,
                  reference_delivered_after_mae_cents=outcome['summary']['delivered_after_mae_cents'],
                  meaning='Existing reference diagnostic only. Candidate reasons overlap within cases; not model-family causality. No fitting, targets as prediction inputs, new audio, support changes or activation.',
                  diagnostic_source_sha256=ex.sha(__file__))
    ex.write(study.ROOT / 'reference_bottlenecks.json', result)
    print('REFERENCE_BOTTLENECKS', {k: v for k, v in result.items() if k != 'cases'}, flush=True)


if __name__ == '__main__':
    main()
