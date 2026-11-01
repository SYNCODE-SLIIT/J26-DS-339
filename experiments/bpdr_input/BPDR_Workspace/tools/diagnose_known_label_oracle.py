"""Training-side feasibility diagnostic, never a deployable predictor or candidate."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import train_dense_residual_transformer as study
from bpdr_input.repair.held_voice_policy import HeldPolicy, proposal

ROOT = study.ROOT.parent / '2026-10-30_known_label_diagnostic'


def main():
    ex = study.ex; d = study.declare(); ROOT.mkdir(exist_ok=True)
    manifest = study.ROOT / 'train_pairs.json'
    declaration = ROOT / 'declaration.json'
    if not declaration.exists():
        ex.write(declaration, dict(created_at=datetime.now(timezone.utc).isoformat(),
                 source_sha256=ex.sha(Path(__file__)), manifest=str(manifest), manifest_sha256=ex.sha(manifest),
                 input_hashes=d['input_hashes'], policy=d['policy'],
                 rule='C predictions equal known supervised targets where mask is true and zero elsewhere, repeated across3 views; A all zero. Unknown ramp frames stay zero. Run unchanged input-only scope/proposal/controller, no rendering.',
                 interpretation='Unrealizable known-label diagnostic only. Not a runtime model, training input, benchmark contender, general optimum or correction-coverage ceiling: an actual model can predict unknown ramp/context values and supports can differ. Isolates losses even with exact known-frame corrections.',
                 selection_eligible=False, demo_activated=False, research_validation_passed=False))
    locked = ex.read(declaration)
    assert locked['source_sha256'] == ex.sha(Path(__file__))
    assert locked['manifest_sha256'] == ex.sha(manifest)
    for path, digest in locked['input_hashes'].items():
        assert ex.sha(path) == digest
    if (ROOT / 'results.json').exists():
        raise ValueError('Retain existing diagnostic; no overwrite')
    rows = ex.read(manifest); pairs = study.training.held.prepared(rows)
    counts, case_reasons, details = Counter(), Counter(), []
    for index, pair in enumerate(pairs):
        if 'excluded' in pair:
            continue
        counts['scoreable'] += 1
        value = np.where(pair['mask'], pair['target'], 0.).astype(np.float32)
        assert abs(value).max() <= 30.
        analysis = pair['C']
        detail = proposal(pair['audio_C'], analysis['features']['times_seconds'], analysis['views'],
                          np.stack([value] * 3), HeldPolicy(**locked['policy']))
        eligible = detail['plan']['permitted'] and not detail['plan']['identity']
        counts['nonidentity_permitted'] += int(eligible)
        counts['identity'] += int(detail['plan']['identity'])
        counts['controller_refused'] += int(not detail['plan']['permitted'])
        reasons = sorted({c['reason'] or 'ELIGIBLE' for c in detail['candidates']})
        case_reasons.update(reasons or ['NO_CANDIDATE'])
        details.append(dict(index=index, group=pair['group'], permitted_nonidentity=bool(eligible),
                            candidate_reasons=reasons, controller_reasons=detail['plan']['reasons']))
    result = dict(attempted=len(rows), excluded=sum('excluded' in r for r in rows),
                  counts=dict(counts), case_reasons_NONEXCLUSIVE=dict(case_reasons), cases=details,
                  declaration_sha256=ex.sha(declaration), interpretation=locked['interpretation'],
                  rendered_audio_evaluated=False, selection_eligible=False, research_validation_passed=False)
    ex.write(ROOT / 'results.json', result)
    print('KNOWN_LABEL_DIAGNOSTIC', {k: v for k, v in result.items() if k != 'cases'}, flush=True)


if __name__ == '__main__':
    main()
