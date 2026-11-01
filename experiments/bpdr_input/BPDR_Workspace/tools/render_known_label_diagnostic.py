"""Actual-WAV known-label diagnostic; labels are privileged, never a model result."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import train_dense_residual_transformer as study
from bpdr_input.repair.curve_conditioning import conditioned_proposal
from bpdr_input.repair.held_voice_policy import HeldPolicy
from bpdr_input.repair.repair_controller import decision_hash
from bpdr_input.repair.rule_correction import fine_pitch

ROOT = study.ROOT.parent / '2026-10-30_known_label_wav_diagnostic'
PREFLIGHT = study.ROOT.parent / '2026-10-30_known_label_diagnostic'


def main():
    ex = study.ex; d = study.declare(); ROOT.mkdir(exist_ok=True)
    declaration = ROOT / 'declaration.json'
    if not declaration.exists():
        ex.write(declaration, dict(created_at=datetime.now(timezone.utc).isoformat(),
                 source_sha256=ex.sha(Path(__file__)), input_hashes=d['input_hashes'], policy=d['policy'],
                 preflight_sha256=ex.sha(PREFLIGHT / 'results.json'),
                 protocol='Use exact known C labels only where existing mask is known; unknowns zero. Same median11 conditioned proposal, actual Praat rendering, technical and encoded-WAV audits/rollback. No learned predictor, no original-preservation learning test. Score resulting recovery only to diagnose realization losses.',
                 NOT_MODEL_PERFORMANCE=True, selection_eligible=False, demo_activated=False,
                 research_validation_passed=False))
    locked = ex.read(declaration)
    assert locked['source_sha256'] == ex.sha(Path(__file__))
    assert locked['preflight_sha256'] == ex.sha(PREFLIGHT / 'results.json')
    for path, digest in locked['input_hashes'].items():
        assert ex.sha(path) == digest
    if (ROOT / 'results.json').exists():
        raise ValueError('Preserve completed diagnostic')
    rows = ex.read(study.ROOT / 'train_pairs.json'); pairs = study.training.held.prepared(rows)
    outcomes, cases = Counter(), []
    for index, pair in enumerate(pairs):
        if 'excluded' in pair:
            cases.append(dict(index=index, group=pair['group'], excluded=pair['excluded']))
            continue
        values = np.stack([np.where(pair['mask'], pair['target'], 0.).astype(np.float32)] * 3)
        curve, plan, _ = conditioned_proposal(pair, 'C', values, HeldPolicy(**locked['policy']), 'median11')
        assert decision_hash(plan) == plan['decision_sha256']
        folder = ROOT / 'audio' / f'case_{index:03d}'; folder.mkdir(parents=True, exist_ok=True)
        output, transaction = ex.rendered_output(pair, 'C', curve, plan, folder)
        if transaction['outcome'] != 'VERIFIED_FIXTURE_CANDIDATE':
            np.testing.assert_array_equal(output, pair['audio_C'])
        scored = pair['mask'] & (pair['target'] != 0)
        pitch = pair['C']['fine'] if np.array_equal(output, pair['audio_C']) else fine_pitch(output, 22050, pair['C']['features']['times_seconds'])
        before = float(abs(pair['C']['fine'][scored] - pair['A']['fine'][scored]).mean())
        after = float(abs(pitch[scored] - pair['A']['fine'][scored]).mean())
        improved = transaction['outcome'] == 'VERIFIED_FIXTURE_CANDIDATE' and after < before - 1e-6
        outcomes[transaction['outcome']] += 1
        outcomes['delivered_reference_improvement'] += int(improved)
        cases.append(dict(index=index, group=pair['group'], permission=plan, transaction=transaction,
                          before_mae_cents=before, after_mae_cents=after, improved=bool(improved)))
        ex.write(ROOT / 'progress.json', dict(completed_scoreable=sum(v for k, v in outcomes.items() if k != 'delivered_reference_improvement'), outcomes=dict(outcomes)))
    valid = [r for r in cases if 'excluded' not in r]; groups = sorted({r['group'] for r in valid})
    macro = lambda key: float(np.mean([np.mean([r[key] for r in valid if r['group'] == g]) for g in groups]))
    protected = ex.read(study.ROOT.parent / '2026-10-28_v7_guard/declaration.json')['protected_files']
    for path, digest in protected.items():
        assert ex.sha(path) == digest
    assert not (study.ROOT.parent / 'active_model.json').exists()
    result = dict(NOT_MODEL_PERFORMANCE=True, attempted=len(rows), scoreable=len(valid), excluded=len(rows) - len(valid),
                  outcomes=dict(outcomes), singer_macro_before_mae_cents=macro('before_mae_cents'),
                  singer_macro_after_mae_cents=macro('after_mae_cents'), cases=cases,
                  declaration_sha256=ex.sha(declaration), protected_files_unchanged=len(protected),
                  selection_eligible=False, demo_activated=False, research_validation_passed=False,
                  meaning='Privileged known-label controlled diagnostic, not deployable accuracy, model selection or a general upper bound; unknown supports and inferred supports can differ. No listener or natural-intention approval.')
    ex.write(ROOT / 'results.json', result)
    print('KNOWN_LABEL_WAV_DIAGNOSTIC', {k: v for k, v in result.items() if k != 'cases'}, flush=True)


if __name__ == '__main__':
    main()
