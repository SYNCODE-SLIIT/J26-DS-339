"""Expose a verified exploratory example without changing failed research gates."""
import json
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / 'source'))
from bpdr_input.audio.pipeline import checksum
from bpdr_input.modeling.recovery_model import load_locked_diagnostic
from bpdr_input.repair.learned_runtime import CARD


def main():
    folder = Path(json.loads((WORKSPACE / 'results/model_recovery/current.json').read_text())['path'])
    evidence = json.loads((folder / 'evidence.json').read_text())
    verification_path = folder / 'rendered_verification.json'
    verification = json.loads(verification_path.read_text())
    checkpoint = folder / 'selected_model.joblib'
    _, payload = load_locked_diagnostic(checkpoint)
    originals = verification['originals']
    shifts = verification['shifts']
    valid = [r for r in shifts if 'excluded' not in r]
    improved = [r for r in valid if r['verified_improvement']]
    if (len(originals) != 4 or not all(r['preserved'] for r in originals) or
            len(shifts) != 16 or not improved or
            verification['checkpoint_sha256'] != checksum(checkpoint) or
            evidence['rendered_verification_sha256'] != checksum(verification_path)):
        raise ValueError('No intact exploratory listening demonstration is available.')
    metrics = dict(model_type=payload['model_type'], validation='Development pilot; declared generalization criteria FAILED',
        heldout_singers=len(verification['by_singer']),
        singers_with_valid_shift_labels=len({r['singer'] for r in valid}),
        attempted_shifts=len(shifts), audited_shifts=len(valid),
        delivered_previews=sum(r['delivered_preview'] for r in valid), verified_improvements=len(improved),
        originals_preserved=sum(r['preserved'] for r in originals), originals_tested=len(originals),
        input_mae_cents=sum(r['input_mae_cents'] for r in valid)/len(valid),
        output_mae_cents=sum(r['output_mae_cents'] for r in valid)/len(valid),
        by_singer=verification['by_singer'],
        coverage_caveat='Only one of two evaluation singers supplied valid shift labels. Twelve of sixteen interventions are unscorable. This is not a population accuracy estimate.',
        failure_caveat='Two +30-cent examples improved. One +20-cent example was withheld; one -30-cent example changed a different region without improving the target.')
    metrics['error_reduction_fraction'] = 1-metrics['output_mae_cents']/metrics['input_mae_cents']
    shifted = improved[0]
    original = next(r for r in originals if r['input'] == shifted['reference'])
    cases = [dict(id='learned_pitch', label='Experimental learned repair: real voice + controlled 30-cent shift',
        input=shifted['input'], reference=shifted['reference'], score_mask_path=shifted['target_path'],
        source_kind='Selected verified example from an unseen-in-fit VocalSet singer. A known +30-cent shift was added. The full development pilot failed its broader criteria; reference and mask are used only after audio-only inference.'),
        dict(id='learned_original', label='Learned model preservation: unshifted real VocalSet voice',
        input=original['input'], reference=None, source_kind='Unshifted real VocalSet working copy from an unseen-in-fit singer. The experimental model should preserve this input.')]
    card = dict(version='exploratory-vocalset-preview-1', eligible_for_demo=True,
        validation_status='exploratory_preview_failed_generalization_gate', research_gates_passed=False,
        requires_user_review=True, checkpoint=str(checkpoint), checkpoint_sha256=checksum(checkpoint),
        verification_sha256=checksum(verification_path),
        scope='Experimental small local pitch-repair preview on stable sustained vocals. Two improvements in four scored interventions from one unseen-in-fit singer; other singer unscorable. Generalization gate failed; no natural intended-note accuracy claim.',
        operating_policy=payload['operating_policy'], metrics=metrics, demo_cases=cases,
        example_selection='Post-hoc first numerically verified improvement in locked source/shift order. Selected listening example, not representative accuracy.',
        evidence=str(folder / 'evidence.json'))
    CARD.write_text(json.dumps(card, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(card=str(CARD),metrics=metrics), indent=2))


if __name__ == '__main__':
    main()
