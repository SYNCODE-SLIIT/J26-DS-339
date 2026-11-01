"""Summarize measured joint-model results without changing adoption state."""
from pathlib import Path
import json

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE/'results/transformer_usability/2026-10-28_v8_joint'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


if __name__ == '__main__':
    selection = read(ROOT/'locked_selection.json')
    baseline = read(ROOT.parent/'2026-10-28_v6/locked_selection.json')['results']['step_conditioned']
    comparison = {'previous_step': baseline, **selection['results']}
    lines = ['# Joint Transformer: measured comparison', '',
             '**MOD-03 remains active. No model activation or research-completion claim.**', '',
             'These are training-side singer-held-out predictions on 108 scoreable / 200 attempted interventions, from eight scoreable singers and 27 scoreable recordings. Two further training singers have no scoreable repair interventions.', '',
             '| Model | Raw MAE (cents) | Within 10 cents (%) | Original request (cents) | Outside request (cents) | Prediction gate |',
             '|---|---:|---:|---:|---:|---|']
    for arm, result in comparison.items():
        s = result['summary']
        lines.append(f"| {arm} | {s['raw_mae_cents']:.3f} | {s['within_10c_percent']:.2f} | {s['original_abs_cents']:.3f} | {s['outside_abs_cents']:.3f} | {result['prediction_gate_passed']} |")
    lines += ['', f"The fixed selection rule chooses **{selection['arm']}** on balanced prediction error, before development replay. The two joint arms share the same 1,200-update budget. The historical step comparator used 1,000 updates; that comparison does not isolate architecture from training budget. No statistical or general-singing accuracy claim is made.", '',
              'The augmented arm adds training-source pitch-feature controls only. They are not real-WAV recordings, additional independent singers or final-test evidence.', '',
              '## Common-policy replay', '',
              'Raw model selection is unchanged. This replay uses activation 8 cents, step-fit guard .75 and the same numerical controller for all models; it does not render audio.', '',
              '| Model | Error after permission (cents) | Original request after permission (cents) |',
              '|---|---:|---:|']
    common = {}
    for arm in comparison:
        p = ROOT/f'common_policy/{arm}/accuracy.json'
        if p.exists():
            s = read(p)['summary'];common[arm] = s
            lines.append(f"| {arm} | {s['permitted_mae_cents']:.3f} | {s['permitted_original_abs_cents']:.3f} |")
    lines += ['', 'The original CV reports used default activation 12 cents for the joint models, while the preceding V7 runtime policy uses 8. The separately saved common-policy replay makes that distinction explicit. Neither threshold was chosen from new development outcomes.', '']
    development = None
    if (ROOT/'development/evidence.json').exists():
        development = read(ROOT/'development/evidence.json')
        gates = development['gates']
        lines += ['## Previously inspected development WAVs', '',
                  f"Verified improvements: **{gates['cohort_improved']}/{gates['cohort_attempted']} attempts**, with {gates['cohort_scoreable']} scoreable. All-original identity: **{gates['original_controls_all_identity']}**. Scoped development gate: **{gates['passed']}**.", '',
                  'This is a replay on known development recordings. It cannot establish independent generalization. Read [per-singer gates](development/gates.json), [accuracy](development/accuracy.json) and [original controls](development/original_controls.json).', '']
    else:
        lines += ['Actual-WAV development result is pending; no delivered benefit is inferred from the prediction table.', '']
    fits = list(ROOT.glob('cross_validation/**/fit_receipt.json'))+list(ROOT.glob('final/**/fit_receipt.json'))
    lines += [f"Saved neural fits in this experiment: {len(fits)}. Three new tests passed before fitting; earlier test receipts belong to their own runs.", '',
              'Two unused male VocalSet singers are reserved without acoustic access. New independent female confirmation is still needed; no both-voice-category final usability claim can be made from the reservation alone.', '',
              'The old four-singer failed final result remains unchanged, the diagnostic latest-candidate pointer remains on that completed candidate, and the active ExtraTrees export is preserved.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (ROOT/'comparison.json').write_text(json.dumps(dict(selected_arm=selection['arm'], comparison=comparison,
            common_policy=common, development=development, saved_fits=len(fits),
            demo_activated=False, research_validation_passed=False), indent=2)+'\n', encoding='utf-8')
    print(ROOT/'REPORT.md')
