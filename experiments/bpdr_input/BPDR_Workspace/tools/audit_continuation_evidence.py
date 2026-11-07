"""Read-only evidence audit for the November 2026 continuation decision.

Writes only a new audit JSON. Does not fit, change thresholds, activate a model,
read reserved audio, or overwrite any experiment result. Aggregate replay is not
waveform/inference replay. Audio checks cover the existing neural-noise outputs.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json

import numpy as np
import soundfile as sf

WS = Path(__file__).resolve().parents[1]
BASE = WS / 'results' / 'transformer_usability'
OUT = BASE / '2026-11-07_continuation_audit'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_tree(value):
    if not isinstance(value, dict):
        return value if not isinstance(value, list) else {'list_length': len(value)}
    return {k: scalar_tree(v) for k, v in value.items()
            if k not in {'groups', 'records', 'cases', 'predictions', 'decisions',
                         'by_group_boundary_and_response', 'source_hashes'}}


def result_receipt(folder):
    p = BASE / folder
    result, receipt = p / 'results.json', p / 'integrity_receipt.json'
    out = {'folder': folder, 'result_exists': result.exists(),
           'receipt_exists': receipt.exists()}
    if result.exists():
        out['result_sha256'] = sha(result)
        out['recorded_result'] = scalar_tree(read(result))
    if receipt.exists():
        x = read(receipt)
        out['receipt'] = x
        out['receipt_matches_result'] = result.exists() and x.get('results_sha256') == sha(result)
    declaration = p / 'declaration.json'
    if result.exists() and declaration.exists():
        out['result_matches_declaration'] = read(result).get('declaration_sha256') == sha(declaration)
    return out


def bootstrap(groups):
    sample = np.random.default_rng(339).integers(0, len(groups), size=(4000, len(groups)))
    return float(np.quantile(np.asarray(groups)[sample].mean(axis=1), .025))


def summarize_noise(folder, neural):
    p = BASE / folder
    data = read(p / 'results.json')
    records = data['records']
    out = {'folder': folder, 'results_sha256': sha(p / 'results.json'),
           'records': len(records), 'unique_recordings': len({r['track'] for r in records}),
           'singer_groups': len({r['group'] for r in records}),
           'independent_audio_replay': False, 'rows': {}, 'clean_controls': {}}
    declaration = read(p / 'declaration.json')
    tool = WS / 'tools' / ('real_noise_neural.py' if neural else 'real_noise_human_f0.py')
    out['declaration_matches_current_entry_tool'] = declaration['tool_sha256'] == sha(tool)
    out['result_matches_declaration'] = data['declaration_sha256'] == sha(p / 'declaration.json')
    errors = []
    candidate = 'N' if neural else 'R'
    flag = 'accN' if neural else 'accepted'
    delivered = 'GN' if neural else 'G'
    for ext in ('pyin', 'pesto', 'praat'):
        clean = [r for r in records if r['ext'] == ext and r['cond'] == 'clean']
        for snr in (10, 0):
            subset = [r for r in records if r['ext'] == ext and r['cond'] != 'clean' and r['snr'] == snr]
            groups = sorted({r['group'] for r in subset})
            group_mean = lambda key: np.asarray([np.mean([r[key] for r in subset if r['group'] == g]) for g in groups])
            clean_mean = np.mean([np.mean([r['C'] for r in clean if r['group'] == g]) for g in groups])
            input_mean = float(group_mean('C').mean())
            output_mean = float(group_mean(delivered).mean())
            gain = group_mean(delivered) - group_mean('C')
            accepted = [r for r in subset if r[flag]]
            harm = sum(r[delivered] < r['C'] - .02 for r in subset)
            row = dict(cases=len(subset), accepted=len(accepted), input_rpa50=input_mean,
                       delivered_rpa50=output_mean, gain=float(gain.mean()),
                       singer_bootstrap_lower=bootstrap(gain), harmful_cases=harm,
                       harm_all=harm/len(subset),
                       harm_accepted=harm/len(accepted) if accepted else None,
                       measured_clean_score=float(clean_mean),
                       loss_recovered=(output_mean-input_mean)/(clean_mean-input_mean)
                       if clean_mean-input_mean > 1e-9 else None)
            for r in subset:
                expected = r[candidate] if r[flag] else r['C']
                assert r[delivered] == expected
            summary = data['summary'][f'{ext}|{snr}dB']
            mapping = {'input_rpa50': 'C' if neural else 'macro_C',
                       'delivered_rpa50': 'GN' if neural else 'macro_G',
                       'gain': 'gain_GN' if neural else 'gain_G',
                       'singer_bootstrap_lower': 'gain_GN_ci_lo' if neural else 'gain_ci_lo',
                       'harm_all': 'harm_GN' if neural else 'harm_G',
                       'loss_recovered': 'loss_recovered_GN' if neural else 'loss_recovered_G'}
            for key, reported in mapping.items():
                error = abs(row[key]-summary[reported])
                errors.append(error)
                assert error < 1e-12, (folder, ext, snr, key, error)
            out['rows'][f'{ext}|{snr}dB'] = row
        clean_groups = sorted({r['group'] for r in clean})
        out['clean_controls'][ext] = dict(
            cases=len(clean), accepted=sum(r[flag] for r in clean),
            harmful_cases=sum(r[delivered] < r['C']-.02 for r in clean),
            mean_gain=float(np.mean([np.mean([r[delivered]-r['C'] for r in clean if r['group'] == g])
                                     for g in clean_groups])))
    out['maximum_summary_replay_error'] = max(errors)
    return out


def audio_inventory():
    p = BASE / '2026-11-05_real_noise_neural'
    rows = []
    for mixed in sorted((p / 'mix').glob('*.wav')):
        enhanced = p / 'enh' / mixed.name
        a, b = sf.info(mixed), sf.info(enhanced)
        y, sr = sf.read(enhanced, dtype='float64')
        rows.append(dict(file=mixed.name, input_frames=a.frames, output_frames=b.frames,
                         input_rate=a.samplerate, output_rate=b.samplerate,
                         input_channels=a.channels, output_channels=b.channels,
                         finite=bool(np.isfinite(y).all()),
                         sample_peak=float(np.max(np.abs(y))),
                         mixed_sha256=sha(mixed), enhanced_sha256=sha(enhanced)))
    return dict(files_checked=len(rows),
                rate_mismatches=sum(r['input_rate'] != r['output_rate'] for r in rows),
                frame_mismatches=sum(r['input_frames'] != r['output_frames'] for r in rows),
                channel_mismatches=sum(r['input_channels'] != r['output_channels'] for r in rows),
                nonfinite=sum(not r['finite'] for r in rows),
                peaks_above_one=sum(r['sample_peak'] > 1 for r in rows),
                rows=rows, timing_alignment_inferred=False,
                note='Matching shape does not establish absence of filter delay or preserved musical content.')


def objective_progress():
    root = BASE / '2026-11-06_short_objective_comparison'
    receipts = []
    for p in sorted(root.glob('fold_*/*/fit_receipt.json')):
        r = read(p)
        receipts.append(dict(path=str(p.relative_to(WS)),
                             checkpoint_hash_matches=sha(p.parent/'checkpoint.pt') == r['checkpoint_sha256'],
                             declaration_matches=sha(root/'declaration.json') == r['declaration_sha256'],
                             training_rows=len(r['training_ids']), held_rows=len(r['held_ids'])))
    x = read(root/'augmentation_only_interim_readout.json')
    return dict(completed_fit_receipts=receipts, final_result_exists=(root/'results.json').exists(),
                final_replay_receipt_exists=(root/'integrity_receipt.json').exists(),
                augmentation_interim=scalar_tree(x))


def inventory():
    rows = []
    for folder in ('documentation', 'reports', 'research_notes'):
        for p in sorted((WS/folder).rglob('*.md')):
            content = p.read_text(encoding='utf-8-sig')
            rows.append(dict(path=str(p.relative_to(WS)), sha256=sha(p),
                             lines=len(content.splitlines()),
                             headings=[l for l in content.splitlines() if l.startswith('#')]))
    # Inventory filenames only in datasets/results; do not decode any reserved material.
    counts = {}
    for name in ('documentation', 'source', 'tests', 'tools', 'exports', 'selected_model',
                 'notebooks', 'reports', 'research_notes', 'datasets', 'results', 'output'):
        files = [p for p in (WS/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        counts[name] = dict(files=len(files), by_extension=dict(Counter(p.suffix for p in files)))
    return dict(markdown=rows, folder_counts=counts,
                inspection_scope='Markdown indexed; owning code and selected results inspected separately. '
                                 'Not every binary or historical run was replayed.')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(date='2026-11-07', audit_tool_sha256=sha(Path(__file__)),
                  scope='Metadata, checkpoint/result hash binding, saved-record aggregate replay and output WAV integrity; '
                        'no fresh evaluation, full inference replay, listening or model promotion.',
                  studies=[result_receipt(n) for n in (
                      '2026-11-04_v56_preservation_diagnosis',
                      '2026-11-04_v66_preservation_delivery',
                      '2026-11-04_v71_coverage_delivery',
                      '2026-11-06_short_spectral_context_v2',
                      '2026-11-02_v41_short_phrase_models')],
                  noise=[summarize_noise('2026-11-05_real_noise_human_f0', False),
                         summarize_noise('2026-11-05_real_noise_neural', True)],
                  neural_audio=audio_inventory(), objective_progress=objective_progress(), inventory=inventory(),
                  current_export_card=read(WS/'exports/selected_run_inference/demo_model.json'))
    path = OUT/'evidence_audit.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    brief = dict(output=str(path), receipt_matches={r['folder']:r.get('receipt_matches_result') for r in report['studies']},
                 noise_replay_errors=[n['maximum_summary_replay_error'] for n in report['noise']],
                 neural_audio={k:v for k,v in report['neural_audio'].items() if k != 'rows'},
                 objective_completed_fits=len(report['objective_progress']['completed_fit_receipts']),
                 clean_controls=[n['clean_controls'] for n in report['noise']],
                 markdown_indexed=len(report['inventory']['markdown']))
    print(json.dumps(brief, indent=2))


if __name__ == '__main__':
    main()
