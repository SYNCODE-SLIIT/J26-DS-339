"""PP1 end-to-end demonstration: validation path + the proposal's rule-based correction condition.

For each input recording:
  1. Phase 1 (pipeline.process): keep the original, technical checks, working copy, features, status.
  2. If accepted: rule_correction.run -> per-note decisions, permission, Praat render, re-measurement, rollback.
  3. Hand-off: by default the downstream package stays ACCEPT_UNCHANGED and a verified candidate is attached
     for comparison only. With --hand-off-prepared, a verified candidate becomes an ACCEPT_PREPARED package
     (policy rule-chromatic-0.1; uncalibrated thresholds, no listening test).
  4. report.html per case and one index.html for the demo set.
The transformer is not used here; its status is shown as a diagnosed experiment on the index page.

Commands:
  build-set   write the demo input set (real crops from the frozen catalogs + clearly labelled constructed cases)
  run         run every input of a demo set (or one file) and write the reports
"""
import argparse
from dataclasses import asdict
import html
import json
from pathlib import Path
import shutil
import uuid
import numpy as np
import soundfile as sf
from .contracts import AcceptedAudio, ProcessingResult, Status
from .pipeline import checksum, process, validate_package

COMPONENT = Path(__file__).resolve().parents[2]
SR = 22050


def _hum(note_cents, seconds=1.4, sharp_index=None, sharp_cents=0.):
    cents = []
    for k, c in enumerate(note_cents):
        cents += [c+(sharp_cents if k == sharp_index else 0.)]*int(seconds/.02)
    cents = np.asarray(cents, float)
    n = (len(cents)-1)*441
    track = np.interp(np.arange(n)/SR, np.arange(len(cents))*.02, 440*2**(cents/1200))
    ph = 2*np.pi*np.cumsum(track)/SR
    env = np.minimum(1, np.minimum(np.arange(n), n-np.arange(n))/(.05*SR))
    return ((.25*np.sin(ph)+.12*np.sin(2*ph)+.06*np.sin(3*ph))*env).astype(np.float32)


def build_set(evaluation_run):
    """Pick real crops by fixed rules from the rule evaluation, plus constructed technical/control cases."""
    out = COMPONENT/'runs/demo_inputs'/uuid.uuid4().hex; out.mkdir(parents=True)
    ev = json.loads((Path(evaluation_run)/'evidence.json').read_text())
    dev = json.loads((COMPONENT/'runs/support_development_packaging/779ea64608384daa9014865187c365b4/combined_catalog.json').read_text())['records']
    by_key = {(r['dataset'], r['phrase_id'], r['cents']): r for r in dev}
    cases = []
    edited = [d for d in ev['detuned'] if d['shifted_region_edited'] and d['edit_direction_correct']]
    edited.sort(key=lambda d: (-abs(d['injected_cents']), d['phrase_id']))
    if edited:
        d = edited[0]; r = by_key[(d['dataset'], d['phrase_id'], d['injected_cents'])]
        shutil.copyfile(r['waveform_paths']['C'], out/'01_jamendo_detuned.wav')
        shutil.copyfile(r['waveform_paths']['A'], out/'02_jamendo_original_same_phrase.wav')
        cases += [dict(file='01_jamendo_detuned.wav', kind='real crop, known injected shift', detail=f"{d['phrase_id']} with {d['injected_cents']:+g} cents injected on an audited held note"),
                  dict(file='02_jamendo_original_same_phrase.wav', kind='real crop, original', detail=f"{d['phrase_id']} unchanged (no injected shift)")]
    voc = sorted({(r['phrase_id'], r['waveform_paths']['A']) for r in dev if r['dataset'] == 'vocadito'})
    if voc:
        shutil.copyfile(voc[0][1], out/'03_vocadito_original.wav')
        cases.append(dict(file='03_vocadito_original.wav', kind='real crop, amateur singer, original', detail=voc[0][0]))
    sf.write(out/'04_constructed_hum_one_sharp_note.wav', _hum([-900, -700, -500, -400, -200], sharp_index=2, sharp_cents=20.), SR, subtype='PCM_24')
    cases.append(dict(file='04_constructed_hum_one_sharp_note.wav', kind='constructed control (not a real voice)', detail='five held notes; the third is 20 cents sharp'))
    sf.write(out/'05_constructed_hum_in_tune_transposed.wav', _hum([-863, -663, -463, -363, -163]), SR, subtype='PCM_24')
    cases.append(dict(file='05_constructed_hum_in_tune_transposed.wav', kind='constructed control (not a real voice)', detail='the same tune sung 37 cents high throughout: must not be edited'))
    sf.write(out/'06_silent.wav', np.zeros(SR*2, np.float32), SR, subtype='PCM_24')
    cases.append(dict(file='06_silent.wav', kind='constructed technical case', detail='two seconds of silence'))
    loud = np.clip(4*_hum([-900, -700, -500]), -1, 1)
    sf.write(out/'07_clipped.wav', loud, SR, subtype='PCM_24')
    cases.append(dict(file='07_clipped.wav', kind='constructed technical case', detail='overloaded recording (heavy clipping)'))
    sf.write(out/'08_too_long.wav', np.tile(_hum([-900, -700, -500]), 6), SR, subtype='PCM_24')
    cases.append(dict(file='08_too_long.wav', kind='constructed technical case', detail='about 25 seconds, above the 20-second limit'))
    (out/'09_unsupported.mp3').write_bytes(b'not audio')
    cases.append(dict(file='09_unsupported.mp3', kind='constructed technical case', detail='unsupported file type'))
    (out/'cases.json').write_text(json.dumps(dict(cases=cases, evaluation_run=str(evaluation_run)), indent=2), encoding='utf-8')
    print('DEMO_INPUTS:', out)
    return out


def _plot(path, report, times):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Patch
    fig = Figure(figsize=(12, 4.5)); FigureCanvasAgg(fig); ax = fig.subplots()
    cents = np.array([np.nan if c is None else c for c in report['fine_pitch_cents']])
    shown = cents.copy()
    shown[1:][np.abs(np.diff(cents)) > 150] = np.nan  # do not draw lines across note changes or tracker jumps
    ax.plot(times, shown, lw=.9, color='#555', label='measured pitch (YIN, reliable frames)')
    corr = np.asarray(report['corrections_cents'])
    if np.any(corr):
        after = np.where(corr != 0, shown+corr, np.nan)
        if report['outcome'] == 'VERIFIED_FIXTURE_CANDIDATE':
            ax.plot(times, after, lw=2, color='#1a7f37', label='after verified correction')
        else:
            ax.plot(times, after, lw=2, ls='--', color='#bc4c00', label=f"proposed, not delivered ({report['outcome']})")
    finite = cents[np.isfinite(cents)]
    if len(finite):
        lo, hi = np.percentile(finite, [2, 98]); ax.set_ylim(lo-150, hi+150)
        if report['tuning_offset_cents'] is not None:
            for g in np.arange(np.floor((lo-200)/100)*100, hi+300, 100)+report['tuning_offset_cents']:
                ax.axhline(g, color='#ddd', lw=.6, zorder=0)
    colours = dict(CORRECT='#1a7f37', WITHHELD='#bc4c00', NO_EDIT='#8c959f')
    for n in report['notes']:
        if n.get('reason') == 'NOTE_TOO_SHORT':
            continue
        ax.axvspan(n['start_seconds'], n['end_seconds'], color=colours[n['action']], alpha=.08 if n['action'] == 'NO_EDIT' else .18, lw=0)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(color=colours[k], alpha=.3) for k in ('CORRECT', 'WITHHELD', 'NO_EDIT')]
    labels += ['note proposed for correction', 'note withheld (unstable or over limit)', 'held note left unchanged']
    ax.set_xlabel('time (s)'); ax.set_ylabel("cents re A440 (grey: singer's own semitone grid)")
    ax.set_title(f"Rule-based correction: outcome {report['outcome']}"); ax.legend(handles, labels, loc='upper right', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120)


def run_one(source, out_root, hand_off_prepared=False):
    from .analysis import extract
    from .rule_correction import run as rule_run
    result, directory = process(source, out_root, COMPONENT/'checkpoints')
    case = dict(source=str(source), phase1_status=result.status.value, phase1_reasons=list(result.reason_codes),
                user_message=result.user_message, run_directory=str(directory))
    final = result
    if result.accepted_audio is not None:
        audio, sr = sf.read(result.accepted_audio.path, dtype='float32')
        with np.load(directory/'features.npz') as z:
            features = {k: z[k] for k in ('times_seconds', 'f0_hz', 'voiced_mask')}
        analyze = lambda a: extract(a, sr, COMPONENT/'checkpoints')[0]
        output, report = rule_run(audio, sr, features, analyze)
        case['rule_outcome'] = report['outcome']
        case['notes'] = report['notes']
        case['tuning_offset_cents'] = report['tuning_offset_cents']
        case['permission_reasons'] = report['plan']['reasons']
        case['audit'] = report['transaction'].get('post_render_audit')
        if report['outcome'] == 'VERIFIED_FIXTURE_CANDIDATE':
            cand = directory/'prepared_candidate.wav'
            sf.write(cand, output, sr, subtype='PCM_24')
            case['prepared_candidate'] = cand.name
            if hand_off_prepared:
                accepted = AcceptedAudio(str(cand), checksum(cand), sr, len(output))
                final = ProcessingResult(directory.name, result.source_id, Status.ACCEPT_PREPARED, accepted,
                                         user_message='Small pitch-centre corrections were applied to the marked notes; timing and all other audio are unchanged.',
                                         policy_version='rule-chromatic-0.1')
                validate_package(final)
        _plot(directory/'rule_correction.png', report, features['times_seconds'])
        (directory/'rule_report.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    case['hand_off'] = final.to_dict()
    (directory/'handoff.json').write_text(json.dumps(case['hand_off'], indent=2), encoding='utf-8')
    _case_html(directory, case)
    return case


def _case_html(directory, case):
    e = html.escape
    rows = ''.join(f"<tr><td>{n['start_seconds']:.2f}-{n['end_seconds']:.2f}</td><td>{e(n['action'])}</td><td>{e(n.get('reason', ''))}</td>"
                   f"<td>{'' if 'deviation_cents' not in n else f'{n["deviation_cents"]:+.1f}'}</td><td>{'' if 'correction_cents' not in n else f'{n["correction_cents"]:+.1f}'}</td></tr>"
                   for n in case.get('notes', []) if n['action'] != 'NO_EDIT' or n.get('reason') != 'NOTE_TOO_SHORT')
    players = ''
    original = next(Path(directory).glob('original.*'), None)
    if original:
        players += f'<h3>Original (kept unchanged)</h3><audio controls src="{original.name}"></audio>'
    if (Path(directory)/'accepted.wav').exists():
        players += '<h3>Accepted working copy</h3><audio controls src="accepted.wav"></audio>'
    if case.get('prepared_candidate'):
        players += f'<h3>Prepared candidate (verified)</h3><audio controls src="{case["prepared_candidate"]}"></audio>'
    hand = case['hand_off']
    (Path(directory)/'demo_report.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>BPDR demo case</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 16px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px}}code{{background:#f4f4f4}}</style>
<h1>{e(Path(case['source']).name)}</h1>
<p><b>Hand-off status:</b> <code>{e(hand['status'])}</code> {'(reasons: '+e(', '.join(hand['reason_codes']))+')' if hand['reason_codes'] else ''}<br>
<b>Message to the user:</b> {e(hand['user_message'])}<br>
<b>Rule-based correction:</b> {e(case.get('rule_outcome', 'not run (recording not accepted)'))}</p>
{players}
{'<img src="rule_correction.png" alt="Measured pitch with note decisions">' if case.get('notes') is not None else ''}
{'<h3>Note decisions</h3><table><tr><th>time (s)</th><th>action</th><th>reason</th><th>deviation (cents)</th><th>correction (cents)</th></tr>'+rows+'</table>' if rows else ''}
<p><a href="handoff.json">Hand-off package</a> · <a href="diagnostics.json">Phase 1 diagnostics</a>{' · <a href="rule_report.json">Rule report</a>' if case.get('notes') is not None else ''}</p>''', encoding='utf-8')


def run_set(inputs, hand_off_prepared=False):
    inputs = Path(inputs)
    cases = json.loads((inputs/'cases.json').read_text())['cases']
    out = COMPONENT/'runs/demo'/uuid.uuid4().hex; out.mkdir(parents=True)
    results = []
    for c in cases:
        r = run_one(inputs/c['file'], out/'cases', hand_off_prepared)
        results.append(dict(c, **{k: r[k] for k in ('phase1_status', 'user_message', 'run_directory')}, rule_outcome=r.get('rule_outcome'),
                            hand_off_status=r['hand_off']['status'], corrected_notes=sum(n['action'] == 'CORRECT' for n in r.get('notes', [])),
                            withheld_notes=sum(n['action'] == 'WITHHELD' for n in r.get('notes', []))))
        print(c['file'], '->', r['hand_off']['status'], r.get('rule_outcome'), flush=True)
    (out/'results.json').write_text(json.dumps(dict(inputs=str(inputs), hand_off_prepared=hand_off_prepared, cases=results), indent=2), encoding='utf-8')
    _index(out, results, hand_off_prepared)
    print('DEMO_RUN:', out)
    return out


def _index(out, results, hand_off_prepared):
    e = html.escape
    rows = ''.join(f"<tr><td><a href=\"{Path(r['run_directory']).relative_to(out).as_posix()}/demo_report.html\">{e(r['file'])}</a></td><td>{e(r['kind'])}</td>"
                   f"<td>{e(r['detail'])}</td><td><code>{e(r['hand_off_status'])}</code></td><td>{e(r['rule_outcome'] or '-')}</td>"
                   f"<td>{r['corrected_notes']}</td><td>{r['withheld_notes']}</td></tr>" for r in results)
    (out/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>BPDR PP1 demo</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 16px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px;vertical-align:top}}code{{background:#f4f4f4}}</style>
<h1>BPDR input component: PP1 demonstration</h1>
<p>Every recording keeps its original. Technical checks decide <code>ACCEPT_UNCHANGED</code>, <code>RERECORD_REQUIRED</code> or <code>TECHNICAL_FAILURE</code>.
Accepted recordings then go through the proposal's conservative rule-based correction: singer's own tuning, chromatic grid, one constant correction per stable held note,
permission limits, Praat rendering, re-measurement and rollback. Hand-off of prepared audio: <b>{'ON (ACCEPT_PREPARED)' if hand_off_prepared else 'OFF: downstream receives the unchanged audio; verified candidates are attached for comparison'}</b>.</p>
<table><tr><th>input</th><th>kind</th><th>detail</th><th>hand-off</th><th>rule outcome</th><th>notes proposed for correction</th><th>notes withheld</th></tr>{rows}</table>
<h2>Learned BPDR transformer: diagnosed experiment</h2>
<p>Not used in this demonstration. Two defects were found and fixed (33-cent pitch resolution of the frozen analyzer; a mean-absolute anchor that pins the output at zero).
A held-out-singer test after 2,400 updates still shows no correction on new voices (response gain about 0), so the learned path stays experimental. See the project plan for the next step.</p>''', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build-set'); b.add_argument('--evaluation-run', type=Path, required=True)
    r = sub.add_parser('run'); r.add_argument('inputs', type=Path); r.add_argument('--hand-off-prepared', action='store_true')
    a = p.parse_args()
    if a.command == 'build-set':
        build_set(a.evaluation_run)
    else:
        if a.inputs.is_file():
            out = COMPONENT/'runs/demo'/uuid.uuid4().hex
            print(json.dumps(run_one(a.inputs, out, a.hand_off_prepared)['hand_off'], indent=2))
        else:
            run_set(a.inputs, a.hand_off_prepared)


if __name__ == '__main__':
    main()
