"""PP1 explainer page for the learned BPDR transformer on the controlled proof-of-concept data.

Shows, from saved runs only:
  1. the research gap and the proposed method (four views, L_A, L_B, L_D, L_S, three matched models),
  2. held-out-voice results of the three models (proposal Table 3 metrics) and their learning curves,
  3. one held-out phrase chosen by a fixed rule (first held-out record with a 20-cent shift and a false split (30 cents is the permission limit)):
     the four predicted correction curves, then the BPDR prediction on the detuned audio passed through the
     component's own permission, Praat rendering, re-measurement and rollback path,
  4. the honest real-data status and the link to the rule-based demonstration.
All controlled results are on constructed voices and are not evidence of real-voice accuracy.
"""
from bpdr_input import COMPONENT_ROOT
import html
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

COMPONENT = COMPONENT_ROOT
SR = 22050
LABELS = dict(augmentation_only='Augmentation-only (71)', frame_only='Frame-only (70)', bpdr='BPDR (71)')


def load_model(path):
    import torch
    from bpdr_input.modeling.examples import Normalizer
    from bpdr_input.experiments.learning.matched_comparison import matched_model
    saved = torch.load(path, weights_only=False, map_location='cpu')
    model = matched_model(saved['protocol'], saved['width'])
    model.load_state_dict(saved['state']); model.eval()
    return model, Normalizer(np.asarray(saved['normalizer']['mean']), np.asarray(saved['normalizer']['scale'])), saved['width']


def predict(model, normalizer, width, pair):
    import torch
    from bpdr_input.modeling.examples import collate
    with torch.inference_mode():
        b = collate([pair], normalizer, width)
        return {v: model(x, b['padding'])[0].numpy().astype(np.float64) for v, x in b['inputs'].items()}


def correction_from_prediction(pred_c, pred_d, mask, threshold=3.):
    """Largest contiguous span where the BPDR curve asks for at least `threshold` cents; values are not clipped."""
    want = mask & (np.abs(pred_c) >= threshold)
    idx = np.flatnonzero(want)
    curve = np.zeros(len(pred_c))
    if not len(idx):
        return curve
    spans = np.split(idx, np.flatnonzero(np.diff(idx) > 1)+1)
    span = max(spans, key=len)
    curve[span[0]:span[-1]+1] = pred_c[span[0]:span[-1]+1]
    return curve


def render_prediction(record, pred, wav='C.wav', main='C', alternate='D'):
    """Run the BPDR correction predicted for one audio (main view, alternate = corrupted-boundary view) through the
    existing permission/render/audit/rollback path. Used on the detuned C audio and on the unshifted original A."""
    from bpdr_input.audio.analysis import extract
    from bpdr_input.repair.framewise_audit import audit_curve
    from bpdr_input.repair.framewise_renderer import render_candidate
    from bpdr_input.repair.repair_controller import plan_fixture, transact_fixture
    from bpdr_input.repair.rule_correction import fine_hz
    folder = Path(record['path']).parent
    audio, sr = sf.read(folder/wav, dtype='float32')
    before, _ = extract(audio, sr, COMPONENT/'checkpoints')
    times = before['times_seconds']
    # Runtime information only: voicing measured on this audio itself, never the private training mask.
    voiced = before['voiced_mask'].astype(bool)
    corrections = correction_from_prediction(pred[main], pred[alternate], voiced)
    uncertainty = np.abs(pred[main]-pred[alternate])
    mask = voiced
    plan = plan_fixture(audio, sr, times, corrections, pred[alternate], uncertainty, voiced, np.zeros(len(times), bool))

    def renderer(a, rate, intervals):
        return render_candidate(a, rate, times, corrections, intervals)[0]

    def auditor(a, b, rate, intervals):
        after, _ = extract(b, rate, COMPONENT/'checkpoints')
        return audit_curve(before, after, fine_hz(a, rate, times), fine_hz(b, rate, times), mask, intervals, corrections)

    output, transaction = transact_fixture(audio, sr, plan, renderer, auditor)
    return output, corrections, plan, transaction


def figures(out, evidence, example, preds, times, A, C, record):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    # Learning curves on held-out voices.
    fig = Figure(figsize=(12, 3.6)); FigureCanvasAgg(fig); axes = fig.subplots(1, 3)
    for c, r in evidence['results'].items():
        u = [p['update'] for p in r['curve']]
        axes[0].plot(u, [p['heldout']['correction_mae'] for p in r['curve']], marker='o', label=LABELS[c])
        axes[1].plot(u, [p['heldout']['median_response_gain'] for p in r['curve']], marker='o', label=LABELS[c])
        axes[2].plot(u, [p['heldout']['boundary_sensitivity'] for p in r['curve']], marker='o', label=LABELS[c])
    zero = next(iter(evidence['results'].values()))['heldout']['zero_mae']
    axes[0].axhline(zero, color='#999', ls=':', label='no correction')
    axes[1].axhline(1, color='#999', ls=':', label='ideal')
    for ax, t in zip(axes, ('correction error, cents (lower is better)', 'paired response gain (ideal 1)', 'boundary sensitivity, cents (lower is better)')):
        ax.set_title(t, fontsize=10); ax.set_xlabel('update')
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'learning_curves.png', dpi=120)
    # Example phrase: input pitch and the four predicted curves.
    fig = Figure(figsize=(12, 7)); FigureCanvasAgg(fig); ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)
    va, vc = A[:, 66] > .5, C[:, 66] > .5
    ax1.plot(times, np.where(va, A[:, 64], np.nan), label='original (view A)', color='#555')
    ax1.plot(times, np.where(vc, C[:, 64], np.nan), label='detuned (view C)', color='#cf222e', lw=.9)
    s0, s1 = record['support_seconds']; ax1.axvspan(s0, s1, color='#cf222e', alpha=.08, label='detuned support')
    for e in record['nominal_edges']:
        ax1.axvline(e, color='#0969da', lw=.6)
    for e in set(map(float, record['corrupted_edges']))-set(map(float, record['nominal_edges'])):
        ax1.axvline(e, color='#bc4c00', lw=1.4, ls='--')
    ax1.set_ylabel('pitch, cents re A440'); ax1.set_title('Input: blue lines = nominal boundaries, orange dashed = corrupted boundary (views B, D)', fontsize=10)
    ax1.legend(fontsize=8, loc='upper right')
    with np.load(record['path']) as z:
        target = z['correction_target_cents']; mask = z['mask']
    for ax, cond in ((ax2, 'bpdr'), (ax3, 'augmentation_only')):
        p = preds[cond]
        ax.plot(times, np.where(mask & (target != 0), target, np.nan), color='k', lw=3, alpha=.35, label='known compensating target')
        for v, col, ls in (('A', '#555', '-'), ('B', '#999', '--'), ('C', '#cf222e', '-'), ('D', '#bc4c00', '--')):
            ax.plot(times, np.where(mask, p[v], np.nan), color=col, ls=ls, lw=1, label=f'r{v}')
        ax.set_ylabel('predicted correction, cents'); ax.set_title(f'{LABELS[cond]}: predicted correction for the four views', fontsize=10)
        ax.legend(fontsize=8, ncol=5, loc='lower right')
    ax3.set_xlabel('time (s)'); fig.tight_layout(); fig.savefig(out/'example_views.png', dpi=120)


def seed_runs():
    """The design-seed run plus any replicate seeds, as [(seed, evidence)]."""
    from bpdr_input.experiments.learning.controlled_study import ROOT
    runs = [(339, Path(json.loads((ROOT/'training_current.json').read_text())['path']))]
    extra = ROOT/'training_replicates.jsonl'
    if extra.exists():
        runs += [(json.loads(l)['seed'], Path(json.loads(l)['path'])) for l in extra.read_text().splitlines() if l.strip()]
    return [(seed, json.loads((path/'evidence.json').read_text())) for seed, path in runs]


def seed_table(seeds):
    keys = ('correction_mae', 'median_response_gain', 'boundary_sensitivity', 'unshifted_alteration')
    return {c: {k: [ev['results'][c]['heldout'][k] for _, ev in seeds] for k in keys} for c in LABELS}


def build(page_dir):
    from bpdr_input.experiments.learning.controlled_study import ROOT, load_pair
    from bpdr_input.experiments.learning.fine_pitch_relative_check import relative_pair
    data = Path(json.loads((ROOT/'current.json').read_text())['path'])
    run = Path(json.loads((ROOT/'training_current.json').read_text())['path'])
    cat = json.loads((data/'catalog.json').read_text())['records']
    evidence = json.loads((run/'evidence.json').read_text())
    record = next(r for r in cat if r['split'] == 'heldout' and abs(r['cents']) == 20 and r['corruption'] == 'split')
    pair = relative_pair(load_pair(record))
    preds = {}
    for c in ('augmentation_only', 'frame_only', 'bpdr'):
        model, normalizer, width = load_model(run/f'{c}.pt')
        preds[c] = predict(model, normalizer, width, pair)
    out = Path(page_dir); out.mkdir(parents=True, exist_ok=True)
    with np.load(record['path']) as z:
        times, A, C = z['times'], z['A'], z['C']
    figures(out, evidence, record, preds, times, A, C, record)
    output, corrections, plan, transaction = render_prediction(record, preds['bpdr'])
    _, original_corrections, original_plan, original_transaction = render_prediction(record, preds['bpdr'], 'A.wav', 'A', 'B')
    folder = Path(record['path']).parent
    import shutil
    shutil.copyfile(folder/'A.wav', out/'example_original.wav'); shutil.copyfile(folder/'C.wav', out/'example_detuned.wav')
    if transaction['outcome'] == 'VERIFIED_FIXTURE_CANDIDATE':
        sf.write(out/'example_bpdr_corrected.wav', output, SR, subtype='PCM_24')
    from bpdr_input.repair.rule_correction import fine_pitch
    with np.load(record['path']) as z:
        P = z['mask'].astype(bool) & (z['correction_target_cents'] != 0)
    a_audio, _ = sf.read(folder/'A.wav', dtype='float32')
    fa, fc, fo = (fine_pitch(x, SR, times) for x in (a_audio, sf.read(folder/'C.wav', dtype='float32')[0], output))
    realized = dict(detuned_vs_original_cents=float(np.median(fc[P]-fa[P])), corrected_vs_original_cents=float(np.median(fo[P]-fa[P])))
    result = dict(record=dict(phrase_id=record['phrase_id'], voice_id=record['voice_id'], cents=record['cents'], corruption=record['corruption']),
                  outcome=transaction['outcome'], permission_reasons=plan['reasons'], audit=transaction.get('post_render_audit'),
                  realized=realized, rollback_reasons=transaction['reasons'] if transaction.get('rollback') else [],
                  original_chain=dict(outcome=original_transaction['outcome'], permission_reasons=original_plan['reasons'],
                                      proposed_max_abs_cents=float(np.abs(original_corrections).max()),
                                      max_abs_prediction_on_original_cents=float(np.abs(preds['bpdr']['A'][pair['inputs']['A'][:, 66] > .5]).max())),
                  seeds=[seed for seed, _ in seed_runs()], seed_table=seed_table(seed_runs()))
    (out/'example_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    page(out, evidence, result, run, data)
    print('SHOWCASE:', out/'bpdr_explainer.html'); print(json.dumps(result, indent=1)[:1500])
    return out


def page(out, evidence, result, run, data):
    e = html.escape
    rows = ''.join(f"<tr><td>{LABELS[c]}</td><td>{r['heldout']['correction_mae']:.2f}</td><td>{r['heldout']['zero_mae']:.2f}</td>"
                   f"<td>{100*r['heldout']['correction_gain']:.0f}%</td><td>{r['heldout']['median_response_gain']:.2f}</td>"
                   f"<td>{r['heldout']['boundary_sensitivity']:.3f}</td><td>{r['heldout']['unshifted_alteration']:.2f}</td><td>{r['heldout']['outside_alteration']:.2f}</td></tr>"
                   for c, r in evidence['results'].items())
    corr = ''.join(f"<tr><td>{LABELS[c]}</td>"+''.join(f"<td>{r['by_corruption'][k]['boundary_sensitivity']:.3f}</td>" for k in ('split', 'merge', 'jitter'))+'</tr>'
                   for c, r in evidence['results'].items())
    rz = result['realized']
    st = result.get('seed_table', {}); seeds = result.get('seeds', [339])
    fmt = lambda v, d=2: ' / '.join(f"{x:.{d}f}" for x in v)   # one value per seed, in seed order
    seed_rows = ''.join(f"<tr><td>{LABELS[c]}</td><td>{fmt(t['correction_mae'])}</td><td>{fmt(t['median_response_gain'])}</td>"
                        f"<td>{fmt(t['boundary_sensitivity'], 3)}</td><td>{fmt(t['unshifted_alteration'])}</td></tr>" for c, t in st.items())
    if st and len(seeds) > 1:
        b, a = st['bpdr']['correction_mae'], st['augmentation_only']['correction_mae']
        separated = max(b) < min(a)
        collapsed = [seeds[i] for i in range(len(seeds)) if all(st[c]['median_response_gain'][i] < .1 for c in st)]
        learned = [x for x in seeds if x not in collapsed]
        ranking = ((f'Seeds {", ".join(map(str, learned))} learn for all three models; ' if learned else '')
                   + (f'seed{"s" if len(collapsed) > 1 else ""} {", ".join(map(str, collapsed))} collapsed to zero output for all three models within 1,500 updates, '
                      'the same failure seen on real data, so training is sensitive to initialisation and sampling order. ' if collapsed else '')
                   + ('Every BPDR run has lower correction error than every augmentation-only run.' if separated else
                      'BPDR and augmentation-only are not separated on correction error across seeds.'))
    else:
        ranking = 'Only one seed has been run, so the three models are comparable but not ranked.'
    oc = result.get('original_chain', {})
    verified = result['outcome'] == 'VERIFIED_FIXTURE_CANDIDATE'
    (out/'bpdr_explainer.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>BPDR transformer explainer</title>
<style>body{{font:16px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:0 16px;color:#1f2328}}img{{width:100%;border:1px solid #ddd}}
table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}
.note{{background:#fff8c5;border:1px solid #d4a72c;padding:8px 12px}}code{{background:#f4f4f4}}</style>
<h1>Boundary-Pitch Disentangled Repair (BPDR): what the transformer does</h1>
<p class="note"><b>Controlled proof of concept.</b> Everything on this page uses constructed hummed phrases and synthetic voices built with the component's own audio and feature code.
It shows that the proposed method works as specified in the implemented code. It is <b>not</b> evidence of accuracy on real singers; see section 5.</p>

<h2>1. The problem and the research gap</h2>
<p>A novice may sing a note slightly out of tune, and the system that later reads the melody also places note boundaries, which are sometimes wrong: a false split inside one note, or two notes merged.
A repair model that leans on those boundaries can change its correction whenever the boundaries change, even though the audio is the same. Existing context-aware correction relies on note segmentation and was not tested on false splits or merges.
<b>Gap:</b> can one model learn to <i>ignore</i> unreliable boundary changes while still <i>responding</i> correctly to a small local detuning, and leave acceptable singing alone?</p>

<h2>2. The proposed method</h2>
<p>Each accepted phrase gives four aligned views on the same frame grid: <b>A</b> original audio + nominal boundaries, <b>B</b> original audio + corrupted boundaries, <b>C</b> locally detuned audio + nominal boundaries, <b>D</b> detuned audio + corrupted boundaries.
A compact bidirectional transformer (4 layers, 128 hidden, 4 heads) predicts a correction in cents for every frame. Training terms:</p>
<ul><li><b>L<sub>A</sub></b> anchor: zero correction for originals, the compensating correction (-e) on the detuned note (implemented as squared error; see section 5)</li>
<li><b>L<sub>B</sub></b> boundary invariance: |r<sub>A</sub> - r<sub>B</sub>| and |r<sub>C</sub> - r<sub>D</sub>|, same audio must get the same repair</li>
<li><b>L<sub>D</sub></b> counter-detuning: r<sub>C</sub> - r<sub>A</sub> must equal -e on the detuned note and 0 elsewhere</li>
<li><b>L<sub>S</sub></b> smoothness of the correction curve on voiced frames</li></ul>
<p>Three matched models: <b>augmentation-only</b> (same four views, L<sub>A</sub>+L<sub>S</sub>), <b>frame-only</b> (no boundary input, 70 features), <b>BPDR</b> (all terms). Same data, initialisation, optimiser and budget ({evidence['train_records']} training records, 1,500 updates each).</p>

<h2>3. Results on held-out synthetic voices ({evidence['heldout_records']} records, voices never seen in training; seed 339)</h2>
<table><tr><th>model</th><th>correction error (cents)</th><th>no-correction error</th><th>error reduced</th><th>paired response gain (ideal 1)</th><th>boundary sensitivity (cents)</th><th>change to originals (cents)</th><th>change outside the note (cents)</th></tr>{rows}</table>
<p>Boundary sensitivity by corruption type (cents; frame-only is 0 by construction because it never sees boundaries):</p>
<table><tr><th>model</th><th>false split</th><th>merge</th><th>jitter</th></tr>{corr}</table>
<img src="learning_curves.png" alt="Held-out learning curves of the three models (seed 339)">
<p>All seeds (values listed in seed order {' / '.join(map(str, seeds))}; the plots and example above use seed 339):</p>
<table><tr><th>model</th><th>correction error (cents)</th><th>paired response gain</th><th>boundary sensitivity (cents)</th><th>change to originals (cents)</th></tr>{seed_rows}</table>
<p><b>Reading these results.</b> When training succeeds, all three transformers learn the counter-detuning response on voices they never saw (response gain near 1) while changing original singing by well under 1 cent.
{ranking} Boundary sensitivity is tiny for every boundary-aware model (a few hundredths of a cent), so this controlled set does <b>not</b> show the boundary-invariance advantage the proposal hypothesises;
testing it needs data where boundary changes actually mislead the augmentation-only model.</p>
<p><b>Design limitation.</b> In every constructed phrase the detuned note is also the longest note (the only one guaranteed to last 1.25-1.6 s), so a model can use "longest note" as a location cue.
That flatters the change-to-originals and change-outside-the-note numbers. Real phrases have no such rule.</p>

<h2>4. One held-out phrase, end to end</h2>
<p>Voice <code>{e(result['record']['voice_id'])}</code>, {result['record']['cents']:+g} cents injected on a held note, and a false split placed inside that note for views B and D.</p>
<img src="example_views.png" alt="Input pitch and the four predicted correction curves for BPDR and augmentation-only">
<p>The BPDR correction predicted for the detuned audio then goes through the component's own permission check, Praat rendering, re-measurement and rollback.
Outcome: <code>{e(result['outcome'])}</code>{(' - reasons: '+e(', '.join(result['permission_reasons'] or result['rollback_reasons']))) if not verified else ''}.
Measured on the detuned note against the original: detuned {rz['detuned_vs_original_cents']:+.1f} cents, after the BPDR correction {rz['corrected_vs_original_cents']:+.1f} cents.</p>
<p><b>Demo rule, not part of the proposal:</b> the frame curve becomes one edit by keeping only the largest contiguous span of voiced frames asking for at least 3 cents; values are not clipped.
In this phrase that rule, not the permission check, discarded the two small spikes on other notes.</p>
<p><b>The same chain on the unshifted original</b> (views A and B of the same phrase): outcome <code>{e(oc.get('outcome', '-'))}</code>{(' - '+e(', '.join(oc['permission_reasons']))) if oc.get('permission_reasons') else ''};
largest predicted correction on voiced frames {oc.get('max_abs_prediction_on_original_cents', float('nan')):.2f} cents (unvoiced frames are never edited), edit proposed {oc.get('proposed_max_abs_cents', float('nan')):.1f} cents.</p>
<h3>Original</h3><audio controls src="example_original.wav"></audio>
<h3>Detuned</h3><audio controls src="example_detuned.wav"></audio>
{'<h3>Corrected by the BPDR transformer</h3><audio controls src="example_bpdr_corrected.wav"></audio>' if verified else '<p>No corrected audio: the edit was not delivered.</p>'}

<h2>5. Status on real singing (honest)</h2>
<p>On real recordings (Jamendo separated vocals and Vocadito) the same transformer has not yet learned a correction that transfers to new singers.
Two real defects were found and fixed on the way: the frozen pitch analyser reads in 33-cent steps (now replaced by fine YIN pitch), and a mean-absolute anchor loss pinned the output at zero (now a squared error).
A held-out-singer test after 2,400 updates still showed no response on new voices. Real singing has natural note-to-note intonation of about 11-19 cents, comparable to the injected shifts,
whereas these controlled voices are steadier. Closing that gap (more and steadier reference recordings, participant recordings after ethics approval) is the next research step.
The working repair path for the demonstration is the proposal's rule-based condition.</p>

<h2>6. Where this lives in the code</h2>
<p><code>controlled_study.py</code> (data and training), <code>model.py</code> (transformer), <code>losses.py</code> + <code>fine_pitch_mse_check.py</code> (objective), <code>fine_pitch.py</code> (fine pitch input),
<code>repair_controller.py</code>, <code>framewise_renderer.py</code>, <code>framewise_audit.py</code> (permission, rendering, re-measurement, rollback), <code>bpdr_showcase.py</code> (this page).
Runs: <code>{e(str(data.relative_to(COMPONENT)))}</code>, <code>{e(str(run.relative_to(COMPONENT)))}</code>.</p>''', encoding='utf-8')


if __name__ == '__main__':
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else COMPONENT/'runs/demo/bpdr_explainer')
