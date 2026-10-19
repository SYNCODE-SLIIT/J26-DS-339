"""Maintain the recorded model walkthrough, without training or analysis writes."""
from pathlib import Path
import nbformat

WORKSPACE = Path(__file__).resolve().parents[1]
path = WORKSPACE / 'notebooks/07_learned_stable_note_recovery.ipynb'
nb = nbformat.read(path, 4)
nb.cells = nb.cells[:13]
for cell in nb.cells:
    cell.source = cell.source.replace('\ufffd', '-')
nb.cells[0].source = nb.cells[0].source.replace('new data, frozen evaluation and audible evidence', 'real data, measured results and experimental audio')
nb.cells += [
nbformat.v4.new_markdown_cell('''## 7. Completed revised pilot: a trained model, with failed broader criteria

The corrected preparation retains full recordings and places controlled shifts in measured stable voice. Both fixed tree families trained successfully. A reporting-key error interrupted the first summary; the same deterministic fits were repeated from cached prepared data after a mechanical fix. The declaration, original executing-source snapshots and technical-resume receipt are retained.

**The broader calibration and two-singer criteria failed.** They remain failed. No research model was adopted. A separate, explicitly experimental demo option exposes a selected verified listening example; it does not promote this failed pilot into a successful validation study. The example was selected after observing the diagnostic outputs and is not representative accuracy.

The same two evaluation singer identities were reused after the preparation correction. They were absent from model fitting, but these results are **development evidence**, not an untouched final test. Only one supplied valid controlled-shift labels.'''),
nbformat.v4.new_code_cell('''assert RECOVERY is not None
FINAL_DECLARATION=read(RECOVERY_DIRECTORY/'declaration.json')
assert sha(RECOVERY_DIRECTORY/'declaration.json')==RECOVERY['declaration_sha256']
assert sha(DATASET/'manifest.json')==FINAL_DECLARATION['manifest_sha256']
CHECKPOINT=Path(RECOVERY['checkpoint'])
assert sha(CHECKPOINT)==RECOVERY['checkpoint_sha256']
from bpdr_input.modeling.recovery_model import load_locked_diagnostic
MODEL,PAYLOAD=load_locked_diagnostic(CHECKPOINT)
assert MODEL['kind']=='extra_trees'
assert not RECOVERY['outer_proposal_gate_passes']
assert not RECOVERY['rendered_demo_gate_passes']
assert not RECOVERY['model_adopted']
print('Model:',MODEL['kind'],'/ trees:',MODEL['estimator'].n_estimators)
print('Locked operating policy:',PAYLOAD['operating_policy'])
print('Model SHA-256:',sha(CHECKPOINT))
print('Full-recording preparation:',FINAL_DECLARATION['preprocessing'])
print('Technical resume:',read(RECOVERY_DIRECTORY/'technical_resume.json'))
display(pd.DataFrame(RECOVERY['fit_receipts']).T)
calibration=[]
for r in read(RECOVERY_DIRECTORY/'calibration_all.json'):
    s=r['summary']
    calibration.append(dict(model=r['model'],**r['operating_policy'],singers=s['groups'],
        raw_mae_cents=s.get('raw_correction_mae'),permitted_mae_cents=s.get('permitted_correction_mae'),
        raw_original_mean_cents=s.get('raw_original_abs_mean'),passes=s['passes']))
display(pd.DataFrame(calibration))
print('Selection was locked before this run prepared its outer examples; calibration gate passed:',RECOVERY['selection']['calibration_gate_passes'])
coverage=[]
for split in splits:
    pp=read(RECOVERY_DIRECTORY/(split+'_pairs.json'))
    ss=read(RECOVERY_DIRECTORY/(split+'_sources.json'))
    coverage.append(dict(split=split,originals=len(ss),signed_attempts=len(pp),
        audited=sum('excluded' not in p for p in pp),excluded=sum('excluded' in p for p in pp)))
display(pd.DataFrame(coverage))'''),
nbformat.v4.new_markdown_cell('''## 8. Delivered WAV results: all attempts remain visible

An intervention is scoreable only after its independent construction audit passes. Twelve of sixteen evaluation interventions failed this prerequisite; their failures are retained and cannot supply target labels. All four scoreable interventions enter the delivered result, including withholding and an edit that did not improve the target.

The exact encoded WAV files were reloaded and audited. Sample count, timing, outside-support identity, pitch movement and clipping checks apply to the delivered bytes. These checks establish numerical behavior, not intended-note truth or a human quality judgment. The scored frame masks are fixed before output scoring; output cannot remove difficult frames from its denominator.'''),
nbformat.v4.new_code_cell('''VERIFICATION_PATH=RECOVERY_DIRECTORY/'rendered_verification.json'
assert sha(VERIFICATION_PATH)==RECOVERY['rendered_verification_sha256']
VERIFICATION=read(VERIFICATION_PATH)
assert VERIFICATION['checkpoint_sha256']==sha(CHECKPOINT)
assert VERIFICATION['declaration_sha256']==RECOVERY['declaration_sha256']
assert not VERIFICATION['rendered_demo_gate_passes']
assert len(VERIFICATION['originals'])==4 and len(VERIFICATION['shifts'])==16
for r in VERIFICATION['originals']:
    original,sr=sf.read(r['input'],dtype='float32')
    output,rate=sf.read(r['output'],dtype='float32')
    assert r['preserved'] and sr==rate and np.array_equal(original,output)
valid=[r for r in VERIFICATION['shifts'] if 'excluded' not in r]
assert len(valid)==4
for r in valid:
    report=read(Path(r['directory'])/'model_report.json')
    item=report['candidate'] if r['delivered_preview'] else report['handoff']['accepted_audio']
    assert sha(r['output'])==(item.get('sha256') or item['checksum'])
    if r['delivered_preview']:
        assert report['transaction']['delivered_wav_audit']['passed']
        assert report['transaction']['delivered_wav_audit']['sha256']==sha(r['output'])
    assert r['missing_output_scored_frames']==0
display(pd.DataFrame(VERIFICATION['shifts']).drop(columns=['input','reference','target_path','output','directory'],errors='ignore'))
display(pd.DataFrame(VERIFICATION['by_singer']).T)
print('Originals preserved:',sum(r['preserved'] for r in VERIFICATION['originals']),'of 4')
print('Verified improvements:',sum(r['verified_improvement'] for r in valid),'of 4 scoreable / 16 attempted interventions')
print('Declared broader gate:',VERIFICATION['rendered_demo_gate_passes'])
fig,ax=plt.subplots(figsize=(10,4))
x=np.arange(len(valid)); labels=[f"{r['singer']} / {Path(r['input']).parent.name} / {r['cents']:+g}c" for r in valid]
ax.bar(x-.18,[r['input_mae_cents'] for r in valid],.36,label='Input retained baseline',color='#9b645e')
ax.bar(x+.18,[r['output_mae_cents'] for r in valid],.36,label='Delivered output (withholds included)',color='#477b88')
ax.set(ylabel='Mean absolute local pitch error (cents)',title='All four scoreable development interventions; one singer')
ax.set_xticks(x,labels,rotation=15);ax.legend();plt.tight_layout();plt.show()'''),
nbformat.v4.new_markdown_cell('''## 9. Hear the actual learned output

The listening example below is the **first verified improvement in the locked source/shift order**, selected after the diagnostic run. A +30-cent shift was added to a real recording; the learned ExtraTrees model receives only that shifted audio. It uses its own measured local pitch context, not the known clean reference, target mask, shift interval, C/D-scale label or requested correction.

The clean reference is used only to construct interventions and score output. The model does not know whether a natural small pitch movement was a mistake or intentional expression. User confirmation remains necessary before an experimental preview is exported. Audio players keep recorded gain.'''),
nbformat.v4.new_code_cell('''CARD_PATH=WORKSPACE/'results/model_recovery/demo_model.json'
CARD=read(CARD_PATH)
assert CARD['validation_status']=='exploratory_preview_failed_generalization_gate'
assert CARD['research_gates_passed'] is False and CARD['requires_user_review']
assert CARD['checkpoint_sha256']==sha(CHECKPOINT)
assert CARD['verification_sha256']==sha(VERIFICATION_PATH)
display(CARD['metrics'])
SELECTED=next(r for r in valid if r['verified_improvement'])
print('Selected example local MAE:',SELECTED['input_mae_cents'],'->',SELECTED['output_mae_cents'],'cents')
for label,key in [('Input: real voice with known added shift','input'),('Actual learned output WAV','output'),('Unshifted performance: scoring reference only','reference')]:
    display(Markdown('**'+label+'**'))
    display(Audio(filename=SELECTED[key],normalize=False))
from bpdr_input.repair.rule_correction import fine_pitch
with np.load(SELECTED['target_path']) as z:
    times=z['times']; fixed=z['mask'].astype(bool)&(z['target']!=0)
ref,sr=sf.read(SELECTED['reference'],dtype='float32')
before,_=sf.read(SELECTED['input'],dtype='float32')
after,_=sf.read(SELECTED['output'],dtype='float32')
reference_pitch=fine_pitch(ref,sr,times)
before_error=fine_pitch(before,sr,times)-reference_pitch
after_error=fine_pitch(after,sr,times)-reference_pitch
assert np.isfinite(before_error[fixed]).all() and np.isfinite(after_error[fixed]).all()
np.testing.assert_allclose(np.abs(before_error[fixed]).mean(),SELECTED['input_mae_cents'],atol=.001)
np.testing.assert_allclose(np.abs(after_error[fixed]).mean(),SELECTED['output_mae_cents'],atol=.001)
fig,ax=plt.subplots(figsize=(11,4))
ax.plot(times[fixed],before_error[fixed],label='Input residual',color='#9b645e')
ax.plot(times[fixed],after_error[fixed],label='Output residual',color='#477b88')
ax.axhline(0,color='#333333',linewidth=.7)
ax.set(xlabel='Seconds from recording start (fixed scored frames)',ylabel='Cents relative to unshifted performance',title='Selected verified example; reference used after inference only')
ax.legend();plt.tight_layout();plt.show()'''),
nbformat.v4.new_markdown_cell('''## 10. What can be said to the panel

- A small ExtraTrees model was trained on controlled modifications of original VocalSet human recordings, using singer-disjoint fitting and calibration groups. This model is not the historical Transformer and does not demonstrate a BPDR advantage.
- Two of four scoreable development interventions improved, both +30-cent shifts. Mean delivered local error across all four, including withholding and a non-improving edit, fell from 27.44 to 14.66 cents (46.6%). This is error reduction on four controlled examples from one singer, **not 46.6% general singing accuracy**.
- Four of four unshifted working recordings were preserved. Twelve of sixteen attempted interventions had no valid labels. Broader calibration and singer-coverage criteria failed; those failures remain unchanged.
- The demo exposes the model as an experimental optional preview. It does not automatically replace the accepted original. A separate user-reviewed package can proceed to the existing melody extractor when acoustic checks pass.
- The remaining work is more diverse reviewed singing, reliable training labels and independent singer validation. Arbitrary reading cannot establish an intended melody. Do not infer correctness solely from a presumed key or force every performance onto a scale.

**Demo:** run `Run-Demo.ps1`, select the experimental learned repair example, process, and compare input/output. The app runs inference from saved weights; it does not train while presenting.'''),
nbformat.v4.new_code_cell('''ui_path=WORKSPACE/'results/demo_sessions/ui_recovery_check.json'
if ui_path.is_file():
    UI=read(ui_path)
    assert UI['passed'] and UI['model_sha256']==sha(CHECKPOINT)
    assert UI['human_musical_judgment'] is False
    display(UI)
else:
    print('Live UI execution receipt is not yet available; no UI success is inferred.')
print('Notebook performed no fitting, download or intervention generation.')
print('Research model adoption remains:',RECOVERY['model_adopted'])
print('Current original manifest SHA-256:',sha(DATASET/'manifest.json'))''')]
nb.metadata['bpdr_verification'] = dict(minimum_plots=4, minimum_audio=5)
nbformat.validate(nb)
nbformat.write(nb, path)
print(path)
