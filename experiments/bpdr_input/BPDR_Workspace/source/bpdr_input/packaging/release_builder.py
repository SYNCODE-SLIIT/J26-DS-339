"""Assemble a truthful presentation release from completed, immutable studies.

prepare: archive maintained sources, refresh the two notebooks and current docs.
seal: require fresh successful notebook execution, hash/copy deliverables, zip.
Neither command trains, generates audio, approves research or enables repair.
"""
from datetime import datetime, timezone
from bpdr_input import SOURCE_ROOT
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import uuid
import zipfile

from bpdr_input.experiments.evaluation.transfer_study import COMPONENT, ROOT, CONDITIONS, read, write, sha, frozen_runs


def selected_run(name):
    return Path(read(ROOT/(name+'_current.json'))['path'])


def md(text):
    import nbformat
    return nbformat.v4.new_markdown_cell(text)


def code(text):
    import nbformat
    return nbformat.v4.new_code_cell(text)


def prepare():
    import nbformat
    from bpdr_input.reporting.bpdr_showcase import load_model
    out=COMPONENT/'runs/presentation_release'/uuid.uuid4().hex;out.mkdir(parents=True)
    (out/'executed_presentation_release.py').write_bytes(Path(__file__).read_bytes())
    previous_pointer=COMPONENT/'runs/presentation_release/prepared.json'
    if previous_pointer.exists():
        previous=Path(read(previous_pointer)['path'])/'prior_maintained_sources'
        if previous.exists():shutil.copytree(previous,out/'pre_presentation_recovery')
    recovery=out/'prior_maintained_sources';recovery.mkdir()
    for name in ['01_input_validation_and_analysis.ipynb','02_data_review_and_paired_experiments.ipynb','evidence_index.json']:
        shutil.copy2(COMPONENT/'notebooks'/name,recovery/name)
    studies={name:selected_run(name) for name in ['transfer','adapt','probe','statistics','baselines']}
    evidence={name:read(path/'evidence.json') for name,path in studies.items()}
    from bpdr_input.reporting.paired_statistics import paired_interval
    adapt_rows={c:read(studies['adapt']/c/'rows.json') for c in CONDITIONS}
    adapt_intervals={ds:{c:paired_interval([r for r in adapt_rows['bpdr'] if r['dataset']==ds],
        [r for r in adapt_rows[c] if r['dataset']==ds]) for c in CONDITIONS if c!='bpdr'} for ds in ['vocadito','jamendo']}
    write(out/'adaptation_paired_uncertainty.json',adapt_intervals)
    # Retention check: an adaptation is not automatically an improvement over
    # the working synthetic model. Score all three final adapted checkpoints.
    import torch
    from bpdr_input.experiments.learning.controlled_study import load_pair, summary
    from bpdr_input.experiments.learning.fine_pitch_relative_check import relative_pair
    from bpdr_input.experiments.evaluation.transfer_study import predictions, score_predictions
    torch.set_num_threads(2)
    data=Path(read(COMPONENT/'runs/controlled_study/current.json')['path'])
    synth_records=[r for r in read(data/'catalog.json')['records'] if r['split']=='heldout' and r['plateau_frames']>=10]
    pairs=[relative_pair(load_pair(r)) for r in synth_records]
    retention={};adapt_challenge={}
    challenge_pairs=[relative_pair(load_pair(r)) for r in read(studies['probe']/'catalog.json')['records']]
    write(out/'retention_declaration.json',dict(optimizer_updates=0,selection='Every final adapted model, fixed 72 synthetic evaluation variants, no checkpoint selection',
        checkpoints={c:sha(studies['adapt']/(c+'.pt')) for c in CONDITIONS},source_sha256=sha(__file__)))
    for c in CONDITIONS:
        model,norm,width=load_model(studies['adapt']/(c+'.pt'))
        rows=[score_predictions(p,predictions(model,norm,width,p,'cpu')) for p in pairs]
        retention[c]=summary(rows)
        adapt_challenge[c]=summary([score_predictions(p,predictions(model,norm,width,p,'cpu')) for p in challenge_pairs])
        print('Retention',c,retention[c]['median_response_gain'],flush=True)
    write(out/'adaptation_synthetic_retention.json',retention)
    write(out/'adaptation_nonlongest_challenge.json',adapt_challenge)
    selection=read(COMPONENT/'runs/presentation_preview_selection.json')
    if not selection['checks_pass'] or not selection['selected_for_presentation_preview']:
        raise ValueError('Mixed-data preview did not pass its declared runtime checks')
    if sha(studies['adapt']/'bpdr.pt')!=selection['sha256']:raise ValueError('Selected checkpoint changed')
    runtime=read(COMPONENT/'runs/presentation_runtime_checks_mixed.json')
    latency=read(COMPONENT/'runs/presentation_latency_mixed/evidence.json')
    base=dict(frozen_runs())[339]
    (out/'models').mkdir()
    for c in CONDITIONS:
        shutil.copy2(studies['adapt']/(c+'.pt'),out/'models'/(c+'.pt'))
    for seed,folder in frozen_runs():
        target=out/'models/synthetic_reference'/str(seed);target.mkdir(parents=True)
        for c in CONDITIONS:shutil.copy2(folder/(c+'.pt'),target/(c+'.pt'))
    model,_,_=load_model(out/'models/bpdr.pt')
    card=dict(version='bpdr-presentation-1',created_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint='models/bpdr.pt',checkpoint_sha256=sha(out/'models/bpdr.pt'),architecture=model.describe(),
        origin=str(studies['adapt']),selection='Experimental presentation preview: mixed-data seed 339 retains synthetic response/accuracy, real originals change <1c, fixed runtime correction and identity checks pass. Not research adoption; real response gate still fails.',
        training='Synthetic pretraining: 384 variants / 16 generated voices, 1500 updates. Adaptation: 995 real variants (77 Vocadito / 918 Jamendo) plus 384 synthetic variants, 300 updates. Squared anchor, smoothness and BPDR response terms.',
        validation='Pretraining: 72 variants / six excluded synthetic voice settings, seeds 339/341 learned and 340 collapsed. Mixed adaptation: seed 339 only, same 72-case retention plus 24-case nonlongest-note diagnostic. No real-singer repair gate pass.',
        real_data='Frozen transfer evaluated all three models and three seeds on 304 real-audio development variants. Matched real+synthetic adaptation trained separately.',
        normalizer='Embedded training-only mean/scale; 71 inputs, phrase-relative fine YIN pitch; PESTO voicing/confidence retained',
        runtime='Audio-only provisional boundaries, +40ms jitter/removal alternatives; explicit-voicing v2 adapter retains legitimate A4=0 cents; 3-cent preview span rule, fixed uncalibrated permission gates',
        intended_use='Local presentation and controlled engineering experiments',
        unsupported_use='Automatic musical acceptance, reliable real-singer correction, participant-level accuracy, demonstrated methodological superiority',
        prepared_handoff_enabled=False,research_model_adopted=False,research_ready_examples=0,
        limitations=['Seed-sensitive training','Synthetic voices are not people','Real-data generalisation gate remains open',
                    'YIN is used for fine input and numerical remeasurement, with PESTO as coarse cross-check',
                    'Historical training caches use a legacy pitch!=0 availability test; the audio-only v2 adapter corrects legitimate A4 handling without altering old caches or results',
                    'Experienced review deferred; no participant collection, blinded listening or downstream extraction F1 study completed'])
    write(out/'model_card.json',card)
    sources={str(path/'evidence.json'):sha(path/'evidence.json') for path in studies.values()}
    sources.update({str(COMPONENT/'runs/presentation_runtime_checks_mixed.json'):sha(COMPONENT/'runs/presentation_runtime_checks_mixed.json'),
                    str(COMPONENT/'runs/presentation_latency_mixed/evidence.json'):sha(COMPONENT/'runs/presentation_latency_mixed/evidence.json')})
    for name in ['presentation_contract_checks_mixed.json','presentation_training_environment.json','presentation_full_regression.log','presentation_ui_test_verified.log','presentation_preview_selection.json','presentation_selected_example_measurement.json']:
        sources[str(COMPONENT/'runs'/name)]=sha(COMPONENT/'runs'/name)
    manifest=dict(version='presentation-studies-1',created_utc=datetime.now(timezone.utc).isoformat(),
        studies={k:str(v) for k,v in studies.items()},source_hashes=sources,model_card_sha256=sha(out/'model_card.json'),
        contract_checks=read(COMPONENT/'runs/presentation_contract_checks_mixed.json'),training_environment=read(COMPONENT/'runs/presentation_training_environment.json'),
        selected_preview=selection,selected_example=read(COMPONENT/'runs/presentation_selected_example_measurement.json'),
        synthetic_runs={str(s):str(p) for s,p in frozen_runs()},runtime_checks=runtime,latency=latency,
        gates=dict(E='open: real-voice generalisation / BPDR advantage not established',F='open: numerical candidates verified, real musical repair unvalidated',
                   D_J4='deferred by owner; no independent musical approvals'),research_ready_examples=0,
        adaptation_paired_intervals=adapt_intervals,adaptation_synthetic_retention=retention,adaptation_nonlongest_challenge=adapt_challenge,
        final_test_accessed=False,shared_pipeline_unchanged=evidence['transfer']['shared_pipeline_unchanged'])
    write(out/'study_manifest.json',manifest)
    index_path=COMPONENT/'notebooks/evidence_index.json';index=read(index_path)
    index['updated']='2026-10-17'
    index['evidence']['presentation_manifest']=str((out/'study_manifest.json').relative_to(COMPONENT))
    for name,path in studies.items():
        index['evidence']['presentation_'+name]=str((path/'evidence.json').relative_to(COMPONENT))
    index['current_gate'].update(work='Working synthetic BPDR and audio-only experimental runtime; completed real transfer, mixed adaptation and challenge studies',
        next_action='Gate E: real-voice learning remains a research requirement. Preserve this release; future work needs stronger audited reference coverage and seed stability. D/J4 review, participant study and downstream benefit remain pending.',
        pending_prerequisite='Independent real-voice preservation-safe recovery and research validation; not an implementation approval request',
        presentation_ready=True,presentation_scope='Reproducible engineering demonstration, not a completed main research study',
        presentation_manifest_sha256=sha(out/'study_manifest.json'),model_adopted=False)
    write(index_path,index)
    # Notebook 1 stays the technical-validation walkthrough; replace stale current claims.
    n1=nbformat.read(COMPONENT/'notebooks/01_input_validation_and_analysis.ipynb',as_version=4)
    n1.cells[2].source='''## Current scope and presentation route

This notebook demonstrates original retention, supported-file checks, channel handling, format conversion, measured acoustics, unchanged technical acceptance and controlled stops. It does not claim musical acceptance.

Notebook 2 is the current model walkthrough: the four-view BPDR method, all three synthetic seeds, real Vocadito/Jamendo transfer and adaptation, the nonlongest-note challenge, numerical waveform repair and audio-only inference. The local interface is started by `Run-Presentation.ps1`.

The mixed-data preview is selected after synthetic-retention and runtime checks; all synthetic references are preserved. Research Gates E/F remain open; D/J4 independent musical review remains deferred. Downstream audio stays unchanged; numerically verified candidates are separate comparison artifacts. No participant, listener, downstream-benefit or final-test result is invented.

The previous chronological discussion is preserved in dated executed notebooks and the presentation release recovery folder. Use the current evidence index and notebook 2 for the latest results.'''
    import re
    for cell in n1.cells:
        if cell.cell_type=='markdown':
            cell.source=re.sub(r'\*\*Current - Gate E diagnosis complete.*?(?=\n\n|$)',
                '**Current presentation status:** a working synthetic BPDR model and audio-only experimental runtime are recorded in notebook 2. Real transfer and matched real+synthetic adaptation are reported with failures retained. Gate E/F remain open; D/J4 review is deferred. This notebook demonstrates the unchanged technical path.',cell.source,flags=re.S)
            cell.source=cell.source.replace('A model\'s correction predictions have not been credited as accepted repaired audio.',
                'Numerically verified model candidates exist as separate previews; the downstream package remains unchanged, and no musical acceptance is inferred.')
    n1.cells[-1].source='''**What this demonstrates:** original retention, technical processing, controlled stops, accepted-only packaging, original timing and measured acoustics. Audio widgets aid playback; saved WAV hashes identify the actual artifacts.

**Implemented and explained in notebook 2:** real/synthetic preparation, four views, matched Transformers and losses, completed studies, numerical candidate verification, rollback and annotation-free live inference.

**Still research requirements:** independent reference suitability, stable real-singer recovery, preservation/acceptance calibration, the BPDR advantage, consented participant and blinded listening studies, downstream extraction benefit and a reserved final test. Research-ready examples remain zero. A verified numerical preview is not musical approval or prepared-audio adoption.'''
    for cell in n1.cells:
        if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
    nbformat.write(n1,COMPONENT/'notebooks/01_input_validation_and_analysis.ipynb')
    notebook=nbformat.v4.new_notebook(cells=notebook_cells(),metadata=dict(
        kernelspec=dict(display_name='BPDR component (uv)',language='python',name='bpdr-local'),
        bpdr_verification=dict(minimum_plots=6,minimum_audio=6)))
    nbformat.write(notebook,COMPONENT/'notebooks/02_data_review_and_paired_experiments.ipynb')
    report=report_text(card,manifest,evidence)
    (COMPONENT/'docs/Presentation_Readiness_2026-10-17.md').write_text(report,encoding='utf-8')
    (out/'README.md').write_text(report,encoding='utf-8')
    intro=('**Current presentation release — 17 October 2026:** the mixed-data BPDR presentation preview uses Vocadito, Jamendo and synthetic training; the working synthetic references are retained. '
           'A local recording/upload interface, audio-only inference, real Vocadito/Jamendo transfer, matched mixed-data adaptation, '
           'nonlongest-note challenge and fresh evidence notebooks are implemented. See '
           '[presentation readiness and proposal traceability](docs/Presentation_Readiness_2026-10-17.md). '
           'E/F remain open; D/J4 review is deferred; the prepared-audio hand-off stays off. '
           'These are controlled engineering results, not a completed human research study.\n\n')
    for rel in ['README.md','docs/PP1_Master_Plan.md','docs/PP1_Phase_Execution_Plan.md','docs/PP1_Evidence_Tracker.md']:
        p=COMPONENT/rel;old=p.read_text(encoding='utf-8')
        shutil.copy2(p,recovery/p.name)
        marker='## Historical engineering log\n\nThe material below is retained history; the linked current release supersedes its older next-step statements.\n\n'
        if old.startswith('**Current presentation release') and marker in old:old=old.split(marker,1)[1]
        prefix=intro if rel=='README.md' else intro.replace('(docs/Presentation_Readiness_2026-10-17.md)','(Presentation_Readiness_2026-10-17.md)')
        p.write_text(prefix+'## Historical engineering log\n\nThe material below is retained history; the linked current release supersedes its older next-step statements.\n\n'+old,encoding='utf-8')
    write(COMPONENT/'runs/presentation_release/prepared.json',dict(path=str(out)))
    print('PREPARED',out,flush=True)
    return out


def notebook_cells():
    return [
    md('''# BPDR: model, real-data experiments and presentation demonstration

This notebook reads completed evidence. It never trains a model, regenerates extraction, changes a review, or enables repaired-audio delivery.

**Current:** the presentation-preview Transformer uses synthetic pretraining plus real Vocadito/Jamendo adaptation. It retains working correction on constructed humming and passes a fixed audio-only candidate/identity check. Useful real-singer repair and the additional benefit of BPDR remain unestablished. Synthetic reference weights are retained; E/F open, D/J4 review deferred.

Follow sections 1–8 for the panel. The prior detailed chronology is retained in dated notebook snapshots and the release's recovery folder.'''),
    code('''import json, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Audio
COMPONENT=Path.cwd()
if not (COMPONENT/'src/bpdr_input').exists(): COMPONENT=COMPONENT.parent
sys.path.insert(0,str(COMPONENT/'src'))
from bpdr_input.notebook_evidence import load_index
from bpdr_input.presentation_study import read, sha
INDEX,EVIDENCE=load_index(COMPONENT)
M=read(EVIDENCE['presentation_manifest']); RELEASE=EVIDENCE['presentation_manifest'].parent
for p,h in M['source_hashes'].items(): assert sha(p)==h, p
assert sha(EVIDENCE['presentation_manifest'])==INDEX['current_gate']['presentation_manifest_sha256']
assert sha(RELEASE/'model_card.json')==M['model_card_sha256']
CARD=read(RELEASE/'model_card.json')
assert sha(RELEASE/CARD['checkpoint'])==CARD['checkpoint_sha256']
print('Verified completed evidence; prepared hand-off:',CARD['prepared_handoff_enabled'])
display(pd.DataFrame([{'gate':k,'status':v} for k,v in M['gates'].items()]))'''),
    md('''## 1. Proposal and dataset roles

Proposal sections 1.3 and 3.2–3.6 require a controlled comparison, not merely a working network. Vocadito is the proposed public development collection; participant research requires ethics approval and consent. Jamendo is a later exploratory source with predicted annotations.

Real-audio engineering pool: 995 training variants (77 Vocadito, 918 Jamendo), 304 development variants (19 Vocadito, 285 Jamendo). Augmentations are not independent singers. Vocadito human annotations describe performed pitch/notes, not guaranteed ideal intonation. Jamendo artist identities are proxies, not verified singer identities.

The synthetic study uses 384 training / 72 evaluation variants from 16 / 6 generated voice settings. These are not human participants. The real adaptation adds actual recordings to training; it does not certify them as approved musical references.'''),
    code('''display(pd.DataFrame([
 {'source':'Vocadito','train_variants':77,'development_variants':19,'train_groups':11,'development_groups':3,'reference':'human performed F0/notes; musical approval pending'},
 {'source':'Jamendo','train_variants':918,'development_variants':285,'train_groups':23,'development_groups':12,'reference':'separated vocals; artist proxies, model-estimated notes'},
 {'source':'Constructed humming','train_variants':384,'development_variants':72,'train_groups':16,'development_groups':6,'reference':'known construction; generated voice settings, not people'}]))
fine=COMPONENT/'runs/fine_pitch/66e3619e611241dd80d33a32578b73c6'
tr=read(fine/'training_catalog.json')['records'];dv=read(fine/'development_catalog.json')['records']
ds=lambda r:r.get('dataset','vocadito' if r['phrase_id'].startswith('vocadito') else 'jamendo')
assert not {(ds(r),r['singer_id']) for r in tr}&{(ds(r),r['singer_id']) for r in dv}
print('Train/development singer or artist-proxy groups do not overlap. Final test not accessed.')'''),
    md(r'''## 2. Four views, model and losses

A = original + nominal boundaries; B = identical audio + changed boundaries. C = locally detuned audio + nominal boundaries; D = identical detuned audio + changed boundaries. Detuning is ±10/20/30 cents inside a held note with smooth ramps. Split/merge/jitter alter timing metadata only.

The model has four bidirectional Transformer layers, hidden width 128, four heads, feed-forward width 512 and dropout 0.1. Inputs contain 64 log-mel bands plus pitch, confidence, voicing, pitch change, energy, onset and a timing-boundary column (71 total). Frame-only omits that final column.

$L=L_A+\lambda_B L_B+\lambda_D L_D+\lambda_S L_S$. Here $L_A$ is squared error to zero for originals and the known compensating correction for detuned views. $L_B$ is mean absolute boundary-response difference; $L_D$ compares paired response with negative injected detuning; $L_S$ penalises abrupt correction changes. BPDR adds the response terms after 100 updates; controls use anchor + smoothness. All three use matched data, schedule and update budgets.

Fine YIN pitch replaces the earlier 33-cent-quantized PESTO pitch columns; PESTO confidence/voicing remains. Pitch is phrase-relative and normalization is training-only. This is a documented change from the original single-estimator plan. No reference target or intervention mask is an acoustic input.'''),
    code('''display(pd.DataFrame([CARD['architecture']]))
from bpdr_input.losses import LossWeights
print('Loss weights:',LossWeights())
fine_build=read(fine/'build_evidence.json')
display(pd.DataFrame(fine_build['realized_shift_training']).T[['records','median_cents','p10','p90']])'''),
    md('''## 3. What the synthetic models actually learned

All three models trained for 1,500 updates per seed. Display every seed, including the failed seed 340. Correction error measures requested cents against the known compensating target. Response gain is the difference between detuned and original predictions divided by the desired correction: 1 is ideal and 0 means no detuning-specific response.

Low boundary sensitivity by itself is not success: a model that always outputs zero also has low sensitivity. No consistent BPDR superiority is established across seeds.'''),
    code('''synth={s:read(Path(p)/'evidence.json') for s,p in M['synthetic_runs'].items()}
rows=[]
for seed,e in synth.items():
 for condition,r in e['results'].items(): rows.append(dict(seed=seed,model=condition,**r['heldout']))
synthetic_table=pd.DataFrame(rows)
display(synthetic_table[['seed','model','correction_mae','zero_mae','median_response_gain','boundary_sensitivity','unshifted_alteration']])
fig,axs=plt.subplots(1,2,figsize=(12,4))
for c in ['augmentation_only','frame_only','bpdr']:
 x=synthetic_table[synthetic_table.model==c];axs[0].plot(x.seed,x.correction_mae,'o-',label=c);axs[1].plot(x.seed,x.median_response_gain,'o-',label=c)
axs[0].axhline(19.3056,color='grey',ls=':');axs[0].set(ylabel='Correction error (cents)',xlabel='Seed')
axs[1].axhline(1,color='grey',ls=':');axs[1].set(ylabel='Paired response gain (ideal 1)',xlabel='Seed');axs[0].legend();plt.tight_layout();plt.show()'''),
    code('''fig,axs=plt.subplots(1,2,figsize=(12,4))
for c,r in synth['339']['results'].items():
 x=r['curve'];axs[0].plot([p['update'] for p in x],[p['heldout']['correction_mae'] for p in x],label=c)
 axs[1].plot([p['update'] for p in x],[p['heldout']['median_response_gain'] for p in x],label=c)
axs[0].set(xlabel='Training update',ylabel='Synthetic evaluation error (cents)');axs[1].set(xlabel='Training update',ylabel='Response gain')
axs[0].legend();plt.tight_layout();plt.show()
stats=read(EVIDENCE['presentation_statistics']);display(pd.DataFrame([dict(seed=s,control=c,**v) for s,rr in stats['synthetic'].items() for c,v in rr.items()]))
print('95% intervals resample synthetic voice settings, not people. Seeds remain separate.')'''),
    md('''## 4. Does it rely on the longest-note shortcut?

The original constructed training always shifts the longest note. A new frozen-model challenge shifts a nonlongest note in each of 24 phrases, with eight new generated voice settings. All cases are retained; no model is refitted on this challenge. It probes one confound, not the full synthetic-to-real gap.'''),
    code('''probe=read(EVIDENCE['presentation_probe'])
display(pd.DataFrame([dict(model=c,**r['all']) for c,r in probe['results'].items()]))
fig,ax=plt.subplots(figsize=(7,3.5));names=list(probe['results']);ax.bar(names,[probe['results'][c]['all']['correction_mae'] for c in names])
ax.axhline(20,color='grey',ls=':',label='No correction');ax.set(ylabel='Correction error (cents)',title='New synthetic phrases: shifted note is never longest');ax.legend();plt.xticks(rotation=12);plt.tight_layout();plt.show()'''),
    md('''## 5. Use of real Vocadito and Jamendo recordings

First, all nine frozen model/seed combinations were evaluated on the same 304 real-audio development variants, without fitting or refitting normalization. Engineering gate: each dataset must have median paired response ≥0.25, original alteration ≤1 cent, and positive correction gain. This is a development diagnostic, not a final test.

Second, a bounded transfer-learning attempt uses 995 real training variants plus 384 synthetic variants: 300 new updates per model, seed 339, two Vocadito + two Jamendo + four synthetic draws per update, learning rate 0.0001. The pretrained normalizer is frozen; BPDR response losses start after 100 updates. Train/development groups are separate. All three final checkpoints and failures are kept. There is no development early stopping or extra retry budget.'''),
    code('''transfer=read(EVIDENCE['presentation_transfer']);adapt=read(EVIDENCE['presentation_adapt']);rr=[]
for seed,conditions in transfer['results'].items():
 for c,result in conditions.items():
  for ds,s in result['by_dataset'].items():rr.append(dict(stage='frozen synthetic',seed=seed,model=c,dataset=ds,passes=result['engineering_gate_passes'],**s))
for c,result in adapt['results'].items():
 for ds,s in result['by_dataset'].items():rr.append(dict(stage='real + synthetic adaptation',seed='339',model=c,dataset=ds,passes=result['engineering_gate_passes'],**s))
real_table=pd.DataFrame(rr)
display(real_table[['stage','seed','model','dataset','records','correction_mae','median_response_gain','unshifted_alteration','passes']])
fig,axs=plt.subplots(1,2,figsize=(12,4))
for ax,ds in zip(axs,['vocadito','jamendo']):
 names=['augmentation_only','frame_only','bpdr'];y=np.arange(3)
 for offset,stage in [(-.18,'frozen synthetic'),(.18,'real + synthetic adaptation')]:
  x=real_table[(real_table.dataset==ds)&(real_table.seed=='339')&(real_table.stage==stage)].set_index('model')
  ax.barh(y+offset,[x.loc[c,'median_response_gain'] for c in names],height=.32,label=stage)
 ax.set_yticks(y,['Augmentation','Frame-only','BPDR']);ax.axvline(.25,ls=':',color='grey');ax.set(title=ds,xlabel='Median paired response gain; gate 0.25');ax.legend(fontsize=8)
plt.tight_layout();plt.show()
display(pd.DataFrame([dict(seed=s,dataset=ds,control=c,**v) for s,dd in stats['real_transfer'].items() for ds,rr in dd.items() for c,v in rr.items()]))'''),
    code('''display(pd.DataFrame([dict(dataset=ds,control=c,**v) for ds,rr in M['adaptation_paired_intervals'].items() for c,v in rr.items()]))
display(pd.DataFrame([dict(model=c,**v) for c,v in M['adaptation_synthetic_retention'].items()]))
display(pd.DataFrame([dict(model=c,**v) for c,v in M['adaptation_nonlongest_challenge'].items()]))
print('Adaptation intervals use the same real groups; one adaptation seed only, no training-stability claim.')'''),
    md('''### Five proposal conditions and preservation

Original audio is the zero-correction reference. The existing conservative rule estimates the singer's tuning offset and requests constant corrections only on stable held notes. The table below scores its raw requested curve on the same 304 variants as the neural models. It uses existing fine-pitch archives and fixed thresholds, without calibration. These prediction scores are not rendered-audio results or a matched-acceptance-coverage study. Musical harmful-edit rate remains unknown without independent references/listening.'''),
    code('''baselines=read(EVIDENCE['presentation_baselines']);five=[]
for c,r in baselines['results'].items():
 for ds,s in r['by_dataset'].items():five.append(dict(model=c,dataset=ds,**s))
for c,r in transfer['results']['339'].items():
 for ds,s in r['by_dataset'].items():five.append(dict(model=c,dataset=ds,**s))
display(pd.DataFrame(five)[['model','dataset','correction_mae','median_response_gain','unshifted_alteration','outside_alteration']])
old=read(COMPONENT/'runs/rule_evaluation/b3c92d055b7e45e48e553ad9a796c726/evidence.json')
display(pd.DataFrame(old['summary']).T[['originals','originals_with_any_edit','detuned','detuned_shift_edited']])
print('Earlier rule decisions are separate engineering evidence; no musical harmful-edit rate is inferred.')'''),
    md('''## 6. End-to-end correction and audio-only runtime

The recorded controlled example supplies nominal/corrupted boundary metadata to the model. Its prediction passes permission, Praat rendering, remeasurement and rollback checks. The audio-only runtime is a separate check: it derives provisional boundaries from the input waveform and does not read note annotations, reference audio or target masks. It compares nominal, jittered and removed-boundary predictions. Permission limits are fixed engineering values, not a calibrated confidence policy.

The first three playback items below are the frozen synthetic-pretraining reference example. The later runtime playbacks and selected-example measurements use the mixed-data preview checkpoint. That selection is for presentation only: its real response gate still fails and no research model or prepared delivery is adopted.

The runtime v2 adapter uses explicit voicing rather than `pitch != 0`, because 0 cents is legitimate A4. Old training caches and their reported results remain frozen with the legacy adapter; the runtime fix does not retrospectively validate their coverage.

All original files are retained. Verified candidates are separate comparison files; downstream receives the unchanged accepted WAV. A numerical verification is not musical approval.'''),
    code('''example=COMPONENT/'runs/demo/bpdr_explainer';result=read(example/'example_result.json')
display(result['realized']);print('Controlled chain:',result['outcome'],'Original:',result['original_chain']['outcome'])
for name,label in [('example_original.wav','Constructed reference'),('example_detuned.wav','Locally detuned'),('example_bpdr_corrected.wav','Numerically verified candidate')]:
 print(label);display(Audio(filename=str(example/name)))
display(pd.DataFrame(M['runtime_checks']))
display(M['selected_example']);display(M['selected_preview'])
for row in M['runtime_checks'][:2]:
 folder=Path(row['path']);print(row['label'],row['outcome']);display(Audio(filename=str(folder/'accepted.wav')))
 if (folder/'experimental_candidate.wav').exists():display(Audio(filename=str(folder/'experimental_candidate.wav')))
row=M['runtime_checks'][0];folder=Path(row['path']);z=np.load(folder/'model_predictions.npz')
fig,ax=plt.subplots(figsize=(11,3.5))
for i,label in enumerate(['Nominal','Jittered','Removed edges']):ax.plot(z['times'],z['predictions'][i],label=label)
ax.set(xlabel='Time (seconds)',ylabel='Requested correction (cents)',title='Audio-only inference: no private reference supplied');ax.legend();plt.tight_layout();plt.show()'''),
    md('''## 7. Reliability, timing and reproducibility

The technical path distinguishes acceptance, re-record requests and processing failures; failed inputs have no accepted audio. The controller withholds excessive/uncertain edits and rolls back failed candidates. Rendering preserves sample count and copies samples outside approved support. These are tested engineering properties, not evidence of musical quality.

The following timing sample repeats one constructed 20-second phrase five times on the recorded Windows device, with technical analysis and model inference, preview rendering off. The p95 is a descriptive sample percentile; it does not establish concurrent-service performance or another OS.'''),
    code('''display(M['latency']['hardware']);display(pd.DataFrame(M['latency']['rows']))
display(M['training_environment']);display(pd.DataFrame(M['contract_checks']['rows']))
print('Median seconds:',M['latency']['median_seconds'],'Sample p95:',M['latency']['p95_seconds'],'30-second target:',M['latency']['passes'])
fig,ax=plt.subplots(figsize=(7,3));ax.bar(range(1,6),[r['seconds'] for r in M['latency']['rows']]);ax.axhline(30,c='grey',ls=':');ax.set(xlabel='Warm repeat after three runtime checks',ylabel='Elapsed seconds',title='Constructed 20-second phrase, inference preview');plt.tight_layout();plt.show()
print('Shared pipeline Python files verified unchanged:',M['shared_pipeline_unchanged'])
print('Regression receipt:',COMPONENT/'runs/presentation_full_regression.log')
print('Local interface:',COMPONENT/'Run-Presentation.ps1')'''),
    md('''## 8. What can be claimed to the panel

**Implemented, with executed checks described above:** the proposal-shaped Transformer and response losses; three matched neural conditions; real and synthetic data preparation; all three synthetic seeds; real transfer and real+synthetic adaptation; a nonlongest-note challenge; local upload/microphone/playback; audio-only inference; conservative permission/render/audit/rollback; versioned unchanged hand-off; and the displayed saved results. Browser smoke tests cover labelled cases; native file-picker automation was unavailable and physical microphone capture remains a user check.

**Not established:** reliable real-singer repair, a consistent BPDR advantage over augmentation-only/frame-only, calibrated musical acceptance or harmful-edit rates, blinded listening, participant recruitment, downstream melody-extraction improvement, or a completed independent final test. Review remains deferred. No synthetic voice is counted as a person, and no engineering candidate is counted as an approved research example.

**Gates:** E open for real-voice recovery and the methodological comparison; F open for validated real delivered repair; D/J4 deferred. The presentation demonstration is usable; the main proposal study is not complete.

The next scientific work is stronger independently reviewed reference coverage, more stable real-voice learning, preservation testing at matched coverage, and the consented participant / blinded / downstream studies. Keep the common pipeline unchanged. See `docs/Presentation_Readiness_2026-10-17.md` for proposal traceability and runnable commands.''')]


def report_text(card,m,evidence):
    rows=[]
    for c,r in evidence['adapt']['results'].items():
        for ds,s in r['by_dataset'].items():rows.append(f"| {c} | {ds} | {s['correction_mae']:.2f} | {s['median_response_gain']:.4f} | {s['unshifted_alteration']:.2f} | {r['engineering_gate_passes']} |")
    return '''# BPDR presentation release — 17 October 2026

The local presentation prototype is implemented. The selected preview Transformer has synthetic pretraining plus adaptation on real Vocadito/Jamendo recordings. It retains controlled synthetic correction; reliable real-singer repair and the proposal's extra BPDR benefit remain unestablished. This release is not completion of the main participant/listening/final-test research study.

## Start and demonstrate

From `experiments/bpdr_input`, run `./Run-Presentation.ps1`, then open http://127.0.0.1:8501. The interface supports WAV/FLAC upload, microphone capture, labelled examples, playback, prediction plots, reasoned feedback, diagnostics download and unchanged accepted-WAV export. Microphone permission is controlled by the browser. Files stay in component-local `runs/`; no automatic research reuse occurs.

For a direct model check: `.venv/Scripts/python.exe -m bpdr_input.model_runtime <recording.wav> --render-preview` with `PYTHONPATH=src`. The preview never replaces the downstream WAV. The module accepts an explicit `--checkpoint`; never load untrusted PyTorch checkpoint files. Windows execution is verified here; macOS execution has not been tested.

For dependencies: `.tools/uv.exe sync --locked --extra rendering --group notebooks --group presentation` (or use an installed `uv`). The rendering extra installs Parselmouth for candidate rendering. Source notebooks are `notebooks/01_input_validation_and_analysis.ipynb` and `02_data_review_and_paired_experiments.ipynb`. Their latest executed copies/HTML are selected by `runs/notebooks/latest.json`. Notebook execution reads saved experiments and runs the technical walkthrough; it does not retrain.

Presentation route: notebook 1 for integrity/status handling; notebook 2 sections 1–5 for proposal, datasets, method, matched results and real-data attempts; section 6 for before/after audio and annotation-free runtime; sections 7–8 for verification, gates and limitations. The app's Recorded experiments tab contains the same saved evidence.

## What changed

- All original synthetic checkpoints, including the failed seed 340, are retained with hashes. The presentation preview is the mixed-data seed-339 BPDR; its two corresponding controls are also exported.
- All nine frozen model/seed combinations were tested on 304 real-audio development variants (19 Vocadito, 285 Jamendo). None passed the declared response/preservation gate on both datasets. Final test untouched.
- One matched adaptation experiment used 995 real training variants (77 Vocadito, 918 Jamendo) mixed with 384 synthetic variants. Each model received 300 additional updates, the same eight-record schedule, seed 339, frozen normalization and learning rate 0.0001. BPDR retains synthetic error about 4.47 cents and response gain about 1.03, while real originals change less than 0.25 cents on average. Its fixed audio-only correction and original-identity checks pass, so it is explicitly selected for the experimental presentation preview. Its real correction-response gate still fails; this is not research adoption.
- A 24-phrase challenge across eight new generated voice settings shifts a nonlongest note. The frozen BPDR still responds (median gain about 0.82), but its error is 12.86 cents versus 9.12 for frame-only and 12.13 for augmentation-only; no BPDR advantage is established.
- A separate runtime now derives its boundary views from the input audio alone. With the selected mixed-data checkpoint, recorded synthetic detuning passed numerical verification, its original produced identity, and a real Vocadito original was left unchanged. Downstream audio stayed unchanged in all cases.
- Runtime pitch availability now follows explicit voicing, so legitimate A4 (0 cents) is retained. Historical training caches and results remain frozen with their legacy adapter; coverage of that pitch still needs a versioned training-cache correction in future work.
- Group-bootstrap paired intervals, all failures, raw-rule/zero references and five-repeat 20-second timing results are recorded. These do not substitute for musical review.

## Matched real+synthetic adaptation — raw requested corrections

| Model | Dataset | Correction MAE (cents) | Median response gain (ideal 1) | Original alteration (cents) | Both-dataset gate |
| --- | --- | ---: | ---: | ---: | --- |
'''+ '\n'.join(rows)+'''

The gate requires response gain at least 0.25, original alteration at most 1 cent and positive correction gain on each dataset. Lower training loss or unchanged originals alone does not demonstrate repair. These development measurements are engineering diagnostics; source variants are correlated. Jamendo groups are artist proxies rather than verified singer identities.

## Proposal traceability

| Proposal requirement | Current evidence / limitation |
| --- | --- |
| Sections 1.3 / 2: boundary-invariant local correction advantage | Mechanism implemented; controlled correction works. Additional BPDR advantage and real-singer generalisation not established. |
| Section 3.2: real development data and grouped partitions | Vocadito/Jamendo present in preparation, evaluation and adaptation. Source groups checked; no final-test access. Experienced suitability approvals remain pending. |
| Section 3.2: four views and intervention re-analysis | Existing audited real pairs; synthetic audio generated/rendered and re-extracted through the component path. Fine-pitch correction fixes quantization. |
| Section 3.3: five conditions | Three matched Transformers plus original/zero and conservative rule requested-curve references. Delivered musical results at matched coverage remain pending. |
| Sections 3.4–3.5: correction, preservation, boundary response, paired uncertainty | Saved metrics and group-bootstrap intervals; few groups and seed instability explicitly reported. No independent musical harmful-edit rate. |
| Sections 3.4–3.5: independent listening, participants, extraction F1 | Not completed. Review deferred; ethics/consent precede recruitment. No invented approvals, listener results or downstream gains. |
| Section 3.6: compact Transformer and loss | Four layers / 128 hidden / four heads / 512 FF / 0.1 dropout, 71/70 features. Squared anchor and fine YIN input are documented experimental changes; PESTO confidence remains frozen. |
| FR1 / FR5 / FR7: capture/upload, comparison, plots, understandable feedback | Local Streamlit interface, recorded audio-only checks, diagnostics and candidates. Browser smoke covers labelled cases; native file-picker automation was unavailable and microphone hardware capture needs the user's browser permission/check. |
| FR2–FR4: retain original, integrity checks, timing/rest preservation, bounded edits | Existing technical pipeline and controller; failed/uncertain candidate withheld or rolled back. No note-name/key input required. |
| FR6 / integration: accepted-only versioned hand-off | Contract checks and JSON/WAV export; prepared delivery remains off by owner decision. Live team integration not verified. |
| Conditional noise attenuation | Not enabled: no validated noise-only reference/preservation policy. No denoising claim. |
| Table 5: latency / reliability / reproducibility | Recorded five-repeat timing on one constructed 20-second phrase, regression tests, hashes, run IDs, frozen checkpoints and fresh notebooks. Mac, concurrency and full human usability remain unverified. |

## Gates and next research work

E remains open for real-voice generalisation and the primary methodological comparison. F has working numerical engineering candidates but no validated real musical repair. D/J4 remains deferred because experienced reviewers are unavailable. Research-ready examples remain zero. No model is adopted for automatic prepared-audio hand-off.

Preserve this reproducible demonstration. Future research needs steadier audited reference phrases, a declared seed-stability/real-learning experiment, independent suitability review when available, ethics-approved participant recordings, preservation and acceptance calibration, blinded listening, a fixed downstream extractor comparison and the reserved final test. A null result on BPDR's extra losses is a valid finding; do not change the hypothesis or suppress failed seeds to claim success.

## Evidence and recovery

The release's `study_manifest.json` pins completed studies and their hashes; `model_card.json` records model scope and weights. `release_manifest.json` and `presentation_evidence.zip` are written only after successful fresh-kernel notebook execution. Old notebook sources/plans are retained in `prior_maintained_sources`; old runs remain untouched. The archive is an evidence/model backup, not a complete portable dataset/environment distribution. Full live reproduction uses the component-local datasets, runs and installed dependencies. The shared extraction pipeline is unchanged.

Implementation, debugging, experiment tooling and explanatory documentation used AI assistance. Reported measurements come from saved local execution. The research owner must review the implementation and defend the methodology and claims; no participant responses, approvals or musical judgements were generated as evidence.
'''


def seal():
    out=Path(read(COMPONENT/'runs/presentation_release/prepared.json')['path'])
    manifest=read(out/'study_manifest.json')
    for p,h in manifest['source_hashes'].items():
        if sha(p)!=h:raise ValueError('Study evidence changed')
    nb=Path(read(COMPONENT/'runs/notebooks/latest.json')['path'])
    receipt=read(nb/'verification.json');provenance=read(nb/'provenance.json')
    if provenance['evidence_index_sha256']!=sha(COMPONENT/'notebooks/evidence_index.json'):
        raise ValueError('Notebook index differs')
    for row in receipt:
        if row['errors'] or row['source_sha256']!=sha(COMPONENT/'notebooks'/row['notebook']):
            raise ValueError('Notebook source not freshly executed')
    for folder in ['notebooks','source','evidence','audio_only_example','.streamlit']:(out/folder).mkdir()
    shutil.copy2(COMPONENT/'.streamlit/config.toml',out/'.streamlit/config.toml')
    for p in nb.iterdir():
        if p.suffix in ['.html','.ipynb','.json']:shutil.copy2(p,out/'notebooks'/p.name)
    for p in (COMPONENT/'src/bpdr_input').rglob('*.py'):
        target=out/'source'/p.relative_to(COMPONENT/'src/bpdr_input')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    shutil.copy2(SOURCE_ROOT/'_module_map.json',out/'source/_module_map.json')
    for name in ['transfer','adapt','probe','statistics','baselines']:
        folder=Path(manifest['studies'][name]);shutil.copy2(folder/'evidence.json',out/'evidence'/(name+'.json'))
        shutil.copy2(folder/'declaration.json',out/'evidence'/(name+'_declaration.json'))
    for name in ['pyproject.toml','uv.lock','Run-System.ps1']:shutil.copy2(COMPONENT/name,out/name)
    for name in ['presentation_full_regression.log','presentation_ui_test_verified.log','presentation_runtime_checks_mixed.json','presentation_contract_checks_mixed.json','presentation_training_environment.json','presentation_preview_selection.json','presentation_selected_example_measurement.json']:
        shutil.copy2(COMPONENT/'runs'/name,out/'evidence'/name)
    example=Path(manifest['runtime_checks'][0]['path'])
    for p in example.iterdir():
        if p.suffix in ['.wav','.json','.png']:shutil.copy2(p,out/'audio_only_example'/p.name)
    packages={}
    for name in ['torch','numpy','librosa','pesto-pitch','praat-parselmouth','streamlit','nbclient','nbconvert']:
        packages[name]=importlib.metadata.version(name)
    files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'release_manifest.json',dict(version='bpdr-presentation-release-1',sealed_utc=datetime.now(timezone.utc).isoformat(),
        files=files,packages=packages,notebook_verification=receipt,prepared_handoff_enabled=False,
        research_study_complete=False,review_status='deferred, not approved'))
    archive=out/'presentation_evidence.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p!=archive:z.write(p,p.relative_to(out))
    write(COMPONENT/'runs/presentation_release/current.json',dict(path=str(out),archive_sha256=sha(archive)))
    print('SEALED',out,flush=True)
    return out


if __name__=='__main__':
    {'prepare':prepare,'seal':seal}[sys.argv[1]]()
