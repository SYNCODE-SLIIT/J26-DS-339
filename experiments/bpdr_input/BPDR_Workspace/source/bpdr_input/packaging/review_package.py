"""Seal a reviewable engineering handover; never claim research completion.

This is an evidence/code/model snapshot for the current workspace, not a full
portable training-data or runtime distribution. Earlier releases stay intact.
"""
from pathlib import Path
import json
import shutil
import uuid
import zipfile
from bpdr_input.experiments.evaluation.transfer_study import COMPONENT, read, write, sha


def main():
    root=COMPONENT/'runs/completion_release';out=root/uuid.uuid4().hex;out.mkdir(parents=True)
    files={}
    def add(path,name=None):
        p=Path(path).resolve()
        if not p.is_relative_to(COMPONENT):raise ValueError('Only component-owned artifacts may be packaged')
        if not p.is_file():raise FileNotFoundError(p)
        key=name or str(p.relative_to(COMPONENT)).replace('\\','/')
        if key in files and files[key]!=p:raise ValueError('Duplicate archive target')
        files[key]=p
    def tops(folder,extensions=('.json','.npy','.npz','.py')):
        for p in Path(folder).iterdir():
            if p.is_file() and p.suffix in extensions:add(p)
    for p in (COMPONENT/'src/bpdr_input').rglob('*.py'):add(p)
    for p in (COMPONENT/'tests').glob('test*.py'):add(p)
    for name in ['README.md','pyproject.toml','uv.lock','Run-Presentation.ps1','Run-Notebooks.ps1','.streamlit/config.toml',
                 'docs/Presentation_Readiness_2026-10-17.md','docs/Reference_Review_Guide.md','docs/PP1_Master_Plan.md',
                 'docs/PP1_Phase_Execution_Plan.md','docs/PP1_Evidence_Tracker.md','docs/IT23141506_J26_DS_339.pdf',
                 'extraction/completion_handoff_check.py','extraction/check_handoff.py']:
        add(COMPONENT/name)
    for p in (COMPONENT/'notebooks').glob('*.ipynb'):add(p)
    add(COMPONENT/'notebooks/evidence_index.json')
    nb=Path(read(COMPONENT/'runs/notebooks/latest.json')['path']);verification=read(nb/'verification.json')
    if len(verification)!=2 or any(v['errors'] for v in verification):raise ValueError('Need two successful notebooks')
    for v in verification:
        if sha(COMPONENT/'notebooks'/v['notebook'])!=v['source_sha256']:raise ValueError('Notebook source changed after execution')
    for p in nb.iterdir():
        if p.is_file():add(p)
    index=read(COMPONENT/'notebooks/evidence_index.json')
    for value in index['evidence'].values():
        p=COMPONENT/value
        if p.is_file() and p.suffix=='.json':add(p)
    base=Path(read(COMPONENT/'runs/real_learning_v2/current.json')['path']);tops(base);tops(Path(read(base/'pilot_current.json')['path']))
    hum=Path(read(COMPONENT/'runs/humming_experiment/current.json')['path']);tops(hum);tops(hum/'frozen_evaluation')
    hfit=Path(read(COMPONENT/'runs/humming_learning/current.json')['path']);tops(hfit)
    for sub in ['1afb0db88bd54883aa492bd6ecad9d56','b30e0327c70b4dbf90dd917219dcd8d3']:tops(COMPONENT/'runs/humming_experiment'/sub)
    for sub in ['9c5c481a93194d9ab56c91f0a8318807','cafba265f7064be48af281d508128efb','fea78af792a246c89c5cdb7221d94b5b']:tops(COMPONENT/'runs/environmental_noise_study'/sub)
    for name in ['completion_model_decision.json','humming_noise_live_check.json','completion_app_verification.json',
                 'completion_regression.log','completion_notebooks.log','completion_app_check.py','completion_app_check.log']:
        add(COMPONENT/'runs'/name)
    selected=Path(read(COMPONENT/'runs/presentation_release/current.json')['path']);card=read(selected/'model_card.json')
    cp=selected/card['checkpoint']
    if sha(cp)!=card['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
    add(cp,'models/selected_presentation_bpdr.pt');add(selected/'model_card.json','models/selected_model_card.json')
    add(hfit/'bpdr.pt','models/rejected_humming_pilot.pt')
    add(Path(read(base/'pilot_current.json')['path'])/'augmentation_only.pt','models/rejected_real_input_pilot.pt')
    tops(selected)
    live=read(COMPONENT/'runs/humming_noise_live_check.json');case=Path(live['source_study_case']['directory'])
    for name in ['baseline.wav','noisy.wav']:add(case/name,'demonstration/'+name)
    add(Path(live['report']['noise_candidate']['path']),'demonstration/noise_cleaned_preview.wav')
    example=COMPONENT/'runs/demo/bpdr_explainer'
    for name in ['example_original.wav','example_detuned.wav','example_bpdr_corrected.wav','example_result.json']:add(example/name,'demonstration/'+name)
    for name in ['README.md','intake_declaration.json','intake_receipt.json']:add(COMPONENT/'data/humtrans'/name)
    for name in ['LICENSE','README.md','receipt.json','esc50.csv']:add(COMPONENT/'data/environmental_noise'/name)
    add(COMPONENT/'data/environmental_noise/validation/receipt.json')
    game=Path(read(COMPONENT/'runs/completion_handoff_check/current.json')['path']);tops(game)
    for p in (game/'notes').rglob('*.csv'):add(p)
    shared=read(COMPONENT/'runs/presentation_study/6da1381349154863938d9358f91b4f84/declaration.json')['shared_pipeline']
    if not all(sha(p)==h for p,h in shared.items()):raise ValueError('Shared pipeline changed')
    model_decision=read(COMPONENT/'runs/completion_model_decision.json')
    # Copy actual files, hash the copies and recheck originals before sealing.
    manifest=[]
    for name,p in sorted(files.items()):
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        digest=sha(target)
        if digest!=sha(p):raise ValueError('Copy/source mismatch')
        manifest.append(dict(path=name,sha256=digest,bytes=target.stat().st_size))
    release=dict(version='bpdr-completion-review-1',status='engineering_prototype_verified_research_incomplete',
        selected_checkpoint='models/selected_presentation_bpdr.pt',selected_sha256=sha(cp),
        diagnostic_checkpoints=['models/rejected_humming_pilot.pt','models/rejected_real_input_pilot.pt'],
        model_decision=model_decision,regression_tests=448,notebook_verification=verification,
        app_verification=read(COMPONENT/'runs/completion_app_verification.json'),shared_pipeline_files_unchanged=len(shared),
        research_ready_examples=0,prepared_handoff_enabled=False,final_test_audio_accessed=False,
        research_study_complete=False,model_adopted_for_research=False,
        limitations=['Real-singer pitch repair and BPDR advantage not established','No novice/listener study or human approvals',
                     'Recorded-noise controls are constructed mixtures','Single GAME consistency example showed no extraction gain',
                     'Evidence snapshot, not full portable datasets/checkpoints/runtime; replay in original workspace'],files=manifest)
    write(out/'release_manifest.json',release)
    (out/'START_HERE.md').write_text('''# BPDR presentation evidence package

Open the executed notebook HTML files under `runs/notebooks/` and read
`docs/Presentation_Readiness_2026-10-17.md`. The selected model is in `models/`;
the two rejected pilots are labelled explicitly. Audio comparisons are under
`demonstration/`. In the original workspace, run `Run-Presentation.ps1`.

This package preserves verified engineering work and honest null results.
It is not a completed real-user study, musical approval, full training-data
distribution or self-contained portable runtime. See the manifest and source
attributions. No failed candidate replaces the working presentation model.
''',encoding='utf-8')
    archive=out/'BPDR_presentation_completion.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p!=archive:z.write(p,p.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive CRC failure')
        for item in manifest:
            import hashlib
            if hashlib.sha256(z.read(item['path'])).hexdigest()!=item['sha256']:raise ValueError('Archive checksum failure')
    write(root/'current.json',dict(path=str(out),archive=str(archive),sha256=sha(archive),files=len(manifest),status=release['status']))
    print('COMPLETION_RELEASE',out,'files',len(manifest),'archive_MB',round(archive.stat().st_size/1024**2,2),flush=True)


if __name__=='__main__':main()
