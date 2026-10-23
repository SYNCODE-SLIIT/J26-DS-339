"""Frozen checkpoint-to-controller wiring on one previously used training crop; no edits or fitting."""
from bpdr_input import COMPONENT_ROOT, module_source
from dataclasses import asdict
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from bpdr_input.audio.analysis import extract
from bpdr_input.preparation.sources.data_builder import write_json
from bpdr_input.modeling.examples import Normalizer
from bpdr_input.modeling.model import CorrectionEncoder, ModelConfig
from bpdr_input.audio.pipeline import checksum
from bpdr_input.repair.prediction_adapter import ProposalPolicy, VIEWS, infer_views, propose, validate_features


def synthetic_cases():
    audio=np.full(8*22050,.1,dtype=np.float32)
    times=np.arange(len(audio)//441+1)*.02;n=len(times)
    frame=np.zeros((n,70),dtype=np.float32);frame[:,65]=.9;frame[:,66]=1.
    features=dict(times_seconds=times,confidence=np.full(n,.9),voiced_mask=np.ones(n,dtype=bool),
        provisional_boundary_score=np.zeros(n),frame_features_70=frame,
        boundary_features_71=np.column_stack([frame,np.zeros(n)]).astype(np.float32))
    base=np.where((times>=1.)&(times<1.9),20.,0.)
    variable=np.where(base!=0,10.+10.*np.clip((times-1.)/.9,0,1),0.)
    opposite=base.copy();opposite[(times>=1.9)&(times<2.8)]=-20.
    short=base.copy();short[(times>=1.45)&(times<1.9)]=-20.
    cases=[]
    for name,curve,mode,uncertainty,mask,expected in [
        ('bounded_variable',variable,'synthetic_fixture',np.ones(n),np.zeros(n,dtype=bool),'FIXTURE_PROPOSAL_PERMITTED'),
        ('opposite_supported',opposite,'synthetic_fixture',np.ones(n),np.zeros(n,dtype=bool),'FIXTURE_PROPOSAL_PERMITTED'),
        ('opposite_short',short,'synthetic_fixture',np.ones(n),np.zeros(n,dtype=bool),'WITHHELD'),
        ('missing_uncertainty',base,'runtime_preview',None,None,'WITHHELD'),
        ('supplied_but_uncalibrated',base,'runtime_preview',np.ones(n),np.zeros(n,dtype=bool),'WITHHELD'),
        ('subthreshold',np.full(n,.2),'runtime_preview',None,None,'NO_EDIT_PROPOSED')]:
        result,_=propose(audio,22050,features,{v:curve.copy() for v in VIEWS},uncertainty,mask,mode)
        if result['outcome']!=expected: raise ValueError(f'Unexpected constructed adapter outcome: {name}')
        cases.append(dict(name=name,expected=expected,result=result))
    return cases


def run():
    import torch
    component=COMPONENT_ROOT
    selected=dict(phrase_id='vocadito_37_p00',
        data_run='runs/phase2/15fb678d1f44406ab4da3fddd7c2f809',
        preparation='runs/coverage/1993b73598b94feeb4de15d3588f65e9/probes/vocadito_37_p00/preparation/84f3c15b936741c69a45524f334a3387',
        checkpoint_run='runs/engineering/9cd3c1a81e274e728a08e5e278bc8790')
    tracked={}
    def track(path,expected=None):
        path=Path(path).resolve()
        if not path.is_relative_to(component): raise ValueError('Preview evidence must remain in BPDR')
        digest=checksum(path)
        if expected is not None and digest!=expected: raise ValueError(f'Changed input: {path}')
        tracked[str(path)]=digest;return path
    data_run=component/selected['data_run']
    manifest=json.loads(track(data_run/'manifest.json').read_text())
    track(data_run/'reviews.csv')
    phrase=next(p for p in manifest['phrases'] if p['phrase_id']==selected['phrase_id'])
    if phrase['split']!='train': raise ValueError('Only the declared training phrase is allowed')
    preparation=component/selected['preparation']
    diagnostic=json.loads(track(preparation/'diagnostics.json').read_text())
    if diagnostic['source']['sha256']!=phrase['sha256']: raise ValueError('Preparation lineage mismatch')
    source=track(diagnostic['result']['accepted_audio']['path'],diagnostic['result']['accepted_audio']['checksum'])
    checkpoint_run=component/selected['checkpoint_run']
    record=json.loads(track(checkpoint_run/'run_record.json').read_text())
    if record['mode']!='engineering' or record['data_status']!='exploratory_not_research':
        raise ValueError('Expected the declared historical engineering smoke checkpoint')
    checkpoint=track(checkpoint_run/'checkpoint.pt',record['checkpoint_sha256'])
    track(Path(record['pairs_run'])/'summary.json',record['pairs_summary_sha256'])
    # Inference architecture/normalizer must match; a changed historical training command is disclosed separately.
    for name in ('model','examples'):
        track(module_source(name+'.py'),record['source_code_sha256'][name])
    root=component/'runs/prediction_preview'/uuid.uuid4().hex;root.mkdir(parents=True)
    snapshots=root/'source_code_at_run';snapshots.mkdir()
    for name in ('prediction_preview','prediction_adapter','repair_controller','analysis','model','examples'):
        code=track(module_source(name+'.py'));(snapshots/code.name).write_bytes(code.read_bytes())
    evidence=dict(run_status='running',scope='engineering_prediction_preview',data_status='exploratory_not_research',
        selected=selected,source_id=phrase['source_id'],singer_id=phrase['singer_id'],split=phrase['split'],
        input_files=tracked,proposal_policy=asdict(ProposalPolicy()),device='cpu',torch_version=torch.__version__,
        cpu_threads=torch.get_num_threads(),training_updates_performed=0,audio_edits_performed=0,
        accepted_prepared_packages=0,training_ready_examples=0,production_eligible=False,
        checkpoint_training_revision_matches=all(checksum(module_source(n+'.py'))==h for n,h in record['source_code_sha256'].items()),
        inference_model_and_normalizer_revisions_match=True,limitations=[
            'One previously inspected training phrase from S29 and a 30-update engineering checkpoint; not independent evaluation',
            'Smoke checkpoint is not an accurate repair model; stored training fit had essentially zero edit coverage',
            'Training used annotation-derived boundaries; runtime heuristic-column distribution is unvalidated',
            'Shifted acoustic boundary views are sensitivity hypotheses, not calibrated natural-boundary alternatives',
            'No calibrated uncertainty/expressivity provider or production permission policy; runtime edits remain withheld',
            'Constructed adapter cases supply predictions/uncertainty masks and are not model or audio-quality measurements'])
    write_json(root/'evidence.json',evidence)
    print('PREDICTION_PREVIEW_RUN:',root,flush=True)
    try:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        # Legacy checkpoint stores tensors/config/condition only. Its role comes
        # from the recorded run bound to the verified checkpoint SHA, not a guess.
        if saved.get('data_status',record['data_status'])!='exploratory_not_research' or saved['condition']!=record['condition']:
            raise ValueError('Checkpoint role mismatch')
        if any(saved['config'][key]!=record['model'][key] for key in saved['config']):
            raise ValueError('Checkpoint configuration differs from its bound run record')
        evidence['checkpoint_role_provenance']='Verified run_record.json bound to checkpoint SHA; legacy tensor file has no embedded data_status'
        model=CorrectionEncoder(ModelConfig(**saved['config']));model.load_state_dict(saved['state']);model.eval()
        normalizer=Normalizer(**saved['normalizer'])
        audio,sr=sf.read(source,dtype='float32')
        features,analyzer=extract(audio,sr,component/'checkpoints')
        validate_features(features,audio,sr)
        prediction=infer_views(model,normalizer,features)
        repeated=infer_views(model,normalizer,features)
        if not all(np.array_equal(prediction[v],repeated[v]) for v in VIEWS): raise ValueError('Deterministic eval repeat differs')
        report,arrays=propose(audio,sr,features,prediction)
        if report['edit_permission'] or report['production_eligible'] or report['usable_intervals_seconds']:
            raise ValueError('Runtime preview incorrectly exposed edit permission')
        arrays['provisional_boundary_score']=features['provisional_boundary_score']
        archive=root/'predictions_and_masks.npz';np.savez_compressed(archive,**arrays)
        evidence.update(analyzer=analyzer,preview=report,synthetic_cases=synthetic_cases(),
            deterministic_eval_repeat_equal=True,prediction_frames=len(arrays['times_seconds']),
            prediction_artifact=dict(path=str(archive),sha256=checksum(archive)))
        for path,digest in tracked.items():
            if checksum(path)!=digest: raise ValueError('Source/checkpoint/review/code changed during preview')
        evidence['run_status']='complete';write_json(root/'evidence.json',evidence)
        print(json.dumps(dict(outcome=report['outcome'],max_abs_cents=report['raw_max_abs_cents'],
            requested_frames=report['requested_frames'],blockers=report['blockers'],synthetic_cases=len(evidence['synthetic_cases'])),indent=2),flush=True)
    except BaseException as exc:
        evidence.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)))
        write_json(root/'evidence.json',evidence);raise
    return root


if __name__=='__main__': run()
