"""Guarded project adapter for a separately verified held-voice Transformer."""
from pathlib import Path
import json
import numpy as np
import soundfile as sf

from bpdr_input import COMPONENT_ROOT
from bpdr_input.audio.pipeline import process,checksum,validate_package
from bpdr_input.modeling.held_voice_transformer import load,predict_raw
from bpdr_input.repair.held_voice_policy import HeldPolicy,proposal

WORKSPACE=COMPONENT_ROOT/'BPDR_Workspace'
CARD=WORKSPACE/'results/transformer_usability/active_model.json'
METHOD_NAME='Transformer held-voice repair (scoped preview)'


def load_model(card_path=None):
    card=json.loads(Path(card_path or CARD).read_text(encoding='utf-8'))
    if not card.get('eligible_for_scoped_preview') or not card.get('requires_user_review'):
        raise ValueError('Held-voice Transformer has not passed its scoped output gates')
    if card.get('research_validation_passed') is not False:
        raise ValueError('Scoped engineering evidence cannot assert research validation')
    path=Path(card['checkpoint']).resolve()
    if not path.is_relative_to(WORKSPACE.resolve()) or checksum(path)!=card['checkpoint_sha256']:
        raise ValueError('Transformer checkpoint path or hash mismatch')
    for stage in ('development','final'):
        receipt=card['verification'][stage];evidence=Path(receipt['path'])
        if checksum(evidence)!=receipt['sha256']:
            raise ValueError('Transformer verification evidence changed')
        measured=json.loads(evidence.read_text(encoding='utf-8'))
        if not measured['gates']['passed'] or measured['checkpoint_sha256']!=card['checkpoint_sha256'] or measured['policy']!=card['policy']:
            raise ValueError('Transformer verification does not match this model and policy')
    for file,expected in card['runtime_source_hashes'].items():
        if checksum(file)!=expected:raise ValueError('Verified Transformer runtime source changed')
    if card.get('model_family')=='step_context':
        from bpdr_input.modeling.step_transformer import load as load_step
        model,payload=load_step(path)
    else:model,payload=load(path)
    return model,payload,card


def available():
    try:load_model();return True
    except (OSError,ValueError,KeyError,ImportError):return False


def run(source,output_root=None):
    model,payload,card=load_model()
    return run_candidate(source,model,payload,card,output_root)


def run_candidate(source,model,payload,card,output_root=None):
    import torch
    torch.set_num_threads(2)
    from bpdr_input.repair.model_runtime import runtime_views
    from bpdr_input.repair.rule_correction import fine_pitch
    from bpdr_input.experiments.learning.context_improvement import rendered_output
    result,directory=process(Path(source),output_root or WORKSPACE/'results/demo_sessions',COMPONENT_ROOT/'checkpoints')
    report=dict(version='held-voice-transformer-runtime-1',source=str(source),method=METHOD_NAME,handoff=result.to_dict(),prepared_handoff_enabled=False,
        model_path=card['checkpoint'],model_sha256=card['checkpoint_sha256'],model_scope=card['scope'],model_metrics=card.get('metrics',{}),
        research_gates_passed=False,model_validation_status=card.get('validation_status','scoped_engineering_preview_independent_listening_not_assessed'),
        diagnostic_only=not card.get('eligible_for_scoped_preview',False),outcome='NOT_RUN_TECHNICAL_STOP')
    if result.accepted_audio:
        try:
            audio,sr=sf.read(result.accepted_audio.path,dtype='float32')
            with np.load(directory/'features.npz') as z:features={k:z[k] for k in z.files}
            fine=fine_pitch(audio,sr,features['times_seconds'])
            views,reliable,protected,metadata=runtime_views(features,fine)
            infer=predict_raw
            if card.get('model_family')=='step_context':
                from bpdr_input.modeling.step_transformer import predict_raw as infer
            predictions=np.stack([infer(model,raw)[0] for raw in views])
            detail=proposal(audio,features['times_seconds'],views,predictions,HeldPolicy(**card['policy']))
            analysis=dict(features=features,views=views,reliable=reliable,protected=detail['protected'],fine=fine)
            output,transaction=rendered_output(dict(audio_A=audio,A=analysis),'A',detail['curve'],detail['plan'],directory)
            report.update(outcome=transaction['outcome'],permission=detail['plan'],transaction=transaction,
                proposed_intervals_seconds=detail['plan']['intervals_seconds'],input_metadata=metadata,candidates=detail['candidates'])
            if transaction['outcome']=='VERIFIED_FIXTURE_CANDIDATE':
                delivered=transaction['delivered_wav_audit']
                report['candidate']=dict(path=str(directory/'A_candidate.wav'),sha256=delivered['sha256'])
                transaction['post_render_audit']=delivered['acoustic_audit']
            np.savez_compressed(directory/'model_predictions.npz',times=features['times_seconds'],predictions=predictions,
                proposed_correction=detail['curve'],reliable=reliable,protected=detail['protected'])
            validate_package(result)
        except Exception as exc:
            report.pop('candidate',None);report.update(outcome='TRANSFORMER_PREVIEW_FAILED_UNCHANGED',reason=str(exc),error_type=type(exc).__name__)
    (directory/'model_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return directory,report
