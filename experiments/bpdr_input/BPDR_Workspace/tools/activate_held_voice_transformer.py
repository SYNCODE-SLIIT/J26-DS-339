"""Expose a scoped BPDR preview only after locked real-WAV gates pass."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.repair.held_voice_runtime import CARD,load_model


def activate(root):
    root=Path(root).resolve();policy=ex.read(root/'locked_policy.json')['selected']['policy']
    checkpoint=root/'final/seed_339/checkpoint.pt';digest=ex.sha(checkpoint)
    verifications={};final=None
    for stage,folder in [('development',root/'development'),('final',root/'final_evaluation')]:
        evidence=ex.read(folder/'evidence.json')
        if not evidence['gates']['passed'] or evidence['checkpoint_sha256']!=digest or evidence['policy']!=policy:
            raise ValueError('Refusing activation: '+stage+' delivered-audio gates failed or model/policy differs')
        if ex.sha(folder/'accuracy.json')!=evidence['accuracy_sha256'] or ex.sha(folder/'original_controls.json')!=evidence['controls_sha256']:
            raise ValueError('Evaluation measurements changed')
        verifications[stage]=dict(path=str(folder/'evidence.json'),sha256=ex.sha(folder/'evidence.json'))
        if stage=='final':final=evidence
    if len(final['gates']['by_singer'])!=4:raise ValueError('Four reserved final singers are required')
    for path,expected in ex.read(root/'declaration.json')['protected_files'].items():
        assert ex.sha(path)==expected,'Protected prior model/notebook changed'
    sources=[ex.WORKSPACE/'source/bpdr_input'/name for name in [
        'modeling/held_voice_transformer.py','repair/held_voice_policy.py','repair/held_voice_runtime.py',
        'repair/framewise_renderer.py','repair/framewise_audit.py','repair/repair_controller.py',
        'repair/learned_runtime.py','repair/model_runtime.py','repair/rule_correction.py',
        'experiments/learning/context_improvement.py','audio/analysis.py']]
    declaration=ex.read(root/'declaration.json')
    if declaration.get('model_family')=='step_context':
        sources.extend([ex.WORKSPACE/'source/bpdr_input/modeling/step_context.py',ex.WORKSPACE/'source/bpdr_input/modeling/step_transformer.py'])
    result=ex.read(root/'final_evaluation/accuracy.json')
    dev=ex.read(root/'development/accuracy.json');pairs=ex.read(root/'development/pairs.json')
    cases=[];seen=set()
    for index,(row,pair) in enumerate(zip(dev['rows'],pairs)):
        if not row.get('delivered_verified_improvement'):continue
        sign=1 if row['cents']>0 else -1
        if sign in seen:continue
        seen.add(sign)
        cases.append(dict(id='held_transformer',label=f'Transformer: controlled {row["cents"]:+g}-cent held-vocal deviation',
            input=pair['path_C'],reference=pair['path_A'],score_mask_path=pair['target_path'],
            source_kind='Real VocalSet development voice with an audited controlled long deviation. First verified example in manifest order for this sign; not a natural-error accuracy claim.'))
    card=dict(activated_at=datetime.now(timezone.utc).isoformat(),checkpoint=str(checkpoint),checkpoint_sha256=digest,policy=policy,
        model_family=declaration.get('model_family','held_voice'),
        eligible_for_scoped_preview=True,requires_user_review=True,research_validation_passed=False,
        scope=declaration.get('practical_scope','Controlled small local deviations in stable held vowels; not general singing correctness or natural-intention validation'),
        verification=verifications,runtime_source_hashes={str(p):ex.sha(p) for p in sources},
        metrics=dict(model_type='StepContextTransformer' if declaration.get('model_family')=='step_context' else 'HeldVoiceTransformer',dataset_role='Four reserved final singers attempted; only reference-scoreable singers assessed for repair benefit',summary=result['summary'],practical_gates=final['gates']),
        run_root=str(root),demo_cases=cases)
    candidate=root/'scoped_preview_card.json';ex.write(candidate,card)
    load_model(candidate)
    ex.write(CARD,card)
    print('SCOPED_TRANSFORMER_ACTIVATED',CARD,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True)
    activate(parser.parse_args().run_root)
