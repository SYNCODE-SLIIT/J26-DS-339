"""A private, unchanged BPDR source snapshot confines its cache/output locations."""
import sys
import os
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
# Reuse optional plotting packages from the main project only after the BPDR
# environment's own packages, preserving its sklearn/joblib serialization versions.
shared=(ROOT/'.venv/Lib/site-packages' if os.name=='nt' else ROOT/'.venv/lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages')
if shared.is_dir():sys.path.append(str(shared))
from pp1_demo.config import APP, BPDR, BPDR_EXPORT, RUNTIME, configure_runtime
configure_runtime()
import hashlib
import json
import os
import runpy
import shutil


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare():
    # The unmodified package locates COMPONENT_ROOT through its pyproject.toml.
    # Copy it rather than changing owner globals or installing into owner folders.
    source=BPDR/'source'
    digest=hashlib.sha256()
    for p in sorted(source.rglob('*.py')):
        digest.update(str(p.relative_to(source)).encode());digest.update(p.read_bytes())
    digest.update((source/'bpdr_input/_module_map.json').read_bytes())
    digest.update((BPDR_EXPORT/'manifest.json').read_bytes())
    component=RUNTIME/'bpdr_sandbox'/digest.hexdigest()[:16]/'bpdr_input'
    workspace=component/'BPDR_Workspace'
    if not (workspace/'source').exists():
        component.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BPDR.parent/'pyproject.toml',component/'pyproject.toml')
        shutil.copytree(source,workspace/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    sys.path.insert(0,str(workspace/'source'))
    card=workspace/'results/model_recovery/notebook_training/2759404560ab4b559a633659d316bec0_inference_export/demo_model.json'
    if not card.exists():
        installer=runpy.run_path(str(BPDR_EXPORT/'install.py'))
        card=installer['install'](activate=False)
    return card


def main():
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_num_threads(2)
    source, folder, output=map(Path,sys.argv[1:])
    card_path=prepare()
    from bpdr_input.repair.learned_runtime import load_model, run_candidate
    model,payload,card=load_model(card_path=card_path)
    directory, report=run_candidate(source,model,payload,card,output_root=folder/'bpdr')
    accepted=report['handoff'].get('accepted_audio')
    chart={}
    if accepted:
        # Use exactly the owner's live feature/proposal functions for the plot,
        # including when a later optional renderer could not produce a preview.
        from bpdr_input.repair.model_runtime import runtime_views
        from bpdr_input.repair.rule_correction import fine_pitch
        from bpdr_input.modeling.recovery_model import propose
        with np.load(directory/'features.npz',allow_pickle=False) as z:
            features={k:z[k] for k in z.files}
        audio,sr=sf.read(accepted['path'],dtype='float32')
        views,_,_,_=runtime_views(features,fine_pitch(audio,sr,features['times_seconds']))
        proposal=propose(views,model,payload)
        step=max(1,len(features['times_seconds'])//500)
        def finite(values):
            return [float(x) if np.isfinite(x) else None for x in values[::step]]
        chart={'times':finite(features['times_seconds']),
               'f0_hz':finite(features['f0_hz']),
               'raw_correction_cents':finite(proposal['raw_predictions'][0]),
               'requested_correction_cents':finite(proposal['requested_curve']),
               'boundary_score':finite(features['provisional_boundary_score']),
               'voiced_fraction':float(np.mean(features['voiced_mask'])),
               'scope':proposal['scope_metadata'],'model_ran':True}
    result={'accepted_audio':accepted['path'] if accepted else None,
            'handoff_status':report['handoff']['status'],
            'message':report['handoff']['user_message'],
            'outcome':report['outcome'],'reason':report.get('reason'),
            'model':type(model['estimator']).__name__,'trees':getattr(model['estimator'],'n_estimators',None),
            'features':getattr(model['estimator'],'n_features_in_',None),'checkpoint_sha256':card['checkpoint_sha256'],
            'run_id':'2759404560ab4b559a633659d316bec0','chart':chart,
            'preview':report.get('candidate',{}).get('path'),
            'model_scope':payload['scope'],'requires_user_review':True,
            'research_gates_passed':False,'full_report':str(directory/'model_report.json'),
            'decision':report.get('permission',{}),'provenance':'live_model_inference',
            'forwarded_audio':'Accepted unchanged audio; repair preview is separate.'}
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'status':result['handoff_status'],'outcome':result['outcome'],'model_ran':chart.get('model_ran',False)}))


if __name__=='__main__':
    main()
