"""Read-only verification of frozen study inputs, lineage and encoded outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.multiscale_transformer import load
from bpdr_input.modeling.correction_accuracy import assert_independent


def verify(root):
    d=ex.read(root/'declaration.json')
    loader=load
    if d['model_family']=='joint_context':
        from bpdr_input.modeling.joint_transformer import load as loader
    checked=0
    for label in ('source_hashes','input_hashes','baseline_checkpoints','parent_hashes'):
        for p,h in d.get(label,{}).items():
            assert ex.sha(p)==h,p
            checked+=1
    for label in ('baseline_cv','baseline_evidence'):
        assert ex.sha(d[label])==d[label+'_sha256']
    for arm,path in d.get('fit_manifests',{}).items():
        assert ex.sha(path)==d['fit_manifest_hashes'][arm]
    if 'pool_receipt' in d:assert ex.sha(d['pool_receipt'])==d['pool_receipt_sha256']
    manifest=ex.read(root/'train_pairs.json')
    assert len(manifest)==200 and sum('excluded' not in p for p in manifest)==108
    fits=[]
    for folder in sorted((root/'cross_validation').glob('fold_*/*')):
        model,payload=loader(folder/'checkpoint.pt')
        receipt=ex.read(folder/'fit_receipt.json')
        fold=int(folder.parent.name.split('_')[-1])
        test=[p for p in manifest if ex.group(p) in d['folds'][fold-1]]
        assert_independent(payload['training_groups'],payload['training_sources'],test)
        assert receipt['checkpoint_sha256']==ex.sha(folder/'checkpoint.pt')
        assert receipt['reload_verified'] and payload['updates']==d['cv_updates']
        assert payload['declaration_sha256']==ex.sha(root/'declaration.json')
        if 'cv_parents' in d:
            assert payload['parent_checkpoint_sha256']==ex.sha(d['cv_parents'][str(fold)])
            assert payload['initial_equivalence_max_error_cents']<=1e-4
        if 'fit_manifests' in d:
            arm=folder.name
            fit_rows=[p for p in ex.read(d['fit_manifests'][arm]) if 'excluded' not in p and ex.group(p) not in d['folds'][fold-1]]
            assert payload['fit_manifest_sha256']==d['fit_manifest_hashes'][arm]
            assert payload['actual_audio_pairs']==len(fit_rows)
            assert all(p['source']['sha256'] in d['approved_training_sources'] for p in fit_rows)
        fits.append(str(folder))
    arms={}
    for folder in sorted((root/'arms').glob('*/training/median11')):
        evidence=ex.read(folder/'evidence.json')
        assert ex.sha(folder/'accuracy.json')==evidence['accuracy_sha256']
        assert ex.sha(folder/'original_controls.json')==evidence['controls_sha256']
        controls=ex.read(folder/'original_controls.json')
        assert len(controls)==50 and all(c['delivered_identity'] for c in controls)
        result=ex.read(folder/'accuracy.json')
        files={ex.sha(p):p for p in (folder/'audio').rglob('*_candidate.wav')}
        verified=set()
        for i,row in enumerate(result['rows']):
            if 'excluded' in row:continue
            for view,trace in row['trace'].items():
                if trace['outcome']!='VERIFIED_FIXTURE_CANDIDATE':continue
                audit=trace['transaction']['delivered_wav_audit']
                assert audit['passed'] and audit['outside_support_unchanged']
                acoustic=audit['acoustic_audit']
                assert acoustic['passed'] and acoustic['fine_wrong_direction_frames']==0 and acoustic['coarse_wrong_direction_frames']==0
                path=files[audit['sha256']]
                audio,sr=sf.read(path,dtype='float32')
                assert hashlib.sha256(audio.tobytes()).hexdigest()==audit['decoded_waveform_sha256']
                source,source_sr=sf.read(manifest[i]['path_'+view],dtype='float32')
                assert source_sr==sr==22050 and source.shape==audio.shape
                outside=np.ones(len(audio),bool)
                for lo,hi in trace['permission']['intervals_seconds']:
                    outside[int(round(lo*sr)):int(round(hi*sr))]=False
                assert np.array_equal(source[outside],audio[outside])
                verified.add(str(path))
        arms[folder.parents[1].name]=dict(originals_identity=50,unique_verified_wavs=len(verified))
    protected=ex.read(root.parent/'2026-10-28_v7_guard/declaration.json')['protected_files']
    for p,h in protected.items():assert ex.sha(p)==h,p
    assert not (root.parent/'active_model.json').exists()
    return dict(study=str(root),hash_checks=checked,cv_fits=len(fits),arms=arms,
                protected_files_unchanged=len(protected),activation_absent=True,
                research_validation_passed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();print(json.dumps(verify(args.root.resolve()),indent=2))
