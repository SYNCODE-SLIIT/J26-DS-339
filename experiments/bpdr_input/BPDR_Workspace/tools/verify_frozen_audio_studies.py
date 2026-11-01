"""Verify saved ensemble/R3 studies without rerendering or accessing fresh data."""
from pathlib import Path
import hashlib
import sys
import numpy as np
import soundfile as sf
import torch

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.repair.repair_controller import decision_hash


def outputs(folder,manifest,sources):
    evidence=ex.read(folder/'evidence.json');result=ex.read(folder/'accuracy.json')
    assert ex.sha(folder/'accuracy.json')==evidence['accuracy_sha256']
    assert ex.sha(folder/'original_controls.json')==evidence['controls_sha256']
    rows=ex.read(manifest);assert len(rows)==len(result['rows'])==200
    assert sum('excluded' not in r for r in rows)==108
    files={ex.sha(p):p for p in (folder/'audio').rglob('*_candidate.wav')};verified=set()
    for i,row in enumerate(result['rows']):
        if 'excluded' in row:continue
        for view,trace in row['trace'].items():
            plan=trace['permission'];assert decision_hash(plan)==plan['decision_sha256']
            if trace['outcome']!='VERIFIED_FIXTURE_CANDIDATE':continue
            audit=trace['transaction']['delivered_wav_audit'];acoustic=audit['acoustic_audit']
            assert audit['passed'] and audit['outside_support_unchanged'] and acoustic['passed']
            assert acoustic['fine_wrong_direction_frames']==acoustic['coarse_wrong_direction_frames']==0
            path=files[audit['sha256']];wave,sr=sf.read(path,dtype='float32')
            assert hashlib.sha256(wave.tobytes()).hexdigest()==audit['decoded_waveform_sha256']
            source,source_sr=sf.read(rows[i]['path_'+view],dtype='float32')
            assert source_sr==sr==22050 and wave.shape==source.shape
            assert hashlib.sha256(source.tobytes()).hexdigest()==plan['source_sha256']
            outside=np.ones(len(wave),bool)
            for lo,hi in plan['intervals_seconds']:outside[int(round(lo*sr)):int(round(hi*sr))]=False
            assert np.array_equal(wave[outside],source[outside])
            verified.add(str(path))
    controls=ex.read(folder/'original_controls.json');assert len(controls)==len(sources)==50
    for control,source in zip(controls,sources):
        assert control['source_sha256']==source['source']['sha256'] and control['delivered_identity']
        wave,sr=sf.read(source['working_audio'],dtype='float32');transaction=control['transaction'];plan=transaction['plan']
        assert hashlib.sha256(wave.tobytes()).hexdigest()==plan['source_sha256']
        assert decision_hash(plan)==plan['decision_sha256']
        assert transaction['outcome'] in ('IDENTITY','WITHHELD','ROLLED_BACK')
    return dict(encoded_verified_wavs=len(verified),original_controls_identity=50,attempted=200,scoreable=108)


def main():
    torch.set_num_threads(2)
    import evaluate_joint_ensemble as ensemble
    import run_rubberband_renderer_study as r3
    d=ensemble.declare();d3,arm=r3.declare()
    rows=ex.read(ensemble.ROOT/'train_pairs.json');sources=ex.read(ensemble.ROOT/'anchor_sources.json')
    for name,run,condition in ensemble.MEMBERS:
        for fold,groups in enumerate(d['folds'],1):
            path=ensemble.ROOT.parent/run/f'cross_validation/fold_{fold}/{condition}/checkpoint.pt'
            payload=torch.load(path,map_location='cpu',weights_only=True)
            assert_independent(payload['training_groups'],payload['training_sources'],[r for r in rows if ex.group(r) in groups])
    with np.load(ensemble.ROOT/'validation/predictions.npz') as a,np.load(ensemble.ROOT/'training/median11/predictions.npz') as b:
        for i,row in enumerate(rows):
            if 'excluded' in row:continue
            for view in ('A','C'):np.testing.assert_array_equal(a[f'{i}_{view}'],b[f'{i}_{view}'])
    reports={'ensemble':outputs(ensemble.ROOT/'training/median11',ensemble.ROOT/'train_pairs.json',sources),
             'rubberband_r3':outputs(arm/'training/span_median',r3.V8/'train_pairs.json',ex.read(r3.V8/'anchor_sources.json'))}
    protected=ex.read(ensemble.ROOT.parent/'2026-10-28_v7_guard/declaration.json')['protected_files']
    for path,digest in protected.items():assert ex.sha(path)==digest
    assert not (ensemble.ROOT.parent/'active_model.json').exists()
    for name,folder in [('ensemble',ensemble.ROOT),('rubberband_r3',r3.ROOT)]:
        ex.write(folder/'integrity_receipt.json',dict(**reports[name],protected_files_unchanged=6,
                 activation_absent=True,frozen_declaration_sha256=ex.sha(folder/'declaration.json'),
                 verifier_sha256=ex.sha(Path(__file__)),research_validation_passed=False))
    print('FROZEN_AUDIO_VERIFIED',reports,flush=True)


if __name__=='__main__':main()
