"""Regression checks for the label, alignment, evaluation and persistence risks."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase4_harmony.data import (CORE,IGNORE,Config,features,function_id,prepare_chords,
                                 targets,root_pc,verify_snapshot,digest)
from phase4_harmony.experiment import class_weights,binary_weight,expand
from phase4_harmony.model import FunctionTransformer,ModelConfig
from phase4_harmony.metrics import metrics,bar_oracle
from phase2_harmony.data import complete_windows,CleanConfig


def event(a,b,y):return {'start_sec':a,'end_sec':b,'function':y}


def test_actual_minor_tonic_and_raised_leading_tone():
    assert function_id('A','min',9,'minor')==0
    assert function_id('D','min',9,'minor')==1
    assert function_id('E','7',9,'minor')==2
    assert function_id('G#','dim',9,'minor')==2
    assert function_id('G','maj',9,'minor')==IGNORE
    assert function_id('E','min',9,'minor')==IGNORE
    assert function_id('C','maj',9,'minor')==IGNORE  # relative major is not tonic!
    assert function_id('A','min',0,'major')==IGNORE
    assert function_id('A','min',0,'major','substitute')==0
    assert function_id('G','sus4',0,'major')==IGNORE
    assert function_id('C','maj(*3)',0,'major')==IGNORE


def test_enharmonic_roots_and_inversion_fields():
    assert root_pc('B#')==root_pc('C')==0
    assert root_pc('F♯')==root_pc('Gb')==6
    obj={'duration_sec':2,'events':[{'start_sec':0,'end_sec':2,'chord_label':'G:7/3','root':'G','quality':'7'}]}
    assert prepare_chords(obj,0,'major','core')[0]['function']==2
    obj['events'][0]['root']='C'
    with pytest.raises(ValueError,match='disagree'):prepare_chords(obj,0,'major','core')


def test_aggregate_by_function_without_filling_unknown_gaps():
    events=[event(0,.45,0),event(.45,.9,0),event(.9,1,IGNORE)]
    y,c=targets(events,np.array([0.,1.]),.8)
    assert y.tolist()==[0];assert c[0]==pytest.approx(.9)
    y,c=targets([event(0,.4,0),event(.6,1,0)],np.array([0.,1.]),.9)
    assert y.tolist()==[IGNORE];assert c[0]==pytest.approx(.8)
    y,_=targets([event(0,.5,0),event(.5,1,1)],np.array([0.,1.]),.7)
    assert y.tolist()==[IGNORE]


def test_overlapping_chords_rejected_even_if_same_function():
    obj={'duration_sec':2,'events':[{'start_sec':0,'end_sec':1,'chord_label':'N'},
                                    {'start_sec':.9,'end_sec':2,'chord_label':'N'}]}
    with pytest.raises(ValueError,match='overlapping'):prepare_chords(obj,0,'major','core')


def test_carry_note_no_fabricated_onset_and_transposition_invariance():
    notes=[{'onset_sec':-1.,'offset_sec':.8,'pitch':60},{'onset_sec':1.,'offset_sec':2.,'pitch':64}]
    bounds=np.array([0.,1.,2.]);beats=[dict(beat_in_bar=i+1,meter_numerator=4,meter_denominator=4) for i in range(2)]
    x=features(notes,bounds,0,beats)
    assert x[0,60:72].sum()==0
    assert x[0,12]>0
    shifted=[{**n,'pitch':n['pitch']+5} for n in notes]
    np.testing.assert_allclose(x,features(shifted,bounds,5,beats),atol=1e-7)


def test_complete_bar_requires_real_next_boundary():
    beats=[dict(time_sec=i/2,bar_index=i//4,beat_in_bar=i%4+1,meter_numerator=4,meter_denominator=4) for i in range(8)]
    cfg=CleanConfig(window_bars=2)
    assert list(complete_windows(beats,cfg))==[]
    beats.append(dict(time_sec=4.,bar_index=2,beat_in_bar=1,meter_numerator=4,meter_denominator=4))
    idx,bounds=next(complete_windows(beats,cfg));assert len(idx)==8;assert bounds[-1]==4.


def test_transition_metrics_do_not_bridge_masks_or_windows():
    y=np.array([[0,IGNORE,1,1],[2,2,IGNORE,0]])
    p=np.array([[0,0,1,1],[2,2,0,0]])
    m=metrics(y,p)
    assert m['accuracy']==1.;assert m['valid_pairs']==2;assert m['true_changes']==0
    m=metrics(np.array([[0,1,2]]),np.array([[0,1,1]]))
    assert m['change_precision']==1.;assert m['change_recall']==.5
    assert m['transition_pair_accuracy']==.5


def test_bar_predictions_evaluated_on_common_beat_support():
    bar=np.array([[0,0,1,1,-1]])
    pred=expand(np.array([[1,2]]),bar,'bar')
    y=np.array([[1,2,2,2,IGNORE]])
    assert pred.tolist()==[[1,1,2,2,1]]
    m=metrics(y,pred);assert m['valid_positions']==4;assert m['accuracy']==.75
    assert bar_oracle(y,bar,2)['accuracy']==.75


def test_weights_use_only_supplied_training_labels():
    y=np.array([[0]*8+[1]*2+[2]+[IGNORE]])
    w,c=class_weights(y,True)
    assert c.tolist()==[8,2,1];assert w[2]>w[1]>w[0]
    assert class_weights(y,False)[0].tolist()==[1,1,1]
    weight,pos,neg=binary_weight(np.array([[0,0,IGNORE,1,2]]))
    assert (pos,neg)==(1,1);assert weight==1.


@pytest.mark.parametrize('resolution',['beat','bar'])
def test_forward_backward_checkpoint_roundtrip_and_padding(resolution,tmp_path):
    torch.manual_seed(7);torch.set_num_threads(1)
    cfg=ModelConfig(resolution=resolution,genre=True,d_model=16,layers=1,dropout=0.)
    model=FunctionTransformer(77,3,2,cfg)
    x=torch.randn(2,8,77);lengths=torch.tensor([8,4]);bars=torch.tensor([[0]*4+[1]*4,[0]*2+[1]*2+[-1]*4])
    mode=torch.tensor([0,1]);genres=torch.tensor([[0.,1.,1.],[1.,0.,0.]])
    model.eval();logits,change=model(x,lengths,bars,mode,genres)
    assert logits.shape==(2,8 if resolution=='beat' else 2,3)
    x2=x.clone();x2[1,4:]=100
    other=model(x2,lengths,bars,mode,genres)[0]
    np.testing.assert_allclose(logits[1,:4 if resolution=='beat' else 2].detach(),other[1,:4 if resolution=='beat' else 2].detach(),atol=1e-6)
    logits.sum().backward();assert model.function_head.weight.grad.abs().sum()>0
    p=tmp_path/'test.pt';torch.save(model.state_dict(),p)
    second=FunctionTransformer(77,3,2,cfg);second.load_state_dict(torch.load(p,weights_only=True));second.eval()
    torch.testing.assert_close(logits,second(x,lengths,bars,mode,genres)[0])


def test_snapshot_hash_tampering_fails(tmp_path):
    p=tmp_path/'features.npz';p.write_bytes(b'original')
    (tmp_path/'metadata.json').write_text(json.dumps({'schema_version':1,'labels':['TONIC','PREDOMINANT','DOMINANT'],'files_sha256':{'features.npz':digest(p)}}))
    verify_snapshot(tmp_path);p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash_mismatch'):verify_snapshot(tmp_path)


def test_replicated_selection_uses_mean_and_fixed_deployment_seed(tmp_path,monkeypatch):
    from phase4_harmony import experiment as exp
    snapshot=tmp_path/'snapshot';snapshot.mkdir();(snapshot/'metadata.json').write_text('{}')
    runroot=tmp_path/'runs'
    specs=[{'name':'A'},{'name':'B'}]
    # A wins one lucky seed; B wins mean performance. Seed 42 remains deployment.
    def fake_suite(snapshot,root,config,specs):
        import pandas as pd
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        scores={'A':.9 if config.seed==42 else .1,'B':.5}
        rows=[]
        for name,score in scores.items():
            d=root/name;d.mkdir();(d/'best.pt').write_bytes(str(config.seed).encode());(d/'result.json').write_text('{}')
            rows.append(dict(name=name,resolution='bar',genre=False,weighted=True,accuracy=score,macro_f1=score,change_f1=score,transition_pair_accuracy=score,selection_score=score,no_melody_macro_f1=0.))
        (root/'baselines.json').write_text('{}')
        return pd.DataFrame(rows),{}
    monkeypatch.setattr(exp,'run_suite',fake_suite)
    comparison,selection=exp.run_replicated_suite(snapshot,runroot,seeds=(42,43,44),specs=specs)
    assert selection['configuration']=='B';assert selection['name']=='seed_42/B'
    assert selection['test_used_for_selection'] is False
    assert selection['seed']==42
