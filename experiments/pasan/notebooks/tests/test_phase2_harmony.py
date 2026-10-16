"""Music/ML invariants that can silently invalidate a training experiment."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase2_harmony.data import (CleanConfig, clean_notes, interval_stats,
    melody_features, chord_targets, complete_windows, LABELS, FAMILIES)
from phase2_harmony.model import HarmonyModel, ModelConfig, metrics, viterbi, progression, harmonize


def test_transposition_and_octave_invariance_even_across_tonic_wrap():
    notes=[{'onset_sec':i*.5,'offset_sec':(i+1)*.5,'pitch':p} for i,p in enumerate([71,73,78,70])]
    original=melody_features(notes,[0,.5,1,1.5,2],11)
    for shift in [-24,-11,1,7,12,25]:
        transposed=[{**n,'pitch':n['pitch']+shift} for n in notes]
        np.testing.assert_allclose(original,melody_features(transposed,[0,.5,1,1.5,2],(11+shift)%12))


def test_union_gaps_and_subbeat_order():
    assert interval_stats([(0,2),(1,3),(4,5)],0,6)==(4/6,1)
    notes=[{'onset_sec':0,'offset_sec':.25,'pitch':60},
           {'onset_sec':.75,'offset_sec':1,'pitch':64}]
    x=melody_features(notes,[0,1],0)
    assert x[0,0]==1 and x[0,45+4]==1
    assert x[0,15+12]==0 and x[0,30+12]==0
    assert not np.array_equal(x,melody_features([{**n,'pitch':64 if n['pitch']==60 else 60} for n in notes],[0,1],0))


def test_conservative_cleaning_does_not_snap_chromatic_notes_or_fill_gaps():
    notes=[{'onset_sec':0,'offset_sec':.3,'midi_pitch_rounded':60},
           {'onset_sec':.4,'offset_sec':.5,'midi_pitch_rounded':72},
           {'onset_sec':.6,'offset_sec':.9,'midi_pitch_rounded':61},
           {'onset_sec':2,'offset_sec':2.01,'midi_pitch_rounded':64},
           {'onset_sec':3,'offset_sec':2,'midi_pitch_rounded':60}]
    clean,actions=clean_notes(notes+[notes[0]],4)
    assert [n['pitch'] for n in clean]==[60,72,61]
    assert clean[1]['onset_sec']==.4
    assert {'too_short_or_outside','invalid_event','exact_duplicate'} <= {a['action'] for a in actions}


def test_chord_masks_and_rare_quality_are_not_coerced_to_major():
    events=[{'start_sec':0,'end_sec':.5,'relative_root_semitones':0,'family':'MAJ','mapping_status':'mapped'},
            {'start_sec':.5,'end_sec':1,'relative_root_semitones':7,'family':'DOM7','mapping_status':'mapped'},
            {'start_sec':1,'end_sec':2,'relative_root_semitones':6,'family':'HDIM7','mapping_status':'mapped'}]
    y,c,cov=chord_targets(events,[0,1,2])
    assert y[0]==-100 and y[1]==6*len(FAMILIES)+FAMILIES.index('HDIM7')
    assert cov==1


def test_incomplete_or_changed_meter_bars_do_not_form_windows():
    beats=[]
    for bar in range(10):
        for pos in range(1,5):
            beats.append({'time_sec':len(beats)*.5,'bar_index':bar,'beat_in_bar':pos,
                          'meter_numerator':4,'meter_denominator':4})
    assert len(list(complete_windows(beats,CleanConfig())))==1
    beats[12]['beat_in_bar']=2
    assert len(list(complete_windows(beats,CleanConfig())))==0


def test_padding_cannot_change_real_beat_predictions():
    torch.manual_seed(3);torch.set_num_threads(2)
    model=HarmonyModel(3,ModelConfig(width=32,layers=1,dropout=0)).eval()
    batch={'x':torch.randn(1,8,77),'lengths':torch.tensor([5]),'mode':torch.tensor([0]),'genres':torch.tensor([[1.,0.,0.]])}
    with torch.no_grad():
        a=model(batch)['logits'][:,:5]
        batch['x'][:,5:]=torch.randn(1,3,77)*100
        b=model(batch)['logits'][:,:5]
    torch.testing.assert_close(a,b)


def test_masked_beats_do_not_create_fake_chord_change_pairs():
    y=np.array([[0,-100,1,1]])
    p=np.array([[0,3,1,1]])
    m=metrics(y,p,[4],[0])
    assert m['accuracy']==1 and m['change_f1']==0
    assert m['segment_edit_similarity']==1


def test_decoder_uses_predictions_only_and_retains_progression_durations():
    z=np.zeros((1,3,len(LABELS)));z[0,0,0]=2;z[0,1,63]=3;z[0,2,63]=3
    path=viterbi(z,[3],[0],np.zeros((2,len(LABELS),len(LABELS))),0)[0]
    assert path.tolist()==[0,63,63]
    out=progression(path,[0,.5,1,1.5])
    assert [v['roman'] for v in out]==['I:MAJ','V:MAJ']
    assert out[1]['start_sec']==.5 and out[1]['end_sec']==1.5


def test_inference_retains_short_boundary_tails_and_is_transposition_invariant():
    torch.manual_seed(4)
    model=HarmonyModel(2,ModelConfig(width=32,layers=1,dropout=0)).eval()
    # Long valid note, only its last 20ms fragment lies in the requested window.
    notes=[{'onset_sec':.98,'offset_sec':1.5,'midi_pitch_rounded':61}]
    result=harmonize(model,notes,[0,.5,1],'major',['pop'],['<unknown>','pop'],tonic_pc=0)
    x=melody_features([{'onset_sec':.98,'offset_sec':1.5,'pitch':61}],[0,.5,1],0)
    with torch.no_grad():
        expected=model({'x':torch.tensor(x)[None],'lengths':torch.tensor([2]),
                        'mode':torch.tensor([0]),'genres':torch.tensor([[0.,1.]])})['logits'][0].numpy()
    np.testing.assert_allclose(result['logits'],expected,atol=1e-5)
    shifted=[{**n,'midi_pitch_rounded':n['midi_pitch_rounded']+14} for n in notes]
    other=harmonize(model,shifted,[0,.5,1],'major',['pop'],['<unknown>','pop'],tonic_pc=2)
    np.testing.assert_allclose(result['logits'],other['logits'],atol=1e-5)


def test_long_song_overlap_inference_covers_every_beat():
    torch.manual_seed(5)
    model=HarmonyModel(1,ModelConfig(width=32,layers=1,dropout=0)).eval()
    notes=[{'onset_sec':i*.5,'offset_sec':(i+1)*.5,'midi_pitch_rounded':60+i%5} for i in range(95)]
    result=harmonize(model,notes,np.arange(96)*.5,'major',[],['<unknown>'],normalized=True)
    assert len(result['beat_chords'])==95 and np.isfinite(result['logits']).all()
    assert result['progression'][0]['start_beat']==0 and result['progression'][-1]['end_beat']==95
