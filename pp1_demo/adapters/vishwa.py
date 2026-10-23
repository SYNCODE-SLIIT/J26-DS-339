import json
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
from pp1_demo.config import ROOT
from pp1_demo.contracts import sha
sys.path.insert(0,str(ROOT/'src'))
from melody_understanding.pipeline import analyze_vocal
from melody_understanding.audio import prepare_audio


def analyze(path):
    result,features=analyze_vocal(path,trim=False)
    # Owner baseline has a historical warning predating pilot training.
    result['warnings']=[w for w in result['warnings'] if w!='No supervised CNN-Transformer model has been trained']
    result['warnings'].append('Learned score-class pilots exist; pYIN supplies the absolute-pitch note stream in this demo.')
    audio=prepare_audio(path,trim=False)
    return result,features,audio

@torch.inference_mode()
def pilot_predictions(features):
    from melody_understanding.score_model import ScoreModel
    folder=ROOT/'experiments/vishwa/checkpoints/weak_score_pilot'
    verify_bundle()
    torch.set_num_threads(2)
    result={}
    with np.load(folder/'train_normalization.npz',allow_pickle=False) as z:
        x=(features['mel_db']-z['mean'][:,None])/z['std'][:,None]
    for name in ['cnn','cnn_transformer']:
        checkpoint=folder/(name+'.pt')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model=ScoreModel(attention=name=='cnn_transformer')
        model.load_state_dict(saved['state_dict'],strict=True);model.eval()
        probabilities=[]
        for start in range(0,x.shape[1],256):
            size=min(256,x.shape[1]-start)
            chunk=np.zeros((1,80,256),np.float32);chunk[0,:,:size]=x[:,start:start+size]
            padding=torch.ones((1,256),dtype=torch.bool);padding[:,:size]=False
            probabilities.extend(model(torch.from_numpy(chunk),padding).softmax(-1)[0,:size].tolist())
        result[name]={'probabilities':probabilities,'checkpoint_sha256':sha(checkpoint),'classes':['rest','C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']}
    return result


def verify_bundle():
    folder=ROOT/'experiments/vishwa/checkpoints/weak_score_pilot'
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['format']!='vishwa-weak-score-pilot-v1':
        raise ValueError('Unsupported Vishwa checkpoint bundle')
    for name,expected in manifest['files'].items():
        if Path(name).name != name or sha(folder/name)!=expected:
            raise ValueError('Vishwa checkpoint integrity failure: '+name)
    if hashlib.sha256((ROOT/'src/melody_understanding/score_model.py').read_text(encoding='utf-8').encode('utf-8')).hexdigest()!=manifest['model_source_sha256']:
        raise ValueError('Vishwa model definition differs from the checkpoint bundle')
    return manifest


def predict(path,folder):
    """Live accepted-audio features and honest parallel learned-class predictions."""
    import soundfile as sf
    from pp1_demo.contracts import validate_notes
    manifest=verify_bundle()
    result,features,audio=analyze(path)
    validate_notes(result['notes'],audio.source_duration_s)
    for note in result['notes']:
        note['source']='pyin_vocal'
    pilots=pilot_predictions(features)
    sf.write(Path(folder)/'vishwa_prepared.wav',audio.samples,audio.sample_rate)
    np.savez_compressed(Path(folder)/'vishwa_features.npz',**features)
    step=max(1,int(np.ceil(len(features['times_s'])/480)))
    f0=features['f0_hz'][::step]
    voiced=features['voiced'][::step]
    result['chart']={'times_s':features['times_s'][::step].tolist(),
        'f0_hz':[float(x) if ok and np.isfinite(x) else None for x,ok in zip(f0,voiced)],
        'mel_db':features['mel_db'][:,::step].tolist()}
    result['pilot_models']={}
    for name,pilot in pilots.items():
        scores=np.asarray(pilot['probabilities'])
        if scores.shape!=(len(features['times_s']),13) or not np.isfinite(scores).all():
            raise ValueError('Invalid pilot prediction dimensions or values')
        result['pilot_models'][name]={'checkpoint_sha256':pilot['checkpoint_sha256'],
            'classes':pilot['classes'],'predicted_classes':scores.argmax(axis=1).tolist(),
            'chart_probabilities':scores[::step].tolist(),
            'parameters':manifest['models'][name]['parameters'],
            'historical_validation_recording_macro_accuracy':manifest['models'][name]['validation_recording_macro_accuracy']}
    result['bundle_manifest_sha256']=sha(ROOT/'experiments/vishwa/checkpoints/weak_score_pilot/manifest.json')
    result['input_audio_sha256']=sha(path)
    result['checkpoint_scope']=manifest['scope']
    result['downstream_source']='pYIN note events; trained CNN/hybrid predictions are separate experiments'
    return result


def merge_context(saved_notes,live_notes,visible_start,duration):
    """Replace the entire crop, including rests; retain only outside saved fragments."""
    if not np.isfinite(visible_start+duration) or visible_start<0 or duration<=0:
        raise ValueError('Invalid crop interval')
    end=visible_start+duration
    merged=[]
    for note in saved_notes:
        a=note['onset_s'];b=a+note['duration_s']
        for left,right in [(a,min(b,visible_start)),(max(a,end),b)]:
            if right>left:
                merged.append(dict(note,onset_s=left,duration_s=right-left,context_provenance='saved_outside_crop'))
    for note in live_notes:
        if note['onset_s']<0 or note['duration_s']<=0 or note['onset_s']+note['duration_s']>duration+1e-6:
            raise ValueError('Live note outside audible crop')
        merged.append(dict(note,onset_s=note['onset_s']+visible_start,source='pyin_vocal',context_provenance='live_accepted_audio'))
    return sorted(merged,key=lambda n:(n['onset_s'],n['pitch_midi']))
