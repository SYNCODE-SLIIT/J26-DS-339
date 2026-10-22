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
    from scripts.train_score_pilot import ScoreModel
    folder=ROOT/'datasets/humtrans/derived/research/weak_score_features'
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
