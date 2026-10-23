"""Run the saved model on a new melody without reference chords or key comparison."""
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from phase2_harmony.data import CleanConfig,clean_notes,complete_windows
from .data import LABELS,features
from .experiment import atomic_json,load_model


@torch.no_grad()
def harmonize(melody,beats,key,checkpoint,snapshot,genres=()):
    """Predict complete non-overlapping windows; require a melody/user supplied key.

    key: {'tonic_pc': integer, 'mode': 'major'|'minor'}.
    Returns function segments, not uniquely specified Roman-numeral chords.
    """
    model,_,_,meta,_=load_model(checkpoint,snapshot)
    if key.get('tonic_pc') not in range(12) or key.get('mode') not in {'major','minor'}:
        raise ValueError('A tonic pitch class and major/minor mode are required')
    duration=float(melody['duration_sec'])
    notes,actions=clean_notes(melody['notes'],duration,CleanConfig())
    max_end=-1.
    for n in notes:
        if n['onset_sec']<max_end-1e-4:raise ValueError('Ambiguous overlapping melody pitches')
        max_end=max(max_end,n['offset_sec'])
    entries=beats['beats'];times=np.array([b['time_sec'] for b in entries])
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Invalid beats')
    vocab=meta['genre_vocab'];genre=np.zeros((1,len(vocab)),np.float32)
    ids=[vocab.index(g) for g in genres if g in vocab];genre[0,ids or [0]]=1
    cells=[]
    for indices,bounds in complete_windows(entries,CleanConfig(window_bars=meta['config']['bars'])):
        local=[entries[j] for j in indices];meter=local[0]['meter_numerator'];after=entries[indices[-1]+1]
        if after['beat_in_bar']!=1 or after['bar_index']!=local[-1]['bar_index']+1:continue
        x=torch.from_numpy(features(notes,bounds,key['tonic_pc'],local))[None]
        bars=torch.arange(len(indices))[None]//meter
        logits,_=model(x,torch.tensor([len(indices)]),bars,torch.tensor([int(key['mode']=='minor')]),torch.from_numpy(genre))
        probabilities=logits.softmax(-1)[0].numpy()
        output_bounds=bounds if model.config.resolution=='beat' else bounds[::meter]
        for i,prob in enumerate(probabilities):
            cells.append({'start_sec':float(output_bounds[i]),'end_sec':float(output_bounds[i+1]),
                          'function':LABELS[int(prob.argmax())],'probabilities':dict(zip(LABELS,map(float,prob)))})
    segments=[]
    for cell in cells:
        if segments and segments[-1]['function']==cell['function'] and abs(segments[-1]['end_sec']-cell['start_sec'])<1e-6:
            segments[-1]['end_sec']=cell['end_sec']
        else:segments.append({k:cell[k] for k in ['start_sec','end_sec','function']})
    return {'resolution':model.config.resolution,'key':key,'genres':list(genres),'cells':cells,'segments':segments,
            'note_action_count':len(actions),'unprocessed_partial_bars':'Only complete windows with a real next downbeat are emitted.',
            'label_scope':'Core harmonic-function proxies learned from automatic annotations; probabilities are not calibrated.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['melody','beats','key','checkpoint','snapshot','out']:p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--genres',nargs='*',default=[]);args=p.parse_args()
    melody=json.loads(args.melody.read_text());beats=json.loads(args.beats.read_text());key=json.loads(args.key.read_text())
    key=key.get('melody_key',{}).get('selected') or key
    result=harmonize(melody,beats,key,args.checkpoint,args.snapshot,args.genres)
    args.out.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.out,result)


if __name__=='__main__':main()
