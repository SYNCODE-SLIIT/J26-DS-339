"""Load only named definitions from the notebook; never execute training cells."""
import ast
import json
import math
from functools import lru_cache
import torch
from torch import nn
from pp1_demo.config import PASAN, PASAN_NOTEBOOK, PC_NAMES
from pp1_demo.contracts import sha

FAMILIES = ['MAJ','MIN','DOM','DIM','HDIM','AUG','SUS']
QUALITIES = ['maj','min','7','dim','hdim7','aug','sus4']
NAMES = ['add_positional_encoding','HarmonyDirectTransformer',
         'build_temporal_attention_bias','HarmonyTemporalTransformer']

@lru_cache(maxsize=2)
def load(checkpoint_hash, notebook_hash):
    ns={'torch':torch,'nn':nn,'math':math,'NUM_ROOTS':12,'NUM_FAMILIES':7}
    notebook=json.loads(PASAN_NOTEBOOK.read_text())
    definitions={}
    for cell in notebook['cells']:
        if cell['cell_type']!='code': continue
        try: tree=ast.parse(''.join(cell['source']))
        except SyntaxError: continue
        for node in tree.body:
            if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in NAMES:
                if node.name in definitions: raise ValueError('Ambiguous model definition: '+node.name)
                definitions[node.name]=node
    for name in NAMES:
        module=ast.Module(body=[definitions[name]],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),str(PASAN_NOTEBOOK),'exec'),ns)
    saved=torch.load(PASAN,map_location='cpu',weights_only=True)
    if saved.get('architecture')!='temporal_transformer' or saved.get('genre_conditioning'):
        raise ValueError('Unexpected Pasan checkpoint architecture')
    model=ns['HarmonyTemporalTransformer'](sigma_beats=saved['temporal_sigma'],temporal_strength=saved['temporal_strength'])
    model.load_state_dict(saved['model_state_dict'],strict=True)
    return model.eval(), {k:saved[k] for k in ('epoch','experiment','validation_metrics','temporal_sigma','temporal_strength')}

@torch.inference_mode()
def predict(notes, beats, settings):
    model,metadata=load(sha(PASAN),sha(PASAN_NOTEBOOK))
    torch.set_num_threads(2)
    shift=((-settings.tonic_pc+6)%12)-6
    output=[]
    # Four-bar windows stay inside the model's 16-bar embedding capacity.
    for left in range(0,len(beats),4*settings.meter):
        window=beats[left:left+4*settings.meter]
        start,end=window[0]['start_s'],window[-1]['end_s']
        duration=end-start
        selected=[n for n in notes if n['onset_s']<end and n['onset_s']+n['duration_s']>start]
        if not selected: raise ValueError('A chord window has no melody; no chord predictions fabricated')
        pitches=[n['pitch_midi']+shift for n in selected]
        if any(p<0 or p>127 for p in pitches): raise ValueError('Key normalization exceeds MIDI range')
        timing=[[max(n['onset_s'],start)-start,min(n['onset_s']+n['duration_s'],end)-max(n['onset_s'],start)] for n in selected]
        batch={'melody_pitch':torch.tensor([pitches],dtype=torch.long),
               'melody_time':torch.tensor([timing],dtype=torch.float32)/duration,
               'melody_padding_mask':torch.zeros((1,len(selected)),dtype=torch.bool),
               'beat_time':torch.tensor([[[b['start_s']-start,b['end_s']-b['start_s']] for b in window]],dtype=torch.float32)/duration,
               'bar_index':torch.tensor([[b['bar_number']-window[0]['bar_number'] for b in window]]),
               'beat_position':torch.tensor([[b['beat_position'] for b in window]]),
               'meter':torch.full((1,len(window)),settings.meter,dtype=torch.long),
               'beat_padding_mask':torch.zeros((1,len(window)),dtype=torch.bool)}
        roots,families=model(batch)
        rp,fp=roots.softmax(-1)[0],families.softmax(-1)[0]
        if not torch.isfinite(rp).all() or not torch.isfinite(fp).all(): raise ValueError('Nonfinite chord predictions')
        for i,b in enumerate(window):
            root,family=int(rp[i].argmax()),int(fp[i].argmax())
            absolute=(root+settings.tonic_pc)%12
            output.append(dict(b,relative_root=root,root_pc=absolute,family=FAMILIES[family],
                label=f'{PC_NAMES[absolute]}:{QUALITIES[family]}',
                root_probability=float(rp[i,root]),family_probability=float(fp[i,family]),source='pasan_model'))
    return {'chords':output,'checkpoint':str(PASAN),'checkpoint_sha256':sha(PASAN),'metadata':metadata,
            'normalization_shift_semitones':shift,'window_bars':4,
            'bridge':'Family representatives: DOM→7, HDIM→hdim7, SUS→sus4; family does not specify every extension.',
            'grid_source':'user_configured','genre_conditioning':False}
