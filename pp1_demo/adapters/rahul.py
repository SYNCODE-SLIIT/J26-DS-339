import numpy as np
import torch
from functools import lru_cache
from pp1_demo.config import RAHUL,VOCAB,PC_NAMES
from pp1_demo.contracts import sha
from experiments.rahul.src.encode_features import FeatureEncoder
from experiments.rahul.src.model.multitask_transformer import load_multitask_checkpoint
from experiments.rahul.src.build_synthetic_dataset import context
from experiments.rahul.src.canonicalize_chords import canonicalize_chord

@lru_cache(maxsize=2)
def load(checkpoint_hash,feature_hash,target_hash):
    return load_multitask_checkpoint(RAHUL,VOCAB),FeatureEncoder.load(VOCAB)

@torch.inference_mode()
def predict(notes,chords,settings):
    model,encoder=load(sha(RAHUL),sha(VOCAB/'feature_vocabularies.json'),sha(VOCAB/'target_vocabulary.json'))
    if settings.genre not in encoder.genres: raise ValueError('Genre absent from Rahul vocabulary')
    if 'major' not in encoder.vocabs['mode'].token_to_id: raise ValueError('Major mode absent from saved vocabulary')
    suffix={'MAJ':'','MIN':'m','DOM':'7','DIM':'dim','HDIM':'m7b5','AUG':'aug','SUS':'sus'}
    labels=[canonicalize_chord(PC_NAMES[c['root_pc']]+suffix[c['family']])[0] for c in chords]
    encoded=[]; unknown={}
    for i,c in enumerate(chords):
        a,b=c['start_s'],c['end_s']
        selected=[n for n in notes if n['onset_s']<b and n['onset_s']+n['duration_s']>a]
        family={'MAJ':'maj','MIN':'min','DOM':'dom','DIM':'dim','HDIM':'hdim','AUG':'aug','SUS':'sus'}[c['family']]
        root=PC_NAMES[c['root_pc']]
        roman,function=context({'basic_chord':labels[i],'basic_root':root,'basic_quality':family},PC_NAMES[settings.tonic_pc],'major')
        row={'basic_chord_canonical':labels[i],'basic_root':root,'basic_quality':family,
             'previous_basic_chord':labels[i-1] if i else 'START',
             'next_basic_chord':labels[i+1] if i+1<len(labels) else 'END',
             'key':PC_NAMES[settings.tonic_pc],'mode':'major','key_confidence':'unknown',
             'roman_numeral':roman,'harmonic_function':function,'genre_tags':[settings.genre],
             'beat_index':i,'bar_number':c['bar_number'],'beat_position':c['beat_position'],'event_duration':b-a,
             'melody_midi':[n['pitch_midi'] for n in selected],
             'melody_pitch_classes':[n['pitch_midi']%12 for n in selected],
             'melody_durations':[n['duration_s'] for n in selected],
             'melody_relative_start':[n['onset_s']-a for n in selected],
             'melody_overlap_durations':[min(b,n['onset_s']+n['duration_s'])-max(a,n['onset_s']) for n in selected],
             'melody_sources':[n.get('source','saved_extraction') for n in selected]}
        for field,vocab in encoder.vocabs.items():
            values=row['melody_sources'] if field=='melody_sources' else [encoder.category(row,field)]
            for v in values:
                if v not in vocab.token_to_id: unknown[field+':'+str(v)]=unknown.get(field+':'+str(v),0)+1
        encoded.append(encoder.encode_inputs(row))
    output=[]
    windows=[]
    for first_bar in range(chords[0]['bar_number'],chords[-1]['bar_number']+1,4):
        indices=[i for i,c in enumerate(chords) if first_bar<=c['bar_number']<first_bar+4]
        if not indices: continue
        left=indices[0]
        rows=[encoded[i] for i in indices]
        inputs={k:torch.from_numpy(np.stack([r[k] for r in rows])[None]) for k in rows[0]}
        inputs['genre_features']=torch.from_numpy(rows[0]['genre_features'][None])
        inputs['beat_attention_mask']=torch.ones((1,len(rows)),dtype=torch.bool)
        inputs['relative_bar_number']=torch.tensor([[chords[left+i]['bar_number']-chords[left]['bar_number'] for i in range(len(rows))]],dtype=torch.float32)
        predictions=model.predict_topk(inputs,k=3)[0]
        windows.append({'bars':len({chords[i]['bar_number'] for i in indices}),'beats':len(rows)})
        for i,candidates in enumerate(predictions):
            if not all(np.isfinite(c['probability']) for c in candidates): raise ValueError('Nonfinite Rahul predictions')
            c=chords[left+i]
            output.append({'start_s':c['start_s'],'end_s':c['end_s'],'label':candidates[0]['label'],
                           'probability':candidates[0]['probability'],'top3':candidates,'source':'rahul_model'})
    return {'chords':output,'checkpoint':str(RAHUL),'checkpoint_sha256':sha(RAHUL),
            'mode':'major','mode_source':'demo_constant','unknown_inputs':unknown,
            'input_summary':{'beats':len(chords),'genre':settings.genre,'genre_dimensions':len(encoder.genres),
                             'mode':'major','windows':windows,'full_chord_classes':model.spec.num_target_classes,
                             'quality_classes':len(model.quality_mapping.labels)},
            'scores':'Raw full-chord probabilities; reserved predictions retained, never replaced silently.'}
