"""Evaluate an energy gate without changing or retraining the pitch estimator."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from melody_understanding import analyze_vocal
from melody_understanding.audio import prepare_audio
from melody_understanding.annotations import read_midi_notes
from melody_understanding.energy import gate_voicing
from melody_understanding.evaluation import notes_with_smoothing,score_notes,score_frames


# Scoring a Voicing Gate with the Original Strict Metrics
def measure(reference,features,end,mode,audio):
    """Apply a chosen mode and measure original strict note/frame criteria."""
    # Optionally mask voicing, then segment notes without pitch smoothing.
    if mode!='none':features,gate=gate_voicing(features,audio,mode)
    else:gate=None
    notes=notes_with_smoothing(features,end,1)
    return {**score_notes(reference,notes),**score_frames(reference,features),
        'predicted_notes':len(notes),'gate':gate}


# Selecting an Energy Gate on TRAIN and Comparing It on VALID
def main():
    """Choose mode from TRAIN, then compare unchanged baseline on VALID."""
    # Reuse the fixed pilot recordings and define all candidate energy policies.
    cohort=json.loads((ROOT/'experiments/vishwa/results/pilot_001/cohort.json').read_text())
    modes=['none','relative_20db','relative_10db','adaptive_noise2']
    rows=[];selected=None
    for split,keys in cohort['ids'].items():
        # Evaluate every TRAIN candidate and freeze the selected mode before VALID.
        candidates=modes if split=='TRAIN' else list(dict.fromkeys(['none',selected]))
        for key in keys:
            with np.load(ROOT/'datasets/humtrans/derived/research/pilot_features'/f'{key}.npz') as f:
                features={k:f[k] for k in f.files}
            audio=prepare_audio(ROOT/'datasets/humtrans/downloads/wav_data_sync_with_midi'/f'{key}.wav')
            reference,_=read_midi_notes(ROOT/'datasets/humtrans/downloads/midi_data'/f'{key}.mid')
            for mode in candidates:
                rows.append({'key':key,'split':split,'mode':mode,
                    **measure(reference,features,audio.source_duration_s,mode,audio)})
        # Choose the highest macro note F1; candidate order gives the baseline tie priority.
        if split=='TRAIN':
            means={m:float(np.mean([r['note_f1'] for r in rows if r['mode']==m])) for m in modes}
            selected=max(modes,key=lambda m:means[m])
            print('Frozen TRAIN selection:',selected,means,flush=True)
    # Average recording-level metrics within each split and gate mode.
    aggregate=[]
    metrics=['note_f1','note_offset_f1','midi_frame_pitch_agreement_50c','midi_frame_overall_agreement_50c','midi_voicing_recall']
    for split in cohort['ids']:
        for mode in dict.fromkeys(r['mode'] for r in rows if r['split']==split):
            sample=[r for r in rows if r['split']==split and r['mode']==mode]
            aggregate.append({'split':split,'mode':mode,'n':len(sample),
                **{m:float(np.mean([r[m] for r in sample])) for m in metrics}})
    # Independent known-tone sanity result uses the same frozen selected gate.
    synthetic=[]
    source=json.loads((ROOT/'experiments/vishwa/results/synthetic_pitch_check.json').read_text())
    # Compare all modes on clean and noisy known-tone signals as a sensitivity check.
    for condition in ['clean','10db_noise']:
        path=ROOT/'datasets/humtrans/derived/research/synthetic'/f'{condition}.wav'
        result,features=analyze_vocal(path);audio=prepare_audio(path)
        for mode in modes:
            synthetic.append({'condition':condition,'mode':mode,**measure(source['reference_notes'],features,result['source_duration_s'],mode,audio)})
    # Save real-recording and synthetic results with the fixed evaluation settings.
    report={'recorded_date':'2026-10-21','selected_mode':selected,'selection':'TRAIN macro note F1; baseline wins ties by candidate order',
        'config':{'modes':modes,'rms_frame':512,'hop_length':256,'note_min_s':.06,'onset_tolerance_s':.05,'pitch_tolerance_cents':50},
        'aggregate':aggregate,'synthetic_sensitivity':synthetic,'per_recording':rows,
        'limitations':['Repeated small validation pilot; not independent final evaluation',
            'Energy thresholds can suppress quiet vocals; frame recall may fall',
            'No raw labels, F0 values, or TEST predictions changed']}
    (ROOT/'experiments/vishwa/results/energy_pilot.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_recording'},indent=2))

if __name__=='__main__':main()
