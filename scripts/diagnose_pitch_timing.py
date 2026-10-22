"""Training-only error diagnostics; do not interpret relaxed scores as gains."""
import json
from pathlib import Path
import sys
import numpy as np
import mir_eval.transcription
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from melody_understanding.annotations import read_midi_notes
from melody_understanding.evaluation import notes_with_smoothing


# Investigating Pitch-Class Errors and Onset Tolerance on TRAIN
def main():
    """Measure octave-sensitive versus chroma agreement and onset sensitivity."""
    # Reuse the frozen training cohort and its cached pitch predictions.
    cohort=json.loads((ROOT/'experiments/vishwa/results/pilot_001/cohort.json').read_text())
    rows=[]
    for key in cohort['ids']['TRAIN']:
        with np.load(ROOT/'datasets/humtrans/derived/research/pilot_features'/f'{key}.npz') as f:
            feat={k:f[k] for k in f.files}
        ref,_=read_midi_notes(ROOT/'datasets/humtrans/downloads/midi_data'/f'{key}.mid')
        # Build held MIDI targets and compare only valid voiced pitch estimates.
        times=feat['times_s']; target=np.full(len(times),np.nan)
        for start,end,pitch in ref:
            target[(times>=start)&(times<end)]=pitch
        ref_mask=np.isfinite(target)
        valid=ref_mask & feat['voiced'] & np.isfinite(feat['f0_hz']) & (feat['f0_hz']>0)
        diff=69+12*np.log2(feat['f0_hz'][valid]/440)-target[valid]
        # This diagnostic ignores octave differences but still requires pitch class.
        chroma_distance=np.abs((diff+6)%12-6)
        hop=float(np.median(np.diff(times)))
        predicted=notes_with_smoothing(feat,float(times[-1]+hop),1)
        item={'key':key,'median_pitch_error_semitones':float(np.median(diff)),
            'exact_frame_agreement':float((np.abs(diff)<.5).sum()/ref_mask.sum()),
            'chroma_frame_agreement':float((chroma_distance<.5).sum()/ref_mask.sum()),
            'onset_sensitivity':{}}
        # Vary onset tolerance and octave sensitivity to diagnose scoring failures.
        for tolerance in [.05,.1,.2]:
            for pitch_mode in ['exact','chroma']:
                ref_array=np.asarray(ref)
                est=np.array([[n['onset_s'],n['onset_s']+n['duration_s'],n['pitch_midi']] for n in predicted])
                # Fold pitches into one octave only for the chroma diagnostic.
                def hz(p):
                    if pitch_mode=='chroma':p=60+p%12
                    return 440*2**((p-69)/12)
                score=mir_eval.transcription.precision_recall_f1_overlap(ref_array[:,:2],hz(ref_array[:,2]),est[:,:2],hz(est[:,2]),onset_tolerance=tolerance,pitch_tolerance=50,offset_ratio=None)[2] if len(est) else 0.
                item['onset_sensitivity'][f'{pitch_mode}_{tolerance:.2f}s']=float(score)
        rows.append(item)
    # Aggregate recording-level diagnostics and state their interpretation limits.
    report={'recorded_date':'2026-10-21','scope':'TRAIN pilot only; baseline width 1',
        'n':len(rows),'per_recording':rows,
        'macro_exact_frame_agreement':float(np.mean([r['exact_frame_agreement'] for r in rows])),
        'macro_chroma_frame_agreement':float(np.mean([r['chroma_frame_agreement'] for r in rows])),
        'macro_note_f1_sensitivity':{k:float(np.mean([r['onset_sensitivity'][k] for r in rows])) for k in rows[0]['onset_sensitivity']},
        'limitations':['Relaxed pitch/onset criteria are diagnostics, not model improvements',
            'Octave discrepancy does not establish whether estimator or labels are wrong',
            'No reference-driven correction applied to predictions or dataset']}
    path=ROOT/'experiments/vishwa/results/pitch_timing_diagnostics.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_recording'},indent=2))

if __name__=='__main__':main()
