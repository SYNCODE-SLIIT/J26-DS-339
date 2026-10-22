"""Check pYIN on known low/mid pitches without using HumTrans labels."""
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from melody_understanding import analyze_vocal
from melody_understanding.evaluation import score_notes,score_frames


# Checking Known Pitches under Clean and Noisy Conditions
def main():
    """Run deterministic clean and noisy three-tone sanity experiments."""
    # Construct three known sine notes with rests and short fades at their edges.
    sr=22050
    reference=[(.3,1.1,45),(1.4,2.2,57),(2.5,3.3,69)]
    audio=np.zeros(int(3.6*sr))
    for start,end,pitch in reference:
        begin,stop=round(start*sr),round(end*sr)
        t=np.arange(stop-begin)/sr
        envelope=np.ones(len(t)); ramp=round(.01*sr)
        envelope[:ramp]=np.linspace(0,1,ramp)
        envelope[-ramp:]=np.linspace(1,0,ramp)
        audio[begin:stop]=.2*envelope*np.sin(2*np.pi*(440*2**((pitch-69)/12))*t)
    folder=ROOT/'datasets/humtrans/derived/research/synthetic'
    folder.mkdir(parents=True,exist_ok=True)
    # Add seeded noise scaled to 10 dB SNR over the entire waveform, including rests.
    rng=np.random.default_rng(339)
    noise=rng.standard_normal(len(audio))
    noise*=np.sqrt(np.mean(audio**2)/10)/np.sqrt(np.mean(noise**2))
    rows=[]
    # Run the same baseline on the clean signal and its noisy counterpart.
    for name,signal in [('clean',audio),('10db_noise',audio+noise)]:
        path=folder/f'{name}.wav'
        sf.write(path,signal,sr,subtype='FLOAT')
        result,features=analyze_vocal(path)
        core=[]
        # Measure steady-note pitch error away from uncertain onset and offset regions.
        for start,end,pitch in reference:
            mask=(features['times_s']>=start+.15)&(features['times_s']<end-.15)&features['voiced']&np.isfinite(features['f0_hz'])
            estimate=69+12*np.log2(features['f0_hz'][mask]/440)
            core.append({'reference_midi':pitch,'voiced_core_frames':int(mask.sum()),
                'median_error_cents':float(np.median(estimate-pitch)*100) if mask.any() else None})
        rows.append({'condition':name,'notes':len(result['notes']),
            **score_notes(reference,result['notes']),**score_frames(reference,features),'pitch_core':core})
    # Save strict note/frame scores and core pitch errors as a synthetic sanity check.
    report={'recorded_date':'2026-10-21','seed':339,'sample_rate':sr,'reference_notes':reference,
        'conditions':rows,'limitation':'Pure sine sanity check, not vocal robustness evidence or a generalization benchmark; noise SNR is over the full waveform including rests.'}
    p=ROOT/'experiments/vishwa/results/synthetic_pitch_check.json'
    p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
