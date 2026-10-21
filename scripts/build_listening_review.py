"""Build a local listening packet without changing or certifying annotations."""
from pathlib import Path
import sys,json,hashlib,html,os
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from melody_understanding.annotations import read_midi_notes


def synthesize_notes(notes,duration,shift,sr=22050):
    """Sine reference with short fades; shifts are listening hypotheses only."""
    y=np.zeros(int(np.ceil(duration*sr)),np.float32)
    for start,end,pitch in notes:
        a=max(0,round(start*sr));b=min(len(y),round(end*sr))
        if b<=a:continue
        t=np.arange(b-a)/sr;wave=.2*np.sin(2*np.pi*440*2**((pitch+shift-69)/12)*t)
        fade=min(round(.01*sr),len(wave)//2)
        if fade:wave[:fade]*=np.linspace(0,1,fade);wave[-fade:]*=np.linspace(1,0,fade)
        y[a:b]+=wave
    return np.clip(y,-.95,.95)


def main():
    """Generate all eight cases, bounded diagnostics and a persistent review form."""
    report=json.loads((ROOT/'experiments/vishwa/results/annotation_review.json').read_text())
    rows=[r for r in report['recordings'] if r['review_flag']=='shared_octave_discrepancy']
    splits=json.loads((ROOT/'datasets/humtrans/downloads/train_valid_test_keys.json').read_text())
    folder=ROOT/'datasets/humtrans/derived/research/listening_review';folder.mkdir(parents=True,exist_ok=True)
    output=[];sections=[]
    for row in rows:
        key=row['key']
        if key not in splits['TRAIN']:raise ValueError('Non-TRAIN review case')
        wav=ROOT/'datasets/humtrans/downloads/wav_data_sync_with_midi'/f'{key}.wav'
        midi=ROOT/'datasets/humtrans/downloads/midi_data'/f'{key}.mid'
        for path,field in [(wav,'wav_sha256'),(midi,'midi_sha256')]:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row[field]:raise ValueError('Source changed since R006')
        samples,sr=sf.read(wav);duration=len(samples)/sr;notes,_=read_midi_notes(midi)
        with np.load(ROOT/'datasets/humtrans/derived/research/pilot_features'/f'{key}.npz') as z:py={k:z[k] for k in z.files}
        with np.load(ROOT/'datasets/humtrans/derived/research/annotation_review'/f'{key}.npz') as z:pr={k:z[k] for k in z.files}
        times=py['times_s'];target=np.full(len(times),np.nan);interior=np.zeros(len(times),bool)
        for a,b,p in notes:
            target[(times>=a)&(times<b)]=p;interior|=(times>a+.1)&(times<b-.1)
        common=interior & py['voiced'] & pr['voiced'] & np.isfinite(py['f0_hz']) & np.isfinite(pr['f0_hz'])
        evidence={'key':key,'interior_common_frames':int(common.sum()),'candidate_agreement_50c':{}}
        for shift in [-24,-12,0,12]:
            evidence['candidate_agreement_50c'][str(shift)]={}
            for label,f in [('pyin',py),('praat',pr)]:
                pitches=69+12*np.log2(f['f0_hz'][common]/440)
                evidence['candidate_agreement_50c'][str(shift)][label]=float(np.mean(np.abs(pitches-target[common]-shift)<.5)) if common.any() else None
        output.append(evidence)
        fig,axes=plt.subplots(2,1,figsize=(10,5),layout='constrained')
        axes[0].plot(np.arange(len(samples))[::40]/sr,samples[::40],lw=.4);axes[0].set(title=key,ylabel='Amplitude')
        for label,f in [('pYIN',py),('Praat',pr)]:
            valid=f['voiced']&np.isfinite(f['f0_hz'])&(f['f0_hz']>0)
            axes[1].scatter(times[valid],69+12*np.log2(f['f0_hz'][valid]/440),s=2,label=label)
        for i,(a,b,p) in enumerate(notes):axes[1].plot([a,b],[p,p],color='black',label='Original MIDI' if i==0 else None)
        axes[1].set(xlabel='Original time (seconds)',ylabel='MIDI pitch');axes[1].legend()
        fig.savefig(folder/f'{key}.png',dpi=100);plt.close(fig)
        source=html.escape(os.path.relpath(wav,folder),quote=True)
        section=f'<section><h2>{key}</h2><p>Listen to the original first. Record your impression before comparing reference tones.</p><audio controls preload="none" src="{source}"></audio><img src="{key}.png" alt="Waveform and pitch evidence">'
        for shift in [0,-12,-24,12]:
            filename=f'{key}_reference_{shift:+d}.wav';sf.write(folder/filename,synthesize_notes(notes,duration,shift),22050)
            section+=f'<p>Reference tones: MIDI {shift:+d} semitones (hypothesis, not corrected truth)</p><audio controls preload="none" src="{filename}"></audio>'
        section+='<details><summary>Automated interior-frame diagnostics (inspect after listening)</summary><pre>'+html.escape(json.dumps(evidence,indent=2))+'</pre></details></section>'
        sections.append(section)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Eight TRAIN listening cases</title><style>body{font:17px system-ui;max-width:1000px;margin:40px auto;padding:20px;background:#fafafa;color:#202020}section{border-top:2px solid #ccc;padding:24px 0}img{width:100%;display:block}audio{width:100%}pre{white-space:pre-wrap}</style><h1>TRAIN annotation listening review</h1><p>R009 · All decisions remain pending human review. Original recordings are unchanged. Sine references preserve MIDI timing but do not reproduce vocal timbre. Algorithm agreement and octave-shift similarity do not establish correct annotations.</p><p>Record decisions in experiments/vishwa/results/listening_review_decisions.json using the protocol in docs/vishwa/listening_review.md. This page does not save responses.</p>'''+''.join(sections)+'</html>'
    (folder/'index.html').write_text(page)
    result={'recorded_date':'2026-10-21','scope':'8 flagged TRAIN recordings; no VALID or TEST','status':'automated_packet_complete_human_review_pending','boundary_exclusion_s':.1,'candidate_shifts_semitones':[-24,-12,0,12],'source_report_sha256':hashlib.sha256((ROOT/'experiments/vishwa/results/annotation_review.json').read_bytes()).hexdigest(),'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':output}
    (ROOT/'experiments/vishwa/results/listening_review_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    decisions=ROOT/'experiments/vishwa/results/listening_review_decisions.json'
    if not decisions.exists():
        decisions.write_text(json.dumps({'protocol':'R009-v1','cases':[{'key':r['key'],'status':'pending','reviewer':None,'reviewed_at':None,'listened_original':False,'listened_references':False,'outcome':None,'confidence':None,'evidence':None,'suggested_shift_semitones':None} for r in rows]},indent=2)+'\n')
    print(json.dumps(result,indent=2));print('Local review page:',folder/'index.html')
if __name__=='__main__':main()
