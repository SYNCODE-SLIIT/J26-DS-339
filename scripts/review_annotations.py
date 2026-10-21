"""Create a reproducible TRAIN-only annotation review manifest."""
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from melody_understanding.audio import prepare_audio
from melody_understanding.annotations import read_midi_notes
from melody_understanding.evaluation import score_frames
from melody_understanding.review import praat_pitch,compare_pitches


def main():
    """Compare maintained pitch implementations without changing annotations."""
    data=ROOT/'datasets/humtrans/downloads'
    cohort=json.loads((ROOT/'experiments/vishwa/results/pilot_001/cohort.json').read_text())['ids']['TRAIN']
    official=json.loads((data/'train_valid_test_keys.json').read_text())
    assert set(cohort)<=set(official['TRAIN'])
    derived=ROOT/'datasets/humtrans/derived/research/annotation_review'
    derived.mkdir(parents=True,exist_ok=True)
    rows=[]
    for key in cohort:
        with np.load(ROOT/'datasets/humtrans/derived/research/pilot_features'/f'{key}.npz') as f:
            first={k:f[k] for k in f.files}
        wav=data/'wav_data_sync_with_midi'/f'{key}.wav'
        midi=data/'midi_data'/f'{key}.mid'
        audio=prepare_audio(wav)
        reference,_=read_midi_notes(midi)
        second,supported=praat_pitch(audio,first['times_s'])
        # Report metrics on the same supported grid for both estimators.
        common_first={k:v[supported] for k,v in first.items()}
        common_second={k:v[supported] for k,v in second.items()}
        row={'key':key,'split':'TRAIN','review_status':'automated_evidence_only_human_review_pending',
             **compare_pitches(reference,first,second,supported),
             'pyin_common_grid':score_frames(reference,common_first),
             'praat_common_grid':score_frames(reference,common_second),
             'supported_frames':int(supported.sum()),'total_frames':len(supported),
             'wav_sha256':hashlib.sha256(wav.read_bytes()).hexdigest(),
             'midi_sha256':hashlib.sha256(midi.read_bytes()).hexdigest()}
        rows.append(row)
        np.savez_compressed(derived/f'{key}.npz',**second,supported=supported)
        print(key,row['review_flag'],flush=True)
    report={'recorded_date':'2026-10-21','n':len(rows),'split':'TRAIN',
        'config':{'method':'Praat autocorrelation','time_step_s':256/22050,'pitch_floor_hz':32.70319566257483,
            'pitch_ceiling_hz':2093.004522404789,'silence_threshold':.03,'voicing_threshold':.45,
            'mapping':'nearest native frame within half a hop; no interpolation',
            'flag_rule':'shared octave: >=80% estimator agreement within 50 cents and both median errors within .75 semitone of same nonzero octave'},
        'flags':dict(Counter(r['review_flag'] for r in rows)),
        'macro_estimator_agreement':float(np.mean([r['estimator_agreement_50c'] for r in rows if 'estimator_agreement_50c' in r])),
        'macro_pyin_midi_frame_agreement':float(np.mean([r['pyin_common_grid']['midi_frame_pitch_agreement_50c'] for r in rows])),
        'macro_praat_midi_frame_agreement':float(np.mean([r['praat_common_grid']['midi_frame_pitch_agreement_50c'] for r in rows])),
        'versions':{p:version(p) for p in ['praat-parselmouth','librosa','numpy']},
        'implementation_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/melody_understanding/review.py']},
        'recordings':rows,'limitation':'Agreement is not ground truth; this selected TRAIN cohort is not a random corpus-wide audit.'}
    out=ROOT/'experiments/vishwa/results/annotation_review.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='recordings'},indent=2))

if __name__=='__main__':main()
