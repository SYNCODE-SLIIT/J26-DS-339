"""Prepare a TRAIN/VALID feasibility dataset with explicit weak-label policy."""
import hashlib,json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from melody_understanding.audio import prepare_audio
from melody_understanding.annotations import read_midi_notes
from melody_understanding.targets import make_score_features


# Preparing Weak Score Features for the Fixed Pilot Cohort
def main():
    """Build source-derived tensors and normalization from TRAIN frames only."""
    # Load the preserved cohort and official split assignments.
    cohort=json.loads((ROOT/'experiments/vishwa/results/pilot_001/cohort.json').read_text())['ids']
    splits=json.loads((ROOT/'datasets/humtrans/downloads/train_valid_test_keys.json').read_text())
    folder=ROOT/'datasets/humtrans/derived/research/weak_score_features'
    folder.mkdir(parents=True,exist_ok=True)
    rows=[];sums=np.zeros(80);squares=np.zeros(80);count=0
    # Enforce TRAIN/VALID membership before building each recording’s feature tensor.
    for split,keys in cohort.items():
        if split not in ['TRAIN','VALID'] or not set(keys)<=set(splits[split]):raise ValueError('Split membership violation')
        for key in keys:
            wav=ROOT/'datasets/humtrans/downloads/wav_data_sync_with_midi'/f'{key}.wav'
            midi=ROOT/'datasets/humtrans/downloads/midi_data'/f'{key}.mid'
            audio=prepare_audio(wav);notes,_=read_midi_notes(midi)
            features=make_score_features(audio,notes,.1)
            path=folder/f'{key}.npz';np.savez_compressed(path,**features)
            # Accumulate normalization statistics from TRAIN frames only.
            if split=='TRAIN':
                values=features['mel_db'].astype(np.float64)
                sums+=values.sum(axis=1);squares+=(values**2).sum(axis=1);count+=values.shape[1]
            # Record supervision counts and hashes linking tensors to their sources.
            rows.append({'key':key,'split':split,'frames':len(features['times_s']),
                'supervised_frames':int(features['supervision_mask'].sum()),
                'path':str(path.relative_to(ROOT)),
                'wav_sha256':hashlib.sha256(wav.read_bytes()).hexdigest(),
                'midi_sha256':hashlib.sha256(midi.read_bytes()).hexdigest(),
                'tensor_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    # Compute per-band mean and standard deviation with a small variance floor.
    mean=sums/count;std=np.sqrt(np.maximum(squares/count-mean**2,1e-8))
    np.savez(folder/'train_normalization.npz',mean=mean.astype(np.float32),std=std.astype(np.float32))
    # Save the target policy, processing settings and dataset provenance.
    report={'recorded_date':'2026-10-21','target_policy':'weak-score-chroma-v1',
        'target_meaning':'original MIDI intended-score pitch class, not verified performed F0',
        'classes':{'0':'rest','1..12':'MIDI pitch modulo 12 plus 1'},
        'boundary_margin_s':.1,'mel':{'sample_rate':22050,'n_fft':4096,'hop_length':256,'n_mels':80},
        'normalization':'per-band mean/std from all TRAIN frames only, no VALID/TEST fit',
        'split_counts':{split:len(keys) for split,keys in cohort.items()},
        'frames_by_split':{split:sum(r['frames'] for r in rows if r['split']==split) for split in cohort},
        'supervised_frames_by_split':{split:sum(r['supervised_frames'] for r in rows if r['split']==split) for split in cohort},
        'normalization_sha256':hashlib.sha256((folder/'train_normalization.npz').read_bytes()).hexdigest(),
        'recordings':rows,
        'limitations':['100 ms boundary mask is a fixed heuristic, not proven timing correction',
            'Octave invariant task does not establish accurate performed-pitch transcription',
            'Pilot cohorts are small and validation has been inspected previously']}
    out=ROOT/'experiments/vishwa/results/weak_score_dataset.json';out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='recordings'},indent=2))

if __name__=='__main__':main()
