"""Run a training-only segmentation ablation and frozen validation pilot."""
import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from melody_understanding import analyze_vocal
from melody_understanding.annotations import read_midi_notes
from melody_understanding.evaluation import notes_with_smoothing, score_notes, score_frames


def main():
    """Freeze IDs/configuration, evaluate, and persist per-recording evidence."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'experiments/vishwa/configs/baseline_pilot.json')
    p.add_argument('--manifest', type=Path, default=ROOT/'datasets/humtrans/derived/research/manifest.jsonl')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    # Fail rather than silently ignore settings fixed by the current metric protocol.
    expected = {'note_min_duration_s': .06, 'onset_tolerance_s': .05,
                'pitch_tolerance_cents': 50, 'offset_ratio': .2, 'offset_min_tolerance_s': .05}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Metric settings must match protocol v1; version changes explicitly')
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    rng = random.Random(config['seed'])
    groups = defaultdict(list)
    for row in rows:
        if not row['flags'] and row['split'] in ['TRAIN', 'VALID']:
            groups[(row['split'], row['singer'])].append(row['key'])
    cohorts = {'TRAIN': [], 'VALID': []}
    for split in cohorts:
        count = config['train_per_singer'] if split == 'TRAIN' else config['valid_per_singer']
        for (group_split, singer), keys in sorted(groups.items()):
            if group_split != split:
                continue
            cohorts[split].extend(rng.sample(sorted(keys), count))
    if not all(cohorts.values()):
        raise ValueError('Both TRAIN and VALID cohorts are required')
    # Every result directory is immutable; reruns need a new output directory.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'cohort.json').write_text(json.dumps({'config':config,'ids':cohorts},indent=2)+'\n')
    cache = ROOT/'datasets/humtrans/derived/research/pilot_features'
    cache.mkdir(parents=True, exist_ok=True)
    data = ROOT/'datasets/humtrans/downloads'
    records, selected = [], None
    for split, keys in cohorts.items():
        widths = config['smoothing_widths'] if split == 'TRAIN' else sorted({1, selected})
        for index, key in enumerate(keys):
            result, features = analyze_vocal(data/'wav_data_sync_with_midi'/f'{key}.wav')
            reference, _ = read_midi_notes(data/'midi_data'/f'{key}.mid')
            # Cache only frame features needed for review; no raw recordings copied.
            np.savez_compressed(cache/f'{key}.npz', times_s=features['times_s'],
                f0_hz=features['f0_hz'], voiced=features['voiced'],
                voiced_probability=features['voiced_probability'])
            frame_metrics = score_frames(reference, features)
            for width in widths:
                predicted = notes_with_smoothing(features, result['source_duration_s'], width)
                records.append({'key':key,'split':split,'smoothing_width':width,
                    'reference_notes':len(reference),'predicted_notes':len(predicted),
                    'duration_s':result['source_duration_s'],
                    'inference_seconds':result['inference_seconds'],
                    **frame_metrics, **score_notes(reference, predicted)})
            print(f'{split} {index+1}/{len(keys)} {key}', flush=True)
            (args.output/'per_recording.json').write_text(json.dumps(records,indent=2)+'\n')
        if split == 'TRAIN':
            means = {w: np.mean([r['note_f1'] for r in records if r['split']=='TRAIN' and r['smoothing_width']==w]) for w in widths}
            selected = max(widths, key=lambda w:(means[w], -w))
            (args.output/'frozen_selection.json').write_text(json.dumps({'selected_width':selected,'train_macro_note_f1':means},indent=2)+'\n')
            print('Frozen width before VALID:',selected,flush=True)
    metrics = ['note_precision','note_recall','note_f1','note_offset_f1',
        'midi_frame_pitch_agreement_50c','midi_frame_overall_agreement_50c','midi_voicing_recall']
    aggregate = []
    for split in cohorts:
        for width in sorted({r['smoothing_width'] for r in records if r['split']==split}):
            sample=[r for r in records if r['split']==split and r['smoothing_width']==width]
            aggregate.append({'split':split,'width':width,'n':len(sample),
                **{m:float(np.mean([r[m] for r in sample])) for m in metrics},
                'reference_notes':sum(r['reference_notes'] for r in sample),
                'predicted_notes':sum(r['predicted_notes'] for r in sample)})
    source_paths = sorted((ROOT/'src/melody_understanding').glob('*.py'))+[Path(__file__)]
    report={'recorded_date':'2026-10-21','selected_width':selected,'aggregation':'macro mean across recordings',
        'config':config,'aggregate':aggregate,'versions':{p:version(p) for p in ['librosa','numpy','scipy','mir_eval']},
        'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'implementation_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'limitations':['Small pilot; not full-corpus or final TEST performance',
            'MIDI-held note pitches are not continuous vocal F0 labels',
            'Singer identities overlap official splits', 'Note segmentation cannot separate all repeated same-pitch notes']}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['aggregate'],indent=2))

if __name__=='__main__':main()
