"""Select training-artist songs and run the existing pinned separator with a per-track receipt."""
from bpdr_input import COMPONENT_ROOT
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import uuid
import soundfile as sf
from bpdr_input.preparation.sources.data_sources import ALLOWED_LICENSES, audio_licenses, inside
from bpdr_input.preparation.sources.dataset import DataPolicy, singer_splits
from bpdr_input.preparation.sources.data_builder import verified_models
from bpdr_input.audio.pipeline import checksum


def eligible(dataset, minimum_seconds=60.):
    dataset = Path(dataset)
    with (dataset/'melody-extraction-pipeline/manifest.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    licenses = audio_licenses(dataset/'audio_licenses.txt')
    splits = singer_splits([row['artist_id'] for row in rows], DataPolicy().split_seed)
    keep = [dict(row, license=licenses[row['track_id']]) for row in rows
            if splits[row['artist_id']] == 'train' and licenses.get(row['track_id'], {}).get('kind') in ALLOWED_LICENSES
            and 'instrument---voice' in row['tags'].split(';')
            and float(row['duration_annotation_sec']) >= minimum_seconds]
    seed = DataPolicy().split_seed
    return sorted(keep, key=lambda row: (hashlib.sha256(f'{seed}:{row["artist_id"]}'.encode()).hexdigest(),
                                        float(row['duration_annotation_sec']), int(row['track_id'])))


def sample(component, count=5):
    if count < 1:
        raise ValueError('Positive sample size required')
    component = Path(component)
    rows = eligible(component/'data/mtg-jamendo')
    chosen = []
    artists = {row['artist_id'] for row in rows
               if (component/'data/jamendo-vocal-stems'/(row['track_id']+'.wav')).exists()}
    for row in rows:
        if row['artist_id'] in artists:
            continue
        if (component/'data/jamendo-vocal-stems'/(row['track_id']+'.wav')).exists():
            continue
        if any((component/'data/mtg-jamendo/melody-extraction-pipeline/outputs/game_vocal'/(row['track_id']+suffix)).exists()
               for suffix in ('.json', '.csv', '.mid')):
            continue
        artists.add(row['artist_id'])
        chosen.append(row)
        if len(chosen) == count:
            break
    return chosen


def separate(component, track_id):
    component = Path(component).resolve()
    dataset = component/'data/mtg-jamendo'
    matching = [row for row in eligible(dataset) if row['track_id'] == str(track_id)]
    if len(matching) != 1:
        raise ValueError('Track is not one licensed voice-tagged training-artist song of at least 60 seconds')
    row = matching[0]
    original = inside(dataset, row['source_mp3'])
    stem = component/'data/jamendo-vocal-stems'/(str(track_id)+'.wav')
    receipt = stem.with_suffix('.receipt.json')
    if stem.exists() or receipt.exists():
        raise ValueError('Existing stem/receipt will not be overwritten')
    if any((dataset/'melody-extraction-pipeline/outputs/game_vocal'/(str(track_id)+suffix)).exists()
           for suffix in ('.json', '.csv', '.mid')):
        raise ValueError('Existing vocal outputs require explicit recovery; refusing cached runner selection')
    assets = component/'checkpoints/melody-extraction'
    models = verified_models(assets)
    source_hash = checksum(original)
    root = component/'runs/separation'/uuid.uuid4().hex
    root.mkdir(parents=True)
    python = assets/'.venv-xpu/Scripts/python.exe'
    if not python.is_file():
        raise FileNotFoundError('Pinned Intel GPU environment missing')
    command = [str(python), '-B', str(component/'extraction/run_local.py'), '--first', str(track_id), '--limit', '1', '--retry-failed',
               '--device', 'xpu', '--separator-overlap', '4', '--threads', '8', '--vocal-only', '--keep-stems']
    record = dict(run_status='running', track_id=str(track_id), artist_id=row['artist_id'], split='train', license=row['license'],
                  source_path=str(original), source_sha256=source_hash, models=models,
                  settings=dict(device='xpu', overlap=4, threads=8, full_song=True), command=command,
                  launcher_sha256=checksum(component/'extraction/run_local.py'),
                  timing_origin='nominal full-song origin; exact sample alignment not independently established')
    log = root/'run.log'
    (root/'record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=component, stdout=stream, stderr=subprocess.STDOUT)
    # The legacy runner can log individual failures without a nonzero process code.
    if result.returncode or not stem.is_file() or checksum(original) != source_hash:
        record.update(run_status='failed', reason='Separator failed, produced no stem, or source changed; inspect run.log')
        (root/'record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        raise RuntimeError(record['reason'])
    info = sf.info(stem)
    record.update(run_status='complete', stem_path=str(stem), stem_sha256=checksum(stem),
                  sample_rate_hz=info.samplerate, sample_count=info.frames, channels=info.channels,
                  extraction_log_path=str(log), extraction_log_sha256=checksum(log),
                  receipt_kind='contemporaneous_separator_run', musical_usability_verified=False)
    (root/'record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    receipt.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return receipt


def main():
    component = COMPONENT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('plan').add_argument('--count', type=int, default=5)
    commands.add_parser('separate').add_argument('track_id')
    args = parser.parse_args()
    if args.command == 'plan':
        print(json.dumps(sample(component, args.count), indent=2))
    else:
        print('RECEIPT:', separate(component, args.track_id))


if __name__ == '__main__':
    main()
