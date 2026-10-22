"""Rebuild verified v3 events with one outlier window excluded and four note slots.

Run from the repository root:
python -m experiments.rahul.src.rebuild_four_note_dataset --help
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil

from experiments.rahul import config
from experiments.rahul.src.build_final_dataset import TrackFiles, export_split
from experiments.rahul.src.create_sequences import sequence_windows
from experiments.rahul.src.encode_features import FeatureEncoder


def remove_outlier_window(rows, track_id, beat_index, length):
    windows = sequence_windows({track_id: rows}, length, length)
    matches = [w for w in windows if any(
        r['beat_index'] == beat_index for r in rows[w['start_row']:w['stop_row']])]
    if len(matches) != 1:
        raise ValueError('Outlier must identify exactly one source window')
    window = matches[0]
    if window['stop_row'] != len(rows):
        raise ValueError('Only final-window exclusion is supported; refusing to repack internal windows')
    return rows[:window['start_row']], rows[window['start_row']:window['stop_row']]


class RetainedTracks(TrackFiles):
    def __init__(self, root, ids, excluded_track, beat_index, length):
        super().__init__(root, ids)
        self.excluded_track, self.beat_index, self.length = excluded_track, beat_index, length

    def __getitem__(self, track):
        rows = super().__getitem__(track)
        if track == self.excluded_track:
            rows, _ = remove_outlier_window(rows, track, self.beat_index, self.length)
        return rows


def run(previous_name='final_all_tracks_quality_v3', run_name='final_all_tracks_quality_v4_m4'):
    csv.field_size_limit(64 * 1024 * 1024)
    base = config.COMPONENT_DIR / 'outputs/model_preprocessing'
    for name in (previous_name, run_name):
        if not name.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Use simple snapshot names')
    previous, output = base / previous_name, base / run_name
    if output.exists():
        raise FileExistsError('Use a new run name; existing snapshots are protected')
    old_config = json.loads((previous / 'preprocessing_config.json').read_text())
    old_ready = json.loads((previous / 'FINAL_DATASET_READY.json').read_text())
    for name, expected in old_ready['artifacts'].items():
        if hashlib.sha256((previous / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Previous artifact changed: {name}')
    validation = json.loads((previous / 'preprocessing_validation_report.json').read_text())
    for relative, expected in validation['source_sha256'].items():
        if hashlib.sha256((config.REPOSITORY_ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Original extraction changed: {relative}')
    split_manifest = json.loads((previous / 'split_manifest.json').read_text())
    ids = [t for group in split_manifest['splits'].values() for t in group]
    source = config.REPOSITORY_ROOT / old_config['phase1_output_root']
    original = TrackFiles(source, ids)
    track, beat, length = '572381', 667, old_config['sequence_length']
    kept, removed = remove_outlier_window(original[track], track, beat, length)
    outlier = next(r for r in removed if r['beat_index'] == beat)
    exclusion = dict(track_id=track, outlier_beat_index=beat,
        removed_beat_indices=[r['beat_index'] for r in removed],
        removed_real_positions=len(removed), removed_padding_positions=length-len(removed),
        sequence_length=length, original_start_row=len(kept),
        outlier_start_sec=outlier['start_sec'], outlier_end_sec=outlier['end_sec'],
        outlier_duration_sec=outlier['event_duration'], outlier_note_count=len(outlier['melody_midi']),
        removed_note_occurrences=sum(len(r['melody_midi']) for r in removed),
        reason='Final detected beat extended to audio duration: exclude the entire containing sequence without repacking.')
    tracks = RetainedTracks(source, ids, track, beat, length)
    print('Excluding entire source window:', exclusion, flush=True)
    encoder = FeatureEncoder.fit(tracks, split_manifest['splits']['train'], max_notes=4, truncate=True)
    output.mkdir()
    encoder.save(output)
    reports = {t: dict(statistics=dict(final_events=len(tracks[t]))) for t in ids}
    stats, checks, manifest = {}, {}, []
    for split, assigned in split_manifest['splits'].items():
        stat, entries, checked = export_split(split, assigned, tracks, encoder, length, output, reports)
        stats[split], checks[split] = stat, checked
        manifest.extend(entries)
    with (output / 'sequence_manifest.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    for name in ('split_manifest.json', 'discovery_manifest.json', 'track_build_results.json', 'quality_rebuild_validation.json'):
        shutil.copyfile(previous / name, output / name)
    settings = {**old_config, 'max_notes_per_beat': 4,
        'note_capacity_policy': 'Exactly four slots. Keep first four notes in validated source order; pad shorter lists. Lossy truncation, not four simultaneous voices.',
        'truncate_notes': True, 'previous_snapshot': previous_name, 'excluded_sequences': [exclusion],
        'observed_training_max_notes': stats['train']['max_observed_notes'],
        'implementation_sha256': {name: hashlib.sha256((config.COMPONENT_DIR / 'src' / name).read_bytes()).hexdigest()
            for name in ('rebuild_four_note_dataset.py', 'build_final_dataset.py', 'build_synthetic_dataset.py', 'encode_features.py', 'transformer_dataset.py')}}
    result = dict(all_passed=True, splits=checks, source_sha256=validation['source_sha256'],
        source_extractions_unchanged=True, group_split_isolation=True, train_only_fit=True,
        note_truncation_explicit=True, excluded_sequences=[exclusion], excluded_before_vocabulary_fit=True,
        notes_truncated=sum(s['notes_truncated'] for s in stats.values()))

    def dump(name, value):
        (output / name).write_text(json.dumps(value, indent=2), encoding='utf-8')

    dump('preprocessing_config.json', settings)
    dump('preprocessing_statistics.json', stats)
    dump('preprocessing_validation_report.json', result)
    dump('outlier_exclusion.json', exclusion)
    dump('FINAL_DATASET_READY.json', dict(status='ready', validated_tracks=len(ids), splits=stats,
        artifacts={name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in
            ('train_sequences.npz', 'validation_sequences.npz', 'test_sequences.npz',
             'feature_vocabularies.json', 'target_vocabulary.json', 'split_manifest.json')}))
    (output / 'dataset_build_summary.md').write_text(
        '# Four-note dataset\n\nRemoved the complete final sequence of track 572381 containing beat 667. '
        'Source files and original source beat context are retained; no later windows are repacked. '
        'Vocabularies were fitted on retained training events only.\n\n'
        'Each beat has four note slots; first four source-ordered notes are retained, later notes truncated. '
        'Silent and short melodies retain padding and masks.\n\n' +
        json.dumps(dict(exclusion=exclusion, splits=stats), indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), splits=stats, notes_truncated=result['notes_truncated']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='final_all_tracks_quality_v4_m4')
    parser.add_argument('--previous-name', default='final_all_tracks_quality_v3')
    args = parser.parse_args()
    run(args.previous_name, args.run_name)
