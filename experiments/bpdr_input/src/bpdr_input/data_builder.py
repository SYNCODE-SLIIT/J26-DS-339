"""Prepare typed Vocadito/Jamendo crops and a pending review queue, never training pairs."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
import time
import uuid
import numpy as np
import soundfile as sf
from .analysis import extract
from .data_sources import vocadito_sources, jamendo_sources, vocal_windows, fingerprint, probable_duplicate
from .dataset import DataPolicy, reference_mask, REVIEW_FIELDS
from .interventions import AuditPolicy, independent_pitch, select_support
from .pipeline import process, checksum, validate_package
from .model_pins import GAME_COMMIT, GAME_SHA256, SEPARATOR_SHA256

SOURCE_MODULES = ('data_builder', 'data_sources', 'model_pins', 'analysis', 'dataset', 'interventions', 'pipeline', 'contracts')


@dataclass(frozen=True)
class ScreenPolicy:
    version: str = 'preparation-screen-exploratory-0.1'
    min_notes: int = 2
    min_voiced_fraction: float = .25
    min_reliable_share: float = .35
    min_reliable_frames: int = 10
    max_predicted_overlap_fraction: float = .05
    boundary_guard_seconds: float = .10
    tracker_agreement_cents: float = 50.
    note_agreement_cents: float = 100.


def checked_hash(path, expected):
    digest = checksum(path)
    if digest != expected:
        raise ValueError('Pinned file checksum mismatch: '+str(path))
    return digest


def verified_models(assets):
    assets = Path(assets).resolve()
    game = assets/'tools/GAME'
    commit = subprocess.run(['git', '-C', str(game), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(game), 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True, check=True).stdout.strip()
    if commit != GAME_COMMIT or dirty:
        raise ValueError('GAME source is not the unchanged pinned commit')
    python = assets/'.venv-game'/('Scripts/python.exe' if __import__('os').name == 'nt' else 'bin/python')
    if not python.is_file():
        raise FileNotFoundError('GAME environment missing')
    model = assets/'models/GAME-1.0-medium/model.pt'
    separator = assets/'models/mlx-audio-separator/model_bs_roformer_ep_317_sdr_12.9755.ckpt'
    return dict(game_commit=commit, game_model_path=str(model), game_sha256=checked_hash(model, GAME_SHA256),
                separator_model_path=str(separator), separator_sha256=checked_hash(separator, SEPARATOR_SHA256),
                game_python=str(python), game_source=str(game), game_seed='not exposed by pinned extraction CLI')


def decode_batch(inputs, output, models):
    output.mkdir(parents=True)
    command = [models['game_python'], 'infer.py', 'extract', str(inputs), '-m', models['game_model_path'],
               '--glob', '*.wav', '--output-formats', 'mid,csv', '--pitch-format', 'number', '--output-dir', str(output)]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=models['game_source'], capture_output=True, text=True, timeout=1800)
    (output/'stdout.log').write_text(completed.stdout, encoding='utf-8')
    (output/'stderr.log').write_text(completed.stderr, encoding='utf-8')
    if completed.returncode:
        raise RuntimeError('GAME failed; inspect saved decode logs')
    return dict(command=command, elapsed_seconds=time.perf_counter()-started, seed=None)


def parse_notes(path, duration):
    if not Path(path).is_file():
        return [], ['GAME emitted no note CSV']
    notes, warnings = [], []
    with Path(path).open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            onset, offset, pitch = (float(row[key]) for key in ('onset', 'offset', 'pitch'))
            if not all(np.isfinite(x) for x in (onset, offset, pitch)) or not 0 <= onset < offset or not 0 <= pitch <= 127:
                raise ValueError('Invalid GAME note')
            if onset >= duration or offset > duration+.05:
                raise ValueError('GAME note outside accepted audio')
            if offset > duration:
                warnings.append('Output edge clipped to accepted duration; raw CSV retained')
            notes.append(dict(onset_sec=onset, offset_sec=min(offset, duration), model_midi_pitch=pitch))
    if any(a['onset_sec'] > b['onset_sec'] for a, b in zip(notes, notes[1:])):
        raise ValueError('Unordered GAME notes')
    return notes, warnings


def note_mask(times, notes, guard):
    count, pitch = np.zeros(len(times), dtype=int), np.zeros(len(times))
    overlap = np.zeros(len(times), dtype=int)
    for note in notes:
        active = (times >= note['onset_sec']) & (times < note['offset_sec'])
        overlap[active] += 1
        interior = (times >= note['onset_sec']+guard) & (times <= note['offset_sec']-guard)
        count[interior] += 1
        pitch[interior] = 440.*2**((note['model_midi_pitch']-69)/12)
    return count == 1, pitch, float(np.mean(overlap > 1))


def screen(features, yin, notes, policy=ScreenPolicy(), human=None):
    times, pesto, voiced = features['times_seconds'], features['f0_hz'], features['voiced_mask']
    if len(yin) != len(times):
        raise ValueError('Screen time grids differ')
    inside, pitch, overlap = note_mask(times, notes, policy.boundary_guard_seconds)
    reliable = inside & voiced & np.isfinite(yin) & (yin > 0) & np.isfinite(pesto) & (pesto > 0)
    valid = reliable.copy()
    reliable[valid] &= (abs(1200*np.log2(yin[valid]/pesto[valid])) <= policy.tracker_agreement_cents)
    valid = reliable.copy()
    reliable[valid] &= abs(1200*np.log2(pesto[valid]/pitch[valid])) <= policy.note_agreement_cents
    if human is not None:
        reliable &= human
    share = float(reliable.sum()/inside.sum()) if inside.any() else 0.
    reasons = []
    if len(notes) < policy.min_notes:
        reasons.append('TOO_FEW_PREDICTED_NOTES')
    if float(voiced.mean()) < policy.min_voiced_fraction:
        reasons.append('LOW_VOICING')
    if share < policy.min_reliable_share or reliable.sum() < policy.min_reliable_frames:
        reasons.append('INSUFFICIENT_RELIABLE_PITCH')
    if overlap > policy.max_predicted_overlap_fraction:
        reasons.append('OVERLAPPING_PREDICTED_NOTES')
    try:
        support = list(select_support(times, reliable, AuditPolicy()))
    except ValueError:
        support = None
    return dict(status='review_candidate' if not reasons else 'withheld', reasons=reasons,
                note_count=len(notes), voiced_fraction=float(voiced.mean()), reliable_share_inside_notes=share,
                reliable_frames=int(reliable.sum()), predicted_overlap_fraction=overlap,
                v03_potential_support_seconds=support,
                manual_checks_required=['single_voice', 'accompaniment_leakage', 'musical_acceptability', 'ambiguous_transitions'],
                calibration='provisional engineering screen; no suitability-accuracy claim'), reliable


def repeat_diagnostic(first, second):
    unmatched = list(range(len(second)))
    matched = []
    for note in first:
        nearby = [index for index in unmatched if abs(second[index]['onset_sec']-note['onset_sec']) <= .05]
        if nearby:
            best = min(nearby, key=lambda index: abs(second[index]['onset_sec']-note['onset_sec']))
            unmatched.remove(best)
            matched.append(abs(100*(note['model_midi_pitch']-second[best]['model_midi_pitch'])))
    total = len(first)+len(second)
    return dict(onset_agreement_diagnostic=2*len(matched)/total if total else None,
                median_matched_pitch_difference_cents=float(np.median(matched)) if matched else None,
                meaning='Greedy 50ms repeated-decode diagnostic; not human-reference accuracy or calibrated confidence')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def code_hashes():
    folder = Path(__file__).parent
    return {name: checksum(folder/(name+'.py')) for name in SOURCE_MODULES}


def build(component, data_run, vocadito_singers=3, jamendo_artists=3, windows_per_song=2,
          processor=process, decoder=decode_batch, model_verifier=verified_models, jamendo_track_ids=None):
    component = Path(component).resolve()
    if min(vocadito_singers, jamendo_artists) < 0 or windows_per_song < 1:
        raise ValueError('Invalid sample limits')
    sources, voc_provenance = vocadito_sources(data_run, vocadito_singers)
    jam, exclusions, jam_provenance = jamendo_sources(component/'data/mtg-jamendo', component/'data/jamendo-vocal-stems', jamendo_artists, jamendo_track_ids)
    sources += jam
    models = model_verifier(component/'checkpoints/melody-extraction')
    root = component/'runs/preparation'/uuid.uuid4().hex
    root.mkdir(parents=True)
    hashes = code_hashes()
    source_snapshot = root/'source_code_at_run'
    source_snapshot.mkdir()
    for name in SOURCE_MODULES:
        shutil.copyfile(Path(__file__).parent/(name+'.py'), source_snapshot/(name+'.py'))
    manifest = dict(schema_version='bpdr-preparation-0.1', run_status='running', data_status='exploratory_not_research',
                    policy=asdict(ScreenPolicy()), sources=sources, records=[], exclusions=exclusions,
                    source_selection=dict(vocadito=voc_provenance, jamendo=jam_provenance), models=models,
                    source_code_sha256=hashes, duplicate_checks=[],
                    corpus_wide_duplicate_audit='not performed; checks cover selected training material only',
                    reference_semantics='human annotations and model estimates are distinct; neither is an intended-correction target',
                    training_ready_examples=0, examples=[], untouched_partitions=['development', 'pp1_evaluation', 'final_holdout'])
    inputs = root/'game_inputs'
    inputs.mkdir()
    def save():
        write_json(root/'manifest.json', manifest)
    save()
    fingerprints = []
    tracked_inputs = []
    def track(path):
        path = Path(path).resolve()
        previous = next((item for item in tracked_inputs if item['path'] == str(path)), None)
        digest = checksum(path)
        if previous and previous['sha256'] != digest:
            raise ValueError('Source changed while registering preparation inputs')
        if not previous:
            tracked_inputs.append(dict(path=str(path), sha256=digest))
    try:
        for path in (voc_provenance['manifest_path'], voc_provenance['review_path'], jam_provenance['metadata_path'], jam_provenance['licenses_path']):
            track(path)
        for source in sources:
            audio_path = Path(source['source_audio'])
            track(audio_path)
            samples, rate = sf.read(audio_path, dtype='float32', always_2d=True)
            signature = dict(source_id=source['source_id'], dataset=source['dataset'], group_id=source['group_id'],
                             sha256=checksum(audio_path), duration=len(samples)/rate, fingerprint=fingerprint(samples, rate))
            duplicate = next(((old, probable_duplicate(old, signature)) for old in fingerprints if probable_duplicate(old, signature)), None)
            fingerprints.append(signature)
            if duplicate:
                manifest['duplicate_checks'].append(dict(source=source['source_id'], prior=duplicate[0]['source_id'], kind=duplicate[1]))
                manifest['exclusions'].append(dict(dataset=source['dataset'], source_id=source['source_id'], reason='Potential duplicate; held for review'))
                continue
            source['source_sha256'] = signature['sha256']
            source['source_rate_hz'], source['source_samples'] = rate, len(samples)
            if source['dataset'] == 'jamendo':
                for path in (source['original_mp3'], source['full_note_path'], source['legacy_history_path']):
                    track(path)
                if source.get('receipt_path'):
                    track(source['receipt_path'])
                full = json.loads(Path(source['full_note_path']).read_text(encoding='utf-8'))
                if str(full['track_id']) != source['source_id'] or full['method'] != 'game_vocal':
                    raise ValueError('Full-song prediction identity mismatch')
                starts = vocal_windows(full['notes'], len(samples)/rate, count=windows_per_song)
                if not starts:
                    manifest['exclusions'].append(dict(dataset='jamendo', source_id=source['source_id'], reason='No window with five predicted vocal seconds'))
            else:
                starts = [0.]
                for path in source['reference_paths'].values():
                    track(path)
            for start in starts:
                begin = int(round(start*rate))
                end = min(len(samples), begin+20*rate) if source['dataset'] == 'jamendo' else len(samples)
                rid = source['dataset']+'_'+source.get('phrase_id', source['source_id']+'_s'+str(begin))
                folder = root/'records'/rid
                folder.mkdir(parents=True)
                if source['dataset'] == 'jamendo':
                    raw = folder/'crop.wav'
                    sf.write(raw, samples[begin:end], rate, subtype='FLOAT')
                    mapping = dict(original_offset_seconds=begin/rate, start_sample=begin, end_sample=end,
                                   original_sample_rate_hz=rate, origin='full_song_stem', original_mp3=source['original_mp3'])
                else:
                    raw, mapping = audio_path, source['source_map']
                result, preparation = processor(raw, folder/'technical', component/'checkpoints')
                record = dict(record_id=rid, phrase_id=rid, dataset=source['dataset'], source_id=source['source_id'],
                              group_id=source['group_id'], group_kind=source['group_kind'], split='train', source_map=mapping,
                              reference_kind=source['reference_kind'], license=source['license'], raw_crop_sha256=checksum(raw),
                              parent_source_path=str(audio_path.resolve()), parent_source_sha256=source['source_sha256'],
                              separation_provenance=source.get('separation_provenance', 'solo_source_no_separation'),
                              technical_result=result.to_dict(), diagnostics_path=str(preparation/'diagnostics.json'),
                              training_eligible=False, intervention_audit=None, reviews_status='pending', reference_files={})
                if source.get('receipt_path'):
                    copied_receipt = folder/'separator_receipt.json'
                    shutil.copyfile(source['receipt_path'], copied_receipt)
                    record['separator_receipt'] = dict(path=str(copied_receipt), sha256=checksum(copied_receipt))
                manifest['records'].append(record)
                if result.accepted_audio is None:
                    record['screen'] = dict(status='withheld', reasons=list(result.reason_codes))
                    save()
                    continue
                accepted = validate_package(result)
                record.update(sha256=accepted.checksum, accepted_audio_path=accepted.path,
                              duration_seconds=accepted.duration_seconds, features_path=str(preparation/'features.npz'))
                if abs(accepted.duration_seconds-(end-begin)/rate) > 1/accepted.sample_rate_hz+1e-9:
                    raise ValueError('Accepted crop duration mapping failed')
                shutil.copyfile(accepted.path, inputs/(rid+'.wav'))
                for name, path in source['reference_paths'].items():
                    copied = folder/(name+Path(path).suffix)
                    shutil.copyfile(path, copied)
                    record['reference_files'][name] = dict(path=str(copied), sha256=checksum(copied))
                save()
        ready = [r for r in manifest['records'] if 'accepted_audio_path' in r]
        manifest['decode_runs'] = []
        if ready:
            for index in range(2):
                print('GAME accepted-crop decode', index+1, 'of 2 for', len(ready), 'crops', flush=True)
                manifest['decode_runs'].append(decoder(inputs, root/'game'/f'decode_{index}', models))
                save()
        for record in ready:
            folder = root/'records'/record['record_id']
            notes, warnings = parse_notes(root/'game/decode_0'/(record['record_id']+'.csv'), record['duration_seconds'])
            repeated, repeat_warnings = parse_notes(root/'game/decode_1'/(record['record_id']+'.csv'), record['duration_seconds'])
            note_path = folder/'game_notes.json'
            write_json(note_path, dict(reference_kind='model_estimate', time_origin='accepted_crop_start', pitch_field='model_midi_pitch',
                                      pitch_shift_semitones=0, model=models, notes=notes, warnings=warnings))
            record['model_notes'] = dict(path=str(note_path), sha256=checksum(note_path))
            record['repeat_decode'] = repeat_diagnostic(notes, repeated)
            record['decode_warnings'] = warnings+repeat_warnings
            with np.load(record['features_path']) as archive:
                features = {key: archive[key] for key in archive.files}
            audio, rate = sf.read(record['accepted_audio_path'], dtype='float32')
            human = None
            if record['reference_kind'] == 'human_annotation':
                f0 = np.loadtxt(record['reference_files']['f0']['path'], delimiter=',', ndmin=2)
                annotated = json.loads(Path(record['reference_files']['notes']['path']).read_text())
                human, _ = reference_mask(features['times_seconds'], f0, annotated, DataPolicy(**voc_provenance['policy']))
            record['screen'], reliable = screen(features, independent_pitch(audio, rate), notes, human=human)
            mask_path = folder/'screen_masks_NOT_MODEL_INPUT.npz'
            np.savez_compressed(mask_path, times_seconds=features['times_seconds'], reliable_mask=reliable)
            record['screen_masks'] = dict(path=str(mask_path), sha256=checksum(mask_path))
            save()
        queue = [r for r in ready if r['screen']['status'] == 'review_candidate']
        with (root/'reviews.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for record in queue:
                for _ in range(2):
                    writer.writerow(dict(phrase_id=record['record_id'], crop_sha256=record['sha256'], decision='pending', approved_intervals_json='[]'))
        for item in tracked_inputs:
            if checksum(item['path']) != item['sha256']:
                raise ValueError('Source changed during preparation')
        if code_hashes() != hashes:
            raise ValueError('Builder code changed during preparation')
        decode_artifacts = [dict(path=str(path), sha256=checksum(path))
                            for directory in (inputs, root/'game') if directory.exists()
                            for path in directory.rglob('*') if path.is_file()]
        manifest.update(run_status='complete', tracked_source_files=tracked_inputs, decode_artifacts=decode_artifacts, inputs_unchanged=True)
        save()
        summary = dict(data_status='exploratory_not_research', manifest_sha256=checksum(root/'manifest.json'), run_status='complete',
                       records=len(manifest['records']), technical_statuses=dict(Counter(r['technical_result']['status'] for r in manifest['records'])),
                       datasets={name: dict(crops=sum(r['dataset'] == name for r in manifest['records']),
                           groups=len({r['group_id'] for r in manifest['records'] if r['dataset'] == name}),
                           review_candidates=sum(r['dataset'] == name for r in queue)) for name in ('vocadito', 'jamendo')},
                       review_rows=2*len(queue), training_ready_examples=0, examples=[], excluded_sources=len(manifest['exclusions']),
                       limitations=['Plumbing sample from retained training material; not representative J3',
                                    'No calibrated single-voice/leakage detector; manual checks mandatory',
                                    'Legacy stems lack per-track separator receipts', 'No reviewed or audited training pairs'])
        write_json(root/'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
        print('PREPARATION_RUN:', root, flush=True)
        return root, manifest
    except Exception as exc:
        manifest.update(run_status='failed', failure=dict(type=type(exc).__name__, reason=str(exc)))
        save()
        raise


def validate_run(root, model_verifier=verified_models):
    root = Path(root).resolve()
    component = root.parents[2]
    manifest = json.loads((root/'manifest.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    if manifest['run_status'] != 'complete' or checksum(root/'manifest.json') != summary['manifest_sha256']:
        raise ValueError('Incomplete or altered preparation manifest')
    models = model_verifier(component/'checkpoints/melody-extraction')
    if any(models[key] != manifest['models'][key] for key in ('game_sha256', 'game_commit', 'separator_sha256')):
        raise ValueError('Extractor changed')
    for item in [*manifest['tracked_source_files'], *manifest['decode_artifacts']]:
        checked_hash(item['path'], item['sha256'])
    for record in manifest['records']:
        if record['training_eligible'] or record['split'] != 'train':
            raise ValueError('Unsafe preparation eligibility/partition')
        accepted_status = record['technical_result']['status'] in ('ACCEPT_UNCHANGED', 'ACCEPT_PREPARED')
        if accepted_status != ('accepted_audio_path' in record):
            raise ValueError('Rejected record exposes accepted audio')
        if 'accepted_audio_path' not in record:
            continue
        checked_hash(record['accepted_audio_path'], record['sha256'])
        info = sf.info(record['accepted_audio_path'])
        audio_metadata = record['technical_result']['accepted_audio']
        if info.channels != 1 or info.samplerate != audio_metadata['sample_rate_hz'] or info.frames != audio_metadata['sample_count'] or info.subtype != 'PCM_24':
            raise ValueError('Invalid accepted format')
        for item in [record['model_notes'], record['screen_masks'], *record['reference_files'].values(),
                     *([record['separator_receipt']] if 'separator_receipt' in record else [])]:
            checked_hash(item['path'], item['sha256'])
    return dict(records=len(manifest['records']), source_hashes=len(manifest['tracked_source_files']), training_ready_examples=0,
                verified=True, reviews_remain_independent_of_training_eligibility=True)


def main():
    component = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build_parser = commands.add_parser('build')
    build_parser.add_argument('--data-run', type=Path)
    build_parser.add_argument('--vocadito-singers', type=int, default=3)
    build_parser.add_argument('--jamendo-artists', type=int, default=3)
    build_parser.add_argument('--jamendo-tracks', help='Comma-separated frozen track IDs; excludes older stems from this sample')
    verify_parser = commands.add_parser('verify')
    verify_parser.add_argument('run', type=Path)
    args = parser.parse_args()
    if args.command == 'verify':
        print(json.dumps(validate_run(args.run), indent=2))
    else:
        data_run = args.data_run or Path(json.loads((component/'runs/phase2/latest.json').read_text())['path'])
        tracks = [value.strip() for value in args.jamendo_tracks.split(',')] if args.jamendo_tracks else None
        build(component, data_run, args.vocadito_singers, args.jamendo_artists, jamendo_track_ids=tracks)


if __name__ == '__main__':
    main()
