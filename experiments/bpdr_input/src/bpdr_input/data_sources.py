"""Typed source adapters and fixed training-only selection for preparation."""
import csv
import json
from pathlib import Path
import re
import numpy as np
from .dataset import DataPolicy, singer_splits
from .pipeline import checksum
from .model_pins import GAME_COMMIT, GAME_SHA256, SEPARATOR_SHA256

ALLOWED_LICENSES = {'by', 'by-sa', 'by-nc', 'by-nc-sa'}


def stem_receipt(stem, original, track_id):
    path = Path(stem).with_suffix('.receipt.json')
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if (receipt['run_status'] != 'complete' or receipt['receipt_kind'] != 'contemporaneous_separator_run'
            or receipt['track_id'] != str(track_id)
            or receipt.get('settings') != dict(device='xpu', overlap=4, threads=8, full_song=True)
            or receipt['stem_sha256'] != checksum(stem) or receipt['source_sha256'] != checksum(original)
            or receipt['models']['separator_sha256'] != SEPARATOR_SHA256
            or receipt['models']['game_sha256'] != GAME_SHA256 or receipt['models']['game_commit'] != GAME_COMMIT):
        raise ValueError('Separation receipt does not match pinned models or current source/stem')
    return path


def inside(root, relative):
    path = (Path(root)/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Source path leaves dataset root')
    return path


def audio_licenses(path):
    result, current = {}, None
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        match = re.fullmatch(r'\d+/(\d+)\.mp3', line)
        if match:
            current = match.group(1)
        elif current:
            match = re.search(r'https?://creativecommons\.org/licenses/([a-z-]+)/([0-9.]+)', line)
            if match:
                result[current] = dict(kind=match.group(1), version=match.group(2), url=match.group(0))
    return result


def vocadito_sources(data_run, singer_limit):
    manifest_path = Path(data_run)/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    policy = DataPolicy(**manifest['policy'])
    rows, seen = [], set()
    for phrase in sorted(manifest['phrases'], key=lambda p: p['phrase_id']):
        if phrase['split'] != 'train' or phrase['singer_id'] in seen:
            continue
        if len(seen) >= singer_limit:
            break
        if manifest['singer_splits'][phrase['singer_id']] != 'train':
            raise ValueError('Vocadito partition mismatch')
        for key, digest in [('audio_path', phrase['sha256']), ('f0_path', phrase['reference_sha256']['f0']),
                            ('notes_path', phrase['reference_sha256']['notes'])]:
            if checksum(phrase[key]) != digest:
                raise ValueError('Vocadito crop/reference checksum changed')
        seen.add(phrase['singer_id'])
        rows.append(dict(dataset='vocadito', source_id=phrase['source_id'], phrase_id=phrase['phrase_id'],
                         group_id=phrase['singer_id'], group_kind='verified_singer_id', split='train',
                         source_audio=phrase['audio_path'], reference_kind='human_annotation',
                         reference_paths={key: phrase[key+'_path'] for key in ('f0', 'notes')},
                         source_map={key: phrase[key] for key in ('original_offset_seconds', 'start_sample', 'end_sample',
                                                                 'original_sample_rate_hz')},
                         duration_seconds=phrase['duration_seconds'], license=dict(kind='by', version='4.0',
                         url='https://zenodo.org/records/5578807', evidence='published dataset record; local archive identity not verified'),
                         legacy_phrase=phrase))
    return rows, dict(manifest_path=str(manifest_path), manifest_sha256=checksum(manifest_path),
                     review_path=str(Path(data_run)/'reviews.csv'), review_sha256=checksum(Path(data_run)/'reviews.csv'),
                     policy=policy.__dict__, selection='First phrase by ID per training singer, for plumbing only')


def jamendo_sources(dataset, stems, artist_limit, track_ids=None):
    dataset, stems = Path(dataset), Path(stems)
    inventory_path = dataset/'melody-extraction-pipeline/manifest.csv'
    with inventory_path.open(encoding='utf-8', newline='') as stream:
        inventory = list(csv.DictReader(stream))
    if len({row['track_id'] for row in inventory}) != len(inventory):
        raise ValueError('Duplicate Jamendo metadata IDs')
    wanted = None if track_ids is None else set(map(str, track_ids))
    if wanted is not None:
        if len(wanted) != len(track_ids) or not wanted or wanted-{row['track_id'] for row in inventory}:
            raise ValueError('Frozen track IDs are empty, duplicated or unknown')
    licenses = audio_licenses(dataset/'audio_licenses.txt')
    splits = singer_splits([row['artist_id'] for row in inventory], DataPolicy().split_seed)
    rows, excluded, seen = [], [], set()
    for row in sorted(inventory, key=lambda r: int(r['track_id'])):
        if wanted is not None and row['track_id'] not in wanted:
            continue
        stem = stems/(row['track_id']+'.wav')
        if not stem.is_file():
            continue
        reason = None
        license = licenses.get(row['track_id'])
        if splits[row['artist_id']] != 'train':
            reason = 'Not a training artist'
        elif not license or license['kind'] not in ALLOWED_LICENSES:
            reason = 'Unknown or non-derivative-permitting license'
        elif row['artist_id'] in seen:
            reason = 'One-song-per-artist cap'
        elif 'instrument---voice' not in row['tags'].split(';'):
            reason = 'No voice tag'
        if reason:
            excluded.append(dict(dataset='jamendo', source_id=row['track_id'], reason=reason))
            continue
        if len(seen) >= artist_limit:
            break
        original = inside(dataset, row['source_mp3'])
        note_path = dataset/'melody-extraction-pipeline/outputs/game_vocal'/(row['track_id']+'.json')
        if not original.is_file() or not note_path.is_file():
            excluded.append(dict(dataset='jamendo', source_id=row['track_id'], reason='Missing original audio or full-song note predictions'))
            continue
        seen.add(row['artist_id'])
        receipt = stem_receipt(stem, original, row['track_id'])
        rows.append(dict(dataset='jamendo', source_id=row['track_id'], group_id=row['artist_id'],
                         group_kind='artist_proxy_singer_unknown', split='train', source_audio=str(stem.resolve()),
                         original_mp3=str(original), full_note_path=str(note_path), reference_kind='model_estimate',
                         reference_paths={}, source_map=None, license=license, tags=row['tags'],
                         separation_provenance='contemporaneous_separator_receipt' if receipt else 'legacy_retained_stem_no_individual_receipt',
                         receipt_path=str(receipt) if receipt else None,
                         legacy_history_path=str(dataset/'melody-extraction-pipeline/history.log')))
    return rows, excluded, dict(metadata_path=str(inventory_path), metadata_sha256=checksum(inventory_path),
                               licenses_path=str(dataset/'audio_licenses.txt'), licenses_sha256=checksum(dataset/'audio_licenses.txt'),
                               artist_splits=splits, selected_track_ids=sorted(wanted, key=int) if wanted else None,
                               selection='Frozen track allowlist' if wanted else 'Retained stems, increasing track ID, one song per training artist')


def vocal_windows(notes, duration, seconds=20., count=2, minimum=5.):
    if not np.isfinite(duration) or duration <= 0 or count < 1:
        raise ValueError('Invalid crop request')
    starts = np.arange(0., max(duration-seconds, 0.)+1e-9, 5.)
    cover = [sum(max(0., min(n['offset_sec'], start+seconds)-max(n['onset_sec'], start)) for n in notes) for start in starts]
    chosen = []
    for index in np.argsort(-np.array(cover), kind='stable'):
        if cover[index] < minimum or len(chosen) == count:
            break
        if all(abs(starts[index]-start) >= seconds for start in chosen):
            chosen.append(float(starts[index]))
    return sorted(chosen)


def fingerprint(audio, rate):
    """Phase-invariant envelope fingerprint; duplicate warning, not identity proof."""
    audio = np.asarray(audio, dtype=float)
    if audio.ndim == 2:
        audio = np.sqrt(np.mean(audio**2, axis=1))
    block = max(1, round(rate*.1))
    count = len(audio)//block
    if count < 10:
        return None
    envelope = np.sqrt(np.mean(audio[:count*block].reshape(count, block)**2, axis=1))
    signature = np.interp(np.linspace(0, count-1, 128), np.arange(count), envelope)
    spread = float(signature.std())
    return ((signature-signature.mean())/spread).tolist() if spread > 1e-8 else None


def probable_duplicate(left, right):
    if left['sha256'] == right['sha256']:
        return 'exact_audio_bytes'
    a, b = left.get('fingerprint'), right.get('fingerprint')
    if a is not None and b is not None and abs(left['duration']-right['duration']) < .2:
        if float(np.dot(a, b)/len(a)) >= .995:
            return 'similar_envelope_manual_review_required'
    return None
