"""Resumable, per-track BPDR export alongside each song's other outputs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from common import ASSETS, DATASET, OUTPUTS, ROOT, WORK, melody_paths, python_in
from .selection import ELIGIBILITY_VERSION, eligible


REVIEW_FIELDS = ('crop_id', 'crop_sha256', 'reviewer_slot', 'reviewer_id', 'musical_experience',
                 'decision', 'reason', 'approved_intervals_json', 'unclear_transitions',
                 'multiple_voices', 'accompaniment_leakage', 'reviewed_at_utc')
ARTIFACTS = ('vocal.wav', 'features.npz', 'boundaries.json', 'masks.npz')


def checksum(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def pending_reviews(record: dict) -> str:
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=REVIEW_FIELDS)
    writer.writeheader()
    for slot in ('1', '2'):
        writer.writerow({'crop_id': record['crop_id'],
                         'crop_sha256': record['artifacts']['vocal.wav'],
                         'reviewer_slot': slot, 'reviewer_id': '', 'musical_experience': '',
                         'decision': 'pending', 'reason': '', 'approved_intervals_json': '',
                         'unclear_transitions': '', 'multiple_voices': '',
                         'accompaniment_leakage': '', 'reviewed_at_utc': ''})
    return buffer.getvalue()


class Export:
    def __init__(self, tracks: list[dict[str, str]], *, output_root: Path = OUTPUTS) -> None:
        self.output_root = output_root
        self.licenses = eligible(tracks, DATASET / 'audio_licenses.txt')
        self.model_sha256 = {
            'game': checksum(ASSETS / 'models/GAME-1.0-medium/model.pt'),
            'separator': checksum(ASSETS / 'models/mlx-audio-separator/model_bs_roformer_ep_317_sdr_12.9755.ckpt'),
        }
        self.code_sha256 = {name: checksum(Path(__file__).with_name(name)) for name in
                            ('analysis.py', 'crop.py', 'worker.py', 'selection.py', 'export.py')}
        self._checked: dict[str, dict | None] = {}

    def includes(self, track_id: str) -> bool:
        return track_id in self.licenses

    def folder(self, track: dict[str, str]) -> Path:
        return self.output_root / track['mtg_track_id'] / 'bpdr'

    def complete(self, track: dict[str, str]) -> dict | None:
        track_id = track['track_id']
        if not self.includes(track_id):
            return None
        if track_id in self._checked:
            return self._checked[track_id]
        folder = self.folder(track)
        marker = folder / 'record.json'
        if not marker.is_file():
            self._checked[track_id] = None
            return None
        try:
            record = json.loads(marker.read_text(encoding='utf-8'))
            if (record.get('track_id') != track_id or
                    record.get('mtg_track_id') != track['mtg_track_id'] or
                    record.get('source_mp3') != track['source_mp3'] or
                    record.get('license') != self.licenses[track_id] or
                    record.get('model_sha256') != self.model_sha256 or
                    record.get('source_mp3_sha256') != checksum(DATASET / track['source_mp3']) or
                    record.get('full_game_json_sha256') !=
                    checksum(melody_paths('game_vocal', track)['json'])):
                raise ValueError('source, model, license, or GAME-note identity changed')
            artifacts = record.get('artifacts')
            if not isinstance(artifacts, dict):
                raise ValueError('missing artifact hashes')
            if record.get('status') == 'exported' and set(artifacts) != set(ARTIFACTS):
                raise ValueError('incomplete exported crop')
            if record.get('status') != 'exported' and artifacts:
                raise ValueError('unexpected artifacts for withheld crop')
            for name, expected in artifacts.items():
                if not (folder / name).is_file() or checksum(folder / name) != expected:
                    raise ValueError(f'artifact changed: {name}')
            if record.get('review_status') == 'pending' and not (folder / 'reviews.csv').is_file():
                raise ValueError('missing pending review rows')
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f'BPDR record needs manual inspection at {folder}: {exc}') from exc
        self._checked[track_id] = record
        return record

    def export(self, track: dict[str, str], stem: Path) -> dict:
        previous = self.complete(track)
        if previous is not None:
            return previous
        if not self.includes(track['track_id']):
            raise ValueError('Track is not license/voice eligible for BPDR')
        if not stem.is_file():
            raise FileNotFoundError(f'Missing full vocal stem: {stem}')
        receipt_path = stem.with_suffix('.receipt.json')
        if not receipt_path.is_file():
            raise ValueError('BPDR stem has no separation receipt')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        source_sha = checksum(DATASET / track['source_mp3'])
        if (receipt.get('track_id') != track['track_id'] or
                receipt.get('source_sha256') != source_sha or
                receipt.get('separator_sha256') != self.model_sha256['separator'] or
                receipt.get('stem_sha256') != checksum(stem)):
            raise ValueError('BPDR stem receipt does not match source/model/stem')
        note_json = melody_paths('game_vocal', track)['json']
        if not note_json.is_file():
            raise FileNotFoundError(f'Missing GAME notes: {note_json}')
        worker_python = python_in(ROOT / '.venv-bpdr')
        if not worker_python.is_file():
            raise FileNotFoundError('BPDR environment missing; install bpdr-requirements.txt')
        staging_root = WORK / 'bpdr_staging'
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = staging_root / f'{track["mtg_track_id"]}-{uuid.uuid4().hex}'
        request = {'track': track, 'stem': str(stem), 'destination': str(staging),
                   'note_json': str(note_json), 'game_python': str(python_in(ASSETS / '.venv-game')),
                   'game_root': str(ASSETS / 'tools/GAME'),
                   'game_model': str(ASSETS / 'models/GAME-1.0-medium/model.pt'),
                   'pesto_cache': str(WORK / 'bpdr_pesto_cache')}
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(ROOT / 'code')
        result = subprocess.run([str(worker_python), '-m', 'bpdr.worker'],
                                input=json.dumps(request), capture_output=True,
                                text=True, env=environment)
        if result.returncode:
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(f'BPDR worker exited {result.returncode}: '
                               f'{(result.stderr or result.stdout)[-700:]}')
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(f'BPDR worker returned invalid response: {result.stdout[-500:]}') from exc
        if payload['status'] == 'exported':
            if not all((staging / name).is_file() for name in ARTIFACTS):
                raise RuntimeError('BPDR worker did not produce all required artifacts')
            import numpy as np
            with np.load(staging / 'features.npz') as archive:
                matrix = archive['boundary_features_71']
                if matrix.ndim != 2 or matrix.shape[1] != 71 or not np.isfinite(matrix).all():
                    raise ValueError('BPDR model input failed validation')
            artifacts = {name: checksum(staging / name) for name in ARTIFACTS}
        else:
            # A rejected preparation may have written a partial WAV. Keep only
            # the decision record, never an unlisted or unvalidated artifact.
            for child in staging.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            artifacts = {}
        record = {**payload, 'schema_version': 'bpdr-track-v1',
                  'track_id': track['track_id'], 'mtg_track_id': track['mtg_track_id'],
                  'crop_id': f'jamendo_{track["track_id"]}_crop_01',
                  'source_mp3': track['source_mp3'], 'source_mp3_sha256': source_sha,
                  'full_game_json_sha256': checksum(note_json),
                  'full_stem_sha256': checksum(stem), 'separation_receipt': receipt,
                  'separation_backend': 'audio-separator 0.47.0 PyTorch' if os.name == 'nt'
                                        else 'mlx-audio-separator',
                  'artist_id_metadata_only': track.get('artist_id', ''),
                  'singer_identity': 'unknown', 'split': 'unassigned',
                  'license': self.licenses[track['track_id']],
                  'eligibility_version': ELIGIBILITY_VERSION,
                  'model_sha256': self.model_sha256, 'code_sha256': self.code_sha256,
                  'artifacts': artifacts, 'training_eligible': False,
                  'data_status': 'exploratory_not_research'}
        if record.get('review_status') == 'pending':
            (staging / 'reviews.csv').write_text(pending_reviews(record), encoding='utf-8')
        atomic_json(staging / 'record.json', record)
        destination = self.folder(track)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if (destination / 'record.json').exists() or (destination / 'reviews.csv').exists():
                raise FileExistsError(f'Existing BPDR evidence must not be overwritten: {destination}')
            orphan = WORK / 'bpdr_orphans' / f'{track["mtg_track_id"]}-{uuid.uuid4().hex}'
            orphan.parent.mkdir(parents=True, exist_ok=True)
            os.replace(destination, orphan)
        os.replace(staging, destination)
        self._checked[track['track_id']] = record
        return record
