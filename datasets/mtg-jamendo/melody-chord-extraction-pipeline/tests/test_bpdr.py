"""BPDR export mechanics using synthetic audio; no model accuracy claims."""

from __future__ import annotations

import json
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from bpdr.crop import choose
from bpdr.selection import eligible
from bpdr import worker
from bpdr.export import Export, checksum, pending_reviews
from common import DATASET, inventory, melody_paths


class BPDRTests(unittest.TestCase):
    def test_selection_does_not_filter_or_split_by_artist(self):
        chosen = eligible(inventory(), DATASET / 'audio_licenses.txt')
        self.assertGreater(len(chosen), 5)
        self.assertTrue({'382', '383', '384', '386', '387'} <= chosen.keys())

    def test_crop_stays_within_twenty_seconds_and_uses_notes(self):
        notes = [{'onset_sec': i, 'offset_sec': i + .7} for i in range(30)]
        crop = choose(notes, 30)
        self.assertIsNotNone(crop)
        self.assertLessEqual(crop.end_sec - crop.start_sec, 20)
        self.assertGreaterEqual(crop.sung_sec, 5)
        self.assertEqual(crop.warnings, ())
        continuous = [{'onset_sec': i * .9, 'offset_sec': i * .9 + .9} for i in range(25)]
        fallback = choose(continuous, 23)
        self.assertIsNotNone(fallback)
        self.assertIn('no_natural_pause_at_crop_edge', fallback.warnings)

    def test_worker_artifacts_and_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rate = 22050
            audio = np.zeros(12 * rate, dtype=np.float32)
            for start, end in [(1, 4), (5, 9)]:
                t = np.arange((end - start) * rate) / rate
                audio[start * rate:end * rate] = .1 * np.sin(2 * np.pi * 440 * t)
            sf.write(root / 'stem.wav', audio, rate, subtype='FLOAT')
            notes = [{'onset_sec': 1, 'offset_sec': 4, 'model_midi_pitch': 69},
                     {'onset_sec': 5, 'offset_sec': 9, 'model_midi_pitch': 69}]
            (root / 'notes.json').write_text(json.dumps({'track_id': '9',
                'method': 'game_vocal', 'notes': notes}))

            def fake_extract(samples, sr, cache):
                times = np.arange((len(samples) + 440) // 441) * .02
                voiced = np.ones(len(times), dtype=bool)
                matrix = np.zeros((len(times), 71), dtype=np.float32)
                matrix[:, 65] = .95
                matrix[:, 66] = 1
                return ({'times_seconds': times, 'voiced_mask': voiced,
                         'f0_hz': np.full(len(times), 440.),
                         'pitch_change_valid': np.r_[False, np.ones(len(times) - 1, dtype=bool)],
                         'boundary_features_71': matrix,
                         'provisional_boundary_score': np.zeros(len(times))},
                        {'pitch_provider': 'synthetic_test_only'})

            with patch.object(worker, 'extract', fake_extract), \
                 patch.object(worker, 'game_decode', return_value=(notes, 'fake_csv_hash')), \
                 patch.object(worker, 'yin_pitch', side_effect=lambda a, sr, n: np.full(n, 440.)):
                result = worker.run({'stem': str(root / 'stem.wav'),
                    'destination': str(root / 'crop'), 'track': {'track_id': '9'},
                    'note_json': str(root / 'notes.json'), 'game_python': sys.executable,
                    'game_root': str(root), 'game_model': str(root / 'fake-model'),
                    'pesto_cache': str(root / 'cache')})
            self.assertEqual(result['status'], 'exported')
            self.assertFalse(result['training_eligible'])
            self.assertEqual(result['crop']['original_sample_rate_hz'], rate)
            self.assertEqual(result['crop']['start_sample'], 0)
            self.assertEqual(result['crop']['end_sample'], 12 * rate)
            with np.load(root / 'crop/features.npz') as archive:
                self.assertEqual(archive['boundary_features_71'].shape[1], 71)
                self.assertTrue(np.isfinite(archive['boundary_features_71']).all())
            with np.load(root / 'crop/masks.npz') as archive:
                self.assertIn('heuristic_boundary_score', archive.files)
            self.assertEqual(sf.info(root / 'crop/vocal.wav').subtype, 'PCM_24')

    def test_resume_preserves_human_review_row(self):
        tracks = inventory()
        track = next(row for row in tracks if row['track_id'] == '382')
        with tempfile.TemporaryDirectory() as tmp:
            export = Export(tracks, output_root=Path(tmp))
            folder = export.folder(track)
            folder.mkdir(parents=True)
            for name in ('vocal.wav', 'features.npz', 'boundaries.json', 'masks.npz'):
                (folder / name).write_bytes(b'synthetic placeholder ' + name.encode())
            record = {'track_id': '382', 'mtg_track_id': track['mtg_track_id'],
                      'crop_id': 'jamendo_382_crop_01', 'status': 'exported',
                      'source_mp3': track['source_mp3'],
                      'license': export.licenses['382'],
                      'model_sha256': export.model_sha256,
                      'source_mp3_sha256': checksum(DATASET / track['source_mp3']),
                      'full_game_json_sha256': checksum(melody_paths('game_vocal', track)['json']),
                      'artifacts': {name: checksum(folder / name) for name in
                                    ('vocal.wav', 'features.npz', 'boundaries.json', 'masks.npz')},
                      'review_status': 'pending', 'training_eligible': False}
            (folder / 'record.json').write_text(json.dumps(record))
            (folder / 'reviews.csv').write_text(pending_reviews(record))
            resumed = Export(tracks, output_root=Path(tmp))
            self.assertTrue(resumed.complete(track))
            with (folder / 'reviews.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            rows[0]['decision'] = 'accept'
            rows[0]['reviewer_id'] = 'human_1'
            with (folder / 'reviews.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            again = Export(tracks, output_root=Path(tmp))
            self.assertTrue(again.complete(track))
            with (folder / 'reviews.csv').open(newline='') as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(saved[0]['decision'], 'accept')
            self.assertEqual(saved[0]['reviewer_id'], 'human_1')
            self.assertEqual(len(saved), 2)


if __name__ == '__main__':
    unittest.main()
