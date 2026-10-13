"""Source safety, typed provenance and preparation integrity, with explicit fake GAME fixtures."""
import csv
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from bpdr_input import data_builder as builder
from bpdr_input.data_sources import (inside, audio_licenses, vocal_windows, fingerprint, probable_duplicate,
                                     vocadito_sources, jamendo_sources)
from bpdr_input.data_sources import stem_receipt
from bpdr_input.dataset import DataPolicy, singer_splits
from bpdr_input.pipeline import process, checksum
from bpdr_input import separation_adapter


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fake_pitch(audio, rate, cache):
    times = np.arange(len(audio)//round(rate*.02)+1)*.02
    return times, np.full(len(times), 440.), np.ones(len(times))


def fixture(root):
    crop = root/'data/vocadito_fixture'
    crop.mkdir(parents=True)
    times = np.arange(0, 8., .02)
    wave = .1*np.sin(2*np.pi*440*np.arange(8*22050)/22050)
    sf.write(crop/'audio.wav', wave, 22050, subtype='FLOAT')
    np.savetxt(crop/'f0.csv', np.column_stack([times, np.full(len(times), 440.)]), delimiter=',')
    notes = {name: [dict(start_seconds=float(start), duration_seconds=4., pitch_hz=440., partial=False)
                    for start in (0, 4)] for name in ('A1', 'A2')}
    dump(crop/'notes.json', notes)
    phrase = dict(phrase_id='vocadito_fixture_p00', source_id='vocadito_fixture', singer_id='S1', split='train',
                  original_offset_seconds=2., start_sample=44100, end_sample=220500,
                  original_sample_rate_hz=22050, duration_seconds=8.,
                  audio_path=str(crop/'audio.wav'), f0_path=str(crop/'f0.csv'), notes_path=str(crop/'notes.json'),
                  sha256=checksum(crop/'audio.wav'), reference_sha256=dict(f0=checksum(crop/'f0.csv'), notes=checksum(crop/'notes.json')))
    data_run = root/'runs/phase2/fixture'
    held = dict(phrase, phrase_id='DO_NOT_READ', split='final_holdout', singer_id='HELD', audio_path='does-not-exist')
    dump(data_run/'manifest.json', dict(policy=asdict(DataPolicy()), phrases=[phrase, held], singer_splits={'S1': 'train', 'HELD': 'final_holdout'}))
    (data_run/'reviews.csv').write_text('old review evidence, unchanged', encoding='utf-8')
    dataset = root/'data/mtg-jamendo'
    metadata = dataset/'melody-extraction-pipeline/manifest.csv'
    metadata.parent.mkdir(parents=True)
    artists = ['artist_'+str(i) for i in range(10)]
    groups = singer_splits(artists, DataPolicy().split_seed)
    chosen = next(artist for artist in artists if groups[artist] == 'train')
    with metadata.open('w', newline='') as stream:
        fields = ['track_id', 'artist_id', 'source_mp3', 'tags', 'duration_annotation_sec']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, artist in enumerate(artists):
            writer.writerow(dict(track_id=str(index+100), artist_id=artist, source_mp3=f'vocal-tagged-audio/{index+100}.low.mp3', tags='instrument---voice', duration_annotation_sec='90'))
    track_id = str(artists.index(chosen)+100)
    (dataset/'audio_licenses.txt').write_text(f'00/{track_id}.mp3\nhttps://creativecommons.org/licenses/by/4.0/\n', encoding='utf-8')
    original = dataset/'vocal-tagged-audio'/(track_id+'.low.mp3')
    original.parent.mkdir()
    original.write_bytes(b'explicit fixture, not actual compressed music')
    history = dataset/'melody-extraction-pipeline/history.log'
    history.write_text('fixture legacy history', encoding='utf-8')
    dump(dataset/'melody-extraction-pipeline/outputs/game_vocal'/(track_id+'.json'),
         dict(track_id=track_id, method='game_vocal', notes=[dict(onset_sec=0., offset_sec=8., model_midi_pitch=69., midi_pitch=81.)]))
    stems = root/'data/jamendo-vocal-stems'
    stems.mkdir()
    # Different byte encoding, duration and envelope avoids falsely duplicating the Vocadito tone.
    length = 10*22050
    envelope = .04+.02*np.sin(2*np.pi*np.arange(length)/(3*22050))
    sf.write(stems/(track_id+'.wav'), envelope*np.sin(2*np.pi*440*np.arange(length)/22050), 22050, subtype='FLOAT')
    return data_run, track_id


def models(_):
    return dict(game_sha256=builder.GAME_SHA256, separator_sha256=builder.SEPARATOR_SHA256,
                game_commit=builder.GAME_COMMIT, origin='explicit fake GAME test fixture')


def decode(inputs, output, _):
    output.mkdir(parents=True)
    for path in inputs.glob('*.wav'):
        length = sf.info(path).duration
        with (output/(path.stem+'.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['onset', 'offset', 'pitch'])
            writer.writeheader()
            writer.writerows([dict(onset=0., offset=length/2, pitch=69.), dict(onset=length/2, offset=length, pitch=69.)])
    return dict(origin='explicit fake GAME test fixture', seed=None)


class SourceSafetyTests(unittest.TestCase):
    def test_frozen_selection_never_falls_back_to_older_stems(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, track_id = fixture(root)
            dataset, stems = root/'data/mtg-jamendo', root/'data/jamendo-vocal-stems'
            rows, excluded, provenance = jamendo_sources(dataset, stems, 5, [track_id])
            self.assertEqual([row['source_id'] for row in rows], [track_id])
            self.assertEqual(provenance['selected_track_ids'], [track_id])
            missing = next(str(i) for i in range(100, 110) if str(i) != track_id)
            rows, _, _ = jamendo_sources(dataset, stems, 5, [missing])
            self.assertEqual(rows, [])

    def test_frozen_ids_empty_unknown_or_duplicate_stop_source_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, track_id = fixture(root)
            for ids in ([], ['unknown'], [track_id, track_id]):
                with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, 'Frozen track IDs'):
                    jamendo_sources(root/'data/mtg-jamendo', root/'data/jamendo-vocal-stems', 5, ids)

    def test_separator_refuses_cached_outputs_if_stem_was_lost(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, track_id = fixture(root)
            (root/'data/jamendo-vocal-stems'/(track_id+'.wav')).unlink()
            with patch.object(separation_adapter, 'verified_models', side_effect=AssertionError('Must stop before model loading')):
                with self.assertRaisesRegex(ValueError, 'cached runner selection'):
                    separation_adapter.separate(root, track_id)

    def test_receipt_cannot_certify_changed_stem(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, track_id = fixture(root)
            stem = root/'data/jamendo-vocal-stems'/(track_id+'.wav')
            original = root/'data/mtg-jamendo/vocal-tagged-audio'/(track_id+'.low.mp3')
            receipt = dict(run_status='complete', receipt_kind='contemporaneous_separator_run', track_id=track_id,
                           settings=dict(device='xpu', overlap=4, threads=8, full_song=True), stem_sha256=checksum(stem), source_sha256=checksum(original), models=models(None))
            dump(stem.with_suffix('.receipt.json'), receipt)
            self.assertEqual(stem_receipt(stem, original, track_id), stem.with_suffix('.receipt.json'))
            stem.write_bytes(b'changed stem')
            with self.assertRaises(ValueError):
                stem_receipt(stem, original, track_id)

    def test_receipt_with_different_overlap_cannot_resume_pinned_sample(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, track_id = fixture(root)
            stem = root/'data/jamendo-vocal-stems'/(track_id+'.wav')
            original = root/'data/mtg-jamendo/vocal-tagged-audio'/(track_id+'.low.mp3')
            receipt = dict(run_status='complete', receipt_kind='contemporaneous_separator_run', track_id=track_id,
                           settings=dict(device='xpu', overlap=2, threads=8, full_song=True),
                           stem_sha256=checksum(stem), source_sha256=checksum(original), models=models(None))
            dump(stem.with_suffix('.receipt.json'), receipt)
            with self.assertRaisesRegex(ValueError, 'receipt does not match'):
                stem_receipt(stem, original, track_id)

    def test_separator_plan_is_licensed_training_only_and_does_not_replace_stems(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, existing = fixture(root)
            dataset = root/'data/mtg-jamendo'
            (dataset/'audio_licenses.txt').write_text(''.join(f'00/{track}.mp3\nhttps://creativecommons.org/licenses/by/4.0/\n' for track in range(100, 110)))
            rows = separation_adapter.sample(root, 2)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row['artist_id'] for row in rows}), 2)
            self.assertNotIn(existing, [row['track_id'] for row in rows])
            with patch.object(separation_adapter, 'verified_models', side_effect=AssertionError('Must refuse before model loading')):
                with self.assertRaises(ValueError):
                    separation_adapter.separate(root, existing)

    def test_escape_path_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                inside(folder, '../outside.wav')

    def test_unknown_license_never_inherits_preceding_track(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'licenses.txt'
            path.write_text('00/1.mp3\nhttps://creativecommons.org/licenses/by-nc-nd/4.0/\n00/2.mp3\nunknown\n')
            result = audio_licenses(path)
            self.assertEqual(result['1']['kind'], 'by-nc-nd')
            self.assertNotIn('2', result)

    def test_fixed_window_selection_retains_source_time_origin(self):
        notes = [dict(onset_sec=55., offset_sec=65.)]
        self.assertEqual(vocal_windows(notes, 100., count=1), [45.])
        self.assertEqual(vocal_windows([], 100.), [])

    def test_exact_and_similar_envelope_duplicates_are_flagged(self):
        audio = np.sin(np.arange(20000)*.13)*np.linspace(.01, .2, 20000)
        left = dict(sha256='a', duration=2., fingerprint=fingerprint(audio, 10000))
        self.assertEqual(probable_duplicate(left, dict(left)), 'exact_audio_bytes')
        right = dict(sha256='b', duration=2., fingerprint=fingerprint(audio*.5, 10000))
        self.assertEqual(probable_duplicate(left, right), 'similar_envelope_manual_review_required')

    def test_reference_checksum_change_blocks_source_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_run, _ = fixture(root)
            (root/'data/vocadito_fixture/notes.json').write_text('{}')
            with self.assertRaises(ValueError):
                vocadito_sources(data_run, 1)

    def test_nd_stem_is_excluded_before_prediction_or_audio_open(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, track_id = fixture(root)
            dataset = root/'data/mtg-jamendo'
            (dataset/'audio_licenses.txt').write_text(f'00/{track_id}.mp3\nhttps://creativecommons.org/licenses/by-nd/4.0/\n')
            rows, excluded, _ = jamendo_sources(dataset, root/'data/jamendo-vocal-stems', 1)
            self.assertEqual(rows, [])
            self.assertIn('non-derivative', excluded[0]['reason'])


class BuilderTests(unittest.TestCase):
    def test_missing_and_invalid_game_notes_cannot_look_usable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'notes.csv'
            self.assertEqual(builder.parse_notes(path, 1.)[0], [])
            path.write_text('onset,offset,pitch\n0,2,69\n')
            with self.assertRaises(ValueError):
                builder.parse_notes(path, 1.)

    def test_overlap_is_withheld_even_when_trackers_agree(self):
        times = np.arange(0, 1, .02)
        features = dict(times_seconds=times, f0_hz=np.full(len(times), 440.), voiced_mask=np.ones(len(times), dtype=bool))
        notes = [dict(onset_sec=0., offset_sec=1., model_midi_pitch=69.)]*2
        result, _ = builder.screen(features, np.full(len(times), 440.), notes)
        self.assertEqual(result['status'], 'withheld')
        self.assertIn('OVERLAPPING_PREDICTED_NOTES', result['reasons'])

    def test_pinned_checkpoint_mismatch_blocks_processing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.pt'
            path.write_bytes(b'altered')
            with self.assertRaises(ValueError):
                builder.checked_hash(path, builder.GAME_SHA256)

    def test_typed_build_preserves_inputs_and_blocks_research_even_after_review_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_run, _ = fixture(root)
            review_hash = checksum(data_run/'reviews.csv')
            prepared, manifest = builder.build(root, data_run, 1, 1, processor=lambda p, o, c: process(p, o, c, pitch_provider=fake_pitch),
                                               decoder=decode, model_verifier=models)
            records = manifest['records']
            self.assertEqual({row['reference_kind'] for row in records}, {'human_annotation', 'model_estimate'})
            self.assertEqual(len(records), 2)
            self.assertEqual(checksum(data_run/'reviews.csv'), review_hash)
            self.assertEqual(records[0]['source_map']['original_offset_seconds'], 2.)
            self.assertTrue(all(not row['training_eligible'] for row in records))
            self.assertTrue(builder.validate_run(prepared, model_verifier=models)['verified'])
            (prepared/'reviews.csv').write_text('forged approved review does not create training pairs')
            self.assertEqual(builder.validate_run(prepared, model_verifier=models)['training_ready_examples'], 0)
            Path(records[0]['accepted_audio_path']).write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                builder.validate_run(prepared, model_verifier=models)


if __name__ == '__main__':
    unittest.main()
