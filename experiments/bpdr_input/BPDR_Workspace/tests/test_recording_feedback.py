import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from bpdr_input.application.recording_feedback import timeline, for_recording
from bpdr_input.audio.pipeline import checksum


class RecordingFeedbackTests(unittest.TestCase):
    def tone(self, rate=22050, seconds=1.):
        return (.15 * np.sin(2 * np.pi * 220 * np.arange(round(rate * seconds)) / rate)).astype(np.float32)

    def pitch(self):
        times = np.arange(51) * .02
        return dict(times_seconds=times, f0_hz=np.full(51, 220.),
                    confidence=np.full(51, .9), voiced_mask=np.ones(51, dtype=bool))

    def test_silence_is_localized_without_changing_acceptance(self):
        result = timeline(np.zeros(22050, dtype=np.float32), 22050)
        self.assertEqual(len(result['observations']), 1)
        row = result['observations'][0]
        self.assertEqual((row['start_s'], row['end_s']), (0., 1.))
        self.assertEqual(row['category'], 'quiet_or_pause')
        self.assertFalse(result['acceptance_changed'])
        self.assertFalse(result['intended_notes_assessed'])

    def test_intentional_pause_remains_neutral_and_samples_are_unchanged(self):
        audio = self.tone(seconds=2.)
        audio[22050:33075] = 0
        preserved = audio.copy()
        result = timeline(audio, 22050)
        quiet = [r for r in result['observations'] if r['category'] == 'quiet_or_pause']
        self.assertEqual(len(quiet), 1)
        self.assertAlmostEqual(quiet[0]['start_s'], 1.)
        self.assertAlmostEqual(quiet[0]['end_s'], 1.5)
        self.assertIn('intentional', quiet[0]['guidance'])
        np.testing.assert_array_equal(audio, preserved)

    def test_peaks_are_observations_without_an_asserted_distortion_label(self):
        audio = self.tone()
        audio[8820:9261] = 1.
        row = next(r for r in timeline(audio, 22050)['observations'] if r['category'] == 'near_full_scale')
        self.assertAlmostEqual(row['start_s'], .4)
        self.assertAlmostEqual(row['end_s'], .42)
        self.assertIn('Listen', row['guidance'])

    def test_disagreement_has_times_but_is_not_an_intended_note_error(self):
        features = self.pitch()
        secondary = np.full(51, 220.)
        secondary[10:20] *= 2 ** (70 / 1200)
        before = copy.deepcopy(features)
        result = timeline(self.tone(), 22050, features, secondary)
        row = next(r for r in result['observations'] if r['category'] == 'uncertain_pitch_measurement')
        self.assertAlmostEqual(row['start_s'], .2)
        self.assertAlmostEqual(row['end_s'], .4)
        self.assertIn('intentional', row['guidance'])
        self.assertEqual(result['independent_pitch_frames_compared'], 50)
        for key in features:
            np.testing.assert_array_equal(features[key], before[key])

    def test_short_disagreement_and_unvoiced_gaps_are_not_flagged_as_wrong_notes(self):
        features = self.pitch()
        secondary = np.full(51, 220.)
        secondary[10:12] *= 2 ** (100 / 1200)
        secondary[20:40] *= 2
        features['voiced_mask'][20:40] = False
        result = timeline(self.tone(), 22050, features, secondary)
        self.assertFalse(any(r['category'] == 'uncertain_pitch_measurement' for r in result['observations']))

    def test_sample_free_working_endpoint_is_excluded(self):
        features = self.pitch()
        features['_audio_duration_seconds'] = 1.
        secondary = np.full(51, 220.)
        secondary[-1] = 440.
        result = timeline(np.r_[self.tone(), np.float32(.1)], 22050, features, secondary)
        self.assertEqual(result['independent_pitch_frames_compared'], 50)
        self.assertFalse(any(r['category'] == 'uncertain_pitch_measurement' for r in result['observations']))

    def test_invalid_grids_and_invalid_audio_are_refused(self):
        for audio in [np.array([]), np.array([np.nan]), np.zeros((3, 2))]:
            with self.assertRaises(ValueError):
                timeline(audio, 22050)
        features = self.pitch()
        features['times_seconds'][10] += .001
        with self.assertRaisesRegex(ValueError, 'regular'):
            timeline(self.tone(), 22050, features, np.full(51, 220.))

    def test_cache_is_bound_to_source_working_audio_and_features(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, working = root / 'original.wav', root / 'accepted.wav'
            sf.write(source, self.tone(), 22050, subtype='PCM_24')
            sf.write(working, self.tone(), 22050, subtype='PCM_24')
            features = self.pitch()
            np.savez_compressed(root / 'features.npz', **features)
            (root / 'diagnostics.json').write_text(json.dumps(dict(source=dict(
                preserved_path=str(source), sha256=checksum(source)))))
            report = dict(source=str(source), handoff=dict(accepted_audio=dict(path=str(working), checksum=checksum(working))))
            with patch('bpdr_input.repair.rule_correction.fine_hz', return_value=np.full(51, 220.)) as pitch:
                first = for_recording(root, report)
                self.assertIs(for_recording(root, report, first), first)
                self.assertEqual(pitch.call_count, 1)
                features['confidence'][:] = .1
                np.savez_compressed(root / 'features.npz', **features)
                second = for_recording(root, report, first)
                self.assertNotEqual(first['binding'], second['binding'])
                self.assertEqual(pitch.call_count, 2)
            working.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Working input changed'):
                for_recording(root, report, first)


if __name__ == '__main__':
    unittest.main()
