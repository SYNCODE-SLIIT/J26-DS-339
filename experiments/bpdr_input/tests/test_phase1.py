import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import soundfile as sf
from bpdr_input.pipeline import process, validate_package, checksum
from bpdr_input.contracts import Status


def fixture_pitch(audio, sr, cache_dir):
    times = np.arange(0, len(audio)/sr, .020)
    f0 = np.full(len(times), 220.)
    confidence = np.full(len(times), .99)
    confidence[(times > .4) & (times < .6)] = 0
    return times, f0, confidence


class Phase1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root/'source.wav'
        self.sr = 44100
        self.audio = (.2*np.sin(2*np.pi*220*np.arange(self.sr)/self.sr)).astype('float32')

    def tearDown(self):
        self.tmp.cleanup()

    def run_audio(self, audio=None, sr=None):
        sf.write(self.source, self.audio if audio is None else audio, sr or self.sr, subtype='FLOAT')
        return process(self.source, self.root/'runs', self.root/'weights', pitch_provider=fixture_pitch)

    def test_preservation_alignment_repeatability_and_contract(self):
        import matplotlib
        original_backend = matplotlib.get_backend()
        first, directory = self.run_audio()
        self.assertEqual(matplotlib.get_backend(), original_backend)
        self.assertEqual(first.status, Status.ACCEPT_UNCHANGED)
        self.assertEqual(validate_package(first).sample_count, 22050)
        self.assertEqual(checksum(self.source), checksum(directory/'original.wav'))
        with np.load(directory/'features.npz') as archive:
            features = {key: archive[key] for key in archive.files}
        self.assertEqual(features['frame_features_70'].shape[1], 70)
        self.assertEqual(features['boundary_features_71'].shape[1], 71)
        self.assertTrue(np.isfinite(features['frame_features_70']).all())
        unvoiced = ~features['voiced_mask']
        self.assertTrue(unvoiced.any())
        self.assertTrue(np.isnan(features['f0_hz'][unvoiced]).all())
        self.assertTrue((features['pitch_change_cents'][~features['pitch_change_valid']] == 0).all())
        second, _ = self.run_audio()
        self.assertEqual(first.accepted_audio.checksum, second.accepted_audio.checksum)
        self.assertEqual(features['times_seconds'][0], 0)
        self.assertAlmostEqual(features['times_seconds'][1], .02)

    def test_antiphase_stereo_is_not_cancelled(self):
        result, directory = self.run_audio(np.column_stack([self.audio, -self.audio]))
        self.assertEqual(result.status, Status.ACCEPT_UNCHANGED)
        output, _ = sf.read(result.accepted_audio.path)
        self.assertGreater(np.sqrt(np.mean(output**2)), .1)
        self.assertEqual(json.loads((directory/'diagnostics.json').read_text())['operations'][0]['operation'], 'select_channel')

    def test_silence_clipping_and_long_input_stop_downstream(self):
        for audio, reason in [(np.zeros(self.sr), 'SILENT_RECORDING'),
                              (np.ones(self.sr), 'EXCESSIVE_CLIPPING'),
                              (np.zeros(self.sr*21), 'DURATION_OUT_OF_SCOPE')]:
            result, directory = self.run_audio(audio)
            self.assertEqual(result.status, Status.RERECORD_REQUIRED)
            self.assertIn(reason, result.reason_codes)
            self.assertFalse((directory/'accepted.wav').exists())
            with self.assertRaises(ValueError):
                validate_package(result)

    def test_corrupt_file_returns_technical_failure(self):
        self.source.write_bytes(b'not audio')
        result, _ = process(self.source, self.root/'runs', self.root/'weights', pitch_provider=fixture_pitch)
        self.assertEqual(result.status, Status.TECHNICAL_FAILURE)

    def test_nonfinite_samples_rejected(self):
        audio = self.audio.copy()
        audio[50] = np.nan
        result, _ = self.run_audio(audio)
        self.assertEqual(result.status, Status.TECHNICAL_FAILURE)
        self.assertIn('NONFINITE_SAMPLES', result.reason_codes)

    def test_empty_and_unsupported(self):
        result, _ = self.run_audio(np.zeros(0))
        self.assertEqual(result.status, Status.RERECORD_REQUIRED)
        other = self.root/'input.mp3'
        other.write_bytes(b'')
        result, _ = process(other, self.root/'runs', self.root/'weights')
        self.assertEqual(result.status, Status.TECHNICAL_FAILURE)
        self.assertIn('UNSUPPORTED_FORMAT', result.reason_codes)

    def test_missing_analyzer_cannot_export_accepted_audio(self):
        sf.write(self.source, self.audio, self.sr)
        def broken(*args):
            raise RuntimeError('model unavailable')
        result, directory = process(self.source, self.root/'runs', self.root/'weights', pitch_provider=broken)
        self.assertEqual(result.status, Status.TECHNICAL_FAILURE)
        self.assertFalse((directory/'accepted.wav').exists())

    def test_tampered_package_rejected(self):
        result, _ = self.run_audio()
        Path(result.accepted_audio.path).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_package(result)

    def test_invalid_analyzer_timestamps_block_export(self):
        sf.write(self.source, self.audio, self.sr)
        def invalid(*args):
            return np.array([0., .02, .01]), np.array([220., 220., 220.]), np.ones(3)
        result, directory = process(self.source, self.root/'runs', self.root/'weights', pitch_provider=invalid)
        self.assertEqual(result.status, Status.TECHNICAL_FAILURE)
        self.assertFalse((directory/'accepted.wav').exists())

    def test_nonstandard_rate_and_output_gain(self):
        sr = 48000
        # Square-wave edges provoke resampling overshoot without an overloaded source.
        audio = .99*np.sign(np.sin(2*np.pi*220*np.arange(sr+7)/sr))
        result, directory = self.run_audio(audio, sr)
        self.assertEqual(result.status, Status.ACCEPT_UNCHANGED)
        accepted = validate_package(result)
        self.assertLessEqual(abs(accepted.duration_seconds-len(audio)/sr), 1/22050)
        operations = json.loads((directory/'diagnostics.json').read_text())['operations']
        self.assertTrue(any(op['operation'] == 'peak_protection_gain' for op in operations))

    def test_flac_and_rests_and_duration(self):
        # Internal silence remains in place; an odd resampling length checks tolerance.
        audio = self.audio[:-7].copy()
        audio[10000:20000] = 0
        self.source = self.root/'source.flac'
        sf.write(self.source, audio, self.sr, subtype='PCM_24')
        result, _ = process(self.source, self.root/'runs', self.root/'weights', pitch_provider=fixture_pitch)
        self.assertEqual(result.status, Status.ACCEPT_UNCHANGED)
        accepted = validate_package(result)
        self.assertLessEqual(abs(accepted.duration_seconds-len(audio)/self.sr), 1/22050)
        output, _ = sf.read(accepted.path)
        self.assertLess(abs(output[5500:9500]).max(), 1e-6)


if __name__ == '__main__':
    unittest.main()
