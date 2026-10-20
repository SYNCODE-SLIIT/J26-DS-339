"""Contract checks for source preservation, time offsets, and note boundaries."""
from pathlib import Path
import tempfile
import sys
import unittest
import numpy as np
import soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from melody_understanding.audio import prepare_audio
from melody_understanding.features import segment_notes

class MelodyContractTests(unittest.TestCase):
    def test_resampling_preserves_duration_and_source(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'tone.wav'
            t = np.arange(44100) / 44100
            sf.write(p, .2 * np.sin(2 * np.pi * 440 * t), 44100)
            original = p.read_bytes()
            audio = prepare_audio(p)
            self.assertEqual(p.read_bytes(), original)
            self.assertEqual(audio.sample_rate, 22050)
            self.assertAlmostEqual(len(audio.samples) / audio.sample_rate, 1.0, places=4)
            self.assertEqual(audio.offset_s, 0.0)

    def test_silence_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'silence.wav'
            sf.write(p, np.zeros(22050), 22050)
            with self.assertRaisesRegex(ValueError, 'silent'):
                prepare_audio(p)

    def test_segmentation_keeps_rest_and_original_offset(self):
        features = {'f0_hz': np.array([440, 440, np.nan, 440, 440]),
            'voiced': np.array([True, True, False, True, True]),
            'voiced_probability': np.array([.9, .9, 0, .8, .8]),
            'times_s': np.array([1.0, 1.1, 1.2, 1.3, 1.4])}
        notes = segment_notes(features, 1.5)
        self.assertEqual(len(notes), 2)
        self.assertEqual(notes[0]['pitch_midi'], 69)
        self.assertAlmostEqual(notes[0]['onset_s'], 1.0)
        self.assertAlmostEqual(notes[0]['duration_s'], .2)
        self.assertAlmostEqual(notes[1]['onset_s'], 1.3)
        self.assertAlmostEqual(notes[1]['duration_s'], .2)

if __name__ == '__main__':
    unittest.main()
