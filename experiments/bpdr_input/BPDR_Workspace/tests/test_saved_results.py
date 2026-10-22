import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from bpdr_input.saved_results import checked_candidate, checksum, paired_snr_gain


class SavedResultsTests(unittest.TestCase):
    def test_changed_waveform_cannot_keep_verified_output_label(self):
        with TemporaryDirectory() as folder:
            p = Path(folder) / 'candidate.wav'
            p.write_bytes(b'original-output')
            candidate = dict(path=str(p), sha256=checksum(p))
            self.assertEqual(checked_candidate(candidate), p)
            p.write_bytes(b'other-output')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                checked_candidate(candidate)

    def test_paired_gain_scores_reduction_against_same_reference(self):
        reference = np.array([0., .2, -.2, 0.])
        noise = np.array([.1, -.1, .1, -.1])
        self.assertAlmostEqual(paired_snr_gain(reference, reference+noise, reference+noise/2), 6.0205999)

    def test_misaligned_audio_cannot_be_scored_as_improvement(self):
        with self.assertRaisesRegex(ValueError, 'same mono sample grid'):
            paired_snr_gain(np.zeros(4), np.zeros(3), np.zeros(4))


if __name__ == '__main__':
    unittest.main()
