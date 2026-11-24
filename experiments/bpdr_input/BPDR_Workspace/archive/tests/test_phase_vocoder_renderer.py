import unittest
import numpy as np

from bpdr_input.repair.phase_vocoder_renderer import render_candidate
from bpdr_input.repair.rule_correction import fine_hz


class PhaseVocoderRendererTests(unittest.TestCase):
    def setUp(self):
        self.audio = (.1*np.sin(2*np.pi*220*np.arange(110250)/22050)).astype(np.float32)
        self.times = np.arange(251)*.02

    def test_actual_small_shift_and_exact_outside_samples(self):
        curve = np.where((self.times >= 2.) & (self.times < 3.2), 20., 0.)
        result, report = render_candidate(self.audio, 22050, self.times, curve, [(2., 3.2)])
        samples = np.arange(len(self.audio))/22050
        outside = (samples < 2.) | (samples >= 3.2)
        np.testing.assert_array_equal(result[outside], self.audio[outside])
        self.assertEqual(result.shape, self.audio.shape)
        self.assertEqual(result.dtype, np.float32)
        before = fine_hz(self.audio, 22050, self.times)
        after = fine_hz(result, 22050, self.times)
        scored = (self.times > 2.3) & (self.times < 2.9)
        measured = 1200*np.log2(after[scored]/before[scored])
        self.assertAlmostEqual(float(np.median(measured)), 20., delta=3.)
        self.assertFalse(report['production_eligible'])

    def test_identity_and_invalid_curves(self):
        result, _ = render_candidate(self.audio, 22050, self.times, np.zeros(251), [])
        np.testing.assert_array_equal(result, self.audio)
        for curve, intervals in [(np.ones(251)*31., [(0., 5.)]),
                (np.where((self.times >= 2.) & (self.times < 3.2), 20.+np.sin(np.arange(251)), 0.), [(2., 3.2)]),
                (np.where((self.times >= 2.) & (self.times < 2.4), 20., 0.), [(2., 2.4)])]:
            with self.assertRaises(ValueError):
                render_candidate(self.audio, 22050, self.times, curve, intervals)


if __name__ == '__main__':
    unittest.main()
