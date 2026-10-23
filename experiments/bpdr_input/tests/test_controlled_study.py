import unittest
import numpy as np
from bpdr_input import controlled_study as cs


class GeneratorTests(unittest.TestCase):
    def test_phrase_has_one_long_note_inside_the_clip(self):
        for seed in range(20):
            rng = np.random.default_rng(seed); v = cs.voice(rng); notes = cs.phrase(v, rng, 6.)
            self.assertGreaterEqual(max(b-a for a, b, _ in notes), 1.0)
            self.assertTrue(all(0 < a < b <= 6. for a, b, _ in notes))
            self.assertTrue(all(notes[i][1] <= notes[i+1][0]+1e-9 for i in range(len(notes)-1)))

    def test_synthesis_is_finite_mono_and_unclipped(self):
        rng = np.random.default_rng(3); v = cs.voice(rng); notes = cs.phrase(v, rng, 6.)
        y = cs.synthesize(v, notes, 6., rng)
        self.assertEqual(y.dtype, np.float32); self.assertEqual(y.shape, (6*cs.SR,))
        self.assertTrue(np.isfinite(y).all()); self.assertLess(np.abs(y).max(), 1)

    def test_summary_matches_definitions(self):
        rows = [dict(correction_mae=10., zero_mae=20., response_gain=.5, boundary_sensitivity=1., unshifted_alteration=.2, outside_alteration=.3),
                dict(correction_mae=30., zero_mae=20., response_gain=.1, boundary_sensitivity=3., unshifted_alteration=.4, outside_alteration=.5)]
        s = cs.summary(rows)
        self.assertAlmostEqual(s['correction_gain'], 0.); self.assertAlmostEqual(s['median_response_gain'], .3)
        self.assertAlmostEqual(s['boundary_sensitivity'], 2.)



class ShowcaseTests(unittest.TestCase):
    def test_correction_span_is_the_largest_contiguous_voiced_request_unclipped(self):
        from bpdr_input.bpdr_showcase import correction_from_prediction
        pred = np.zeros(40); pred[5:8] = 6.; pred[15:30] = 33.; pred[20] = 2.
        voiced = np.ones(40, bool)
        curve = correction_from_prediction(pred, pred, voiced)
        self.assertTrue((curve[5:8] == 0).all())
        # The 2-cent dip at frame 20 splits the request; the longer part (21-29) is kept, values unclipped.
        self.assertTrue((curve[15:21] == 0).all()); np.testing.assert_array_equal(curve[21:30], pred[21:30])
        self.assertFalse(correction_from_prediction(pred, pred, np.zeros(40, bool)).any())


if __name__ == '__main__':
    unittest.main()
