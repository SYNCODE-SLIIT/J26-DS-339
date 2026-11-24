import unittest
import numpy as np

from bpdr_input.repair.curve_conditioning import condition_curve, conditioned_proposal, CONDITIONS
from bpdr_input.repair.held_voice_policy import HeldPolicy, proposal


class CurveConditioningTests(unittest.TestCase):
    def test_no_support_expansion_or_amplitude_increase_across_gaps(self):
        curve = np.r_[np.zeros(7), [11., 20., 12., 25., 13.], np.zeros(3), [-8., -14., -9.], np.zeros(7)]
        original = curve.copy()
        for kind in CONDITIONS:
            result = condition_curve(curve, kind)
            np.testing.assert_array_equal(result != 0, curve != 0)
            np.testing.assert_array_equal(np.sign(result), np.sign(curve))
            self.assertLessEqual(abs(result).max(), abs(curve).max())
            self.assertTrue(np.all(result[12:15] == 0))
        np.testing.assert_array_equal(curve, original)

    def test_changing_input_not_reference_determines_plateau_and_identity(self):
        curve = np.r_[0., 10., 20., 30., 0.]
        np.testing.assert_array_equal(condition_curve(curve, 'span_median'), [0., 20., 20., 20., 0.])
        for kind in CONDITIONS:
            np.testing.assert_array_equal(condition_curve(np.zeros(80), kind), np.zeros(80))

    def test_rejects_nonfinite_unknown_or_mixed_direction_support(self):
        for values, kind in [([np.nan], 'median11'), ([1.], 'unknown'), ([10., -10.], 'span_median')]:
            with self.assertRaises(ValueError):
                condition_curve(np.array(values), kind)

    def test_conditioned_proposal_rechecks_exact_permission_support(self):
        audio = np.zeros(220500, np.float32)
        times = np.arange(501)*.02
        raw = np.zeros((501, 71), np.float32)
        raw[:, 65:67] = 1.
        pred = np.zeros((3, 501), np.float32)
        pred[:, 200:260] = 20.+2.*np.sin(np.arange(60)*.7)
        policy = HeldPolicy(activation_cents=8.)
        analysis = dict(features=dict(times_seconds=times), views=[raw]*3)
        pair = dict(audio_A=audio, A=analysis)
        baseline = proposal(audio, times, [raw]*3, pred, policy)
        self.assertTrue(baseline['plan']['permitted'])
        self.assertFalse(baseline['plan']['identity'])
        for kind in CONDITIONS:
            curve, plan, _ = conditioned_proposal(pair, 'A', pred, policy, kind)
            self.assertTrue(plan['permitted'])
            self.assertEqual(plan['intervals_seconds'], baseline['plan']['intervals_seconds'])
            self.assertEqual(plan['affected_fraction'], baseline['plan']['affected_fraction'])
            self.assertEqual(plan['policy'], baseline['plan']['policy'])
            np.testing.assert_array_equal(curve != 0, baseline['curve'] != 0)


if __name__ == '__main__':
    unittest.main()
