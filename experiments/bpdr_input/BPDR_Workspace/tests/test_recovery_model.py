import unittest
import numpy as np

from bpdr_input.modeling.recovery_model import predict_raw, propose, scope_mask


class ConstantEstimator:
    def __init__(self, cents):
        self.cents = cents

    def predict(self, features):
        return np.full(len(features), self.cents)


def tone(frames=201):
    raw = np.zeros((frames,71),np.float32)
    raw[:,64] = 9.+2*np.sin(np.arange(frames)*.4)
    raw[:,65] = .99
    raw[:,66] = 1.
    raw[:,68] = -30.
    return raw


class RecoveryModelTests(unittest.TestCase):
    def test_operating_policy_is_applied_once_and_never_clips_predictions(self):
        raw = tone()
        model = dict(kind="extra_trees",estimator=ConstantEstimator(12.))
        payload = dict(operating_policy=dict(gain=.5,deadband_cents=5.))
        result = propose([raw,raw.copy(),raw.copy()],model,payload)
        np.testing.assert_array_equal(result["requested_curve"],np.full(len(raw),6.))
        np.testing.assert_array_equal(predict_raw(model,raw),np.full(len(raw),12.))
        np.testing.assert_array_equal(result["disagreement"],np.zeros(len(raw)))

    def test_no_eligible_long_voice_cannot_request_an_edit(self):
        raw = tone(30)
        model = dict(kind="extra_trees",estimator=ConstantEstimator(20.))
        result = propose([raw,raw,raw],model,dict(operating_policy=dict(gain=1.,deadband_cents=3.)))
        np.testing.assert_array_equal(result["requested_curve"],np.zeros(len(raw)))
        self.assertEqual(result["scope_metadata"]["eligible_frames"],0)

    def test_pitch_placeholder_on_unvoiced_frames_is_not_a_valid_scope(self):
        raw = tone()
        raw[:,66] = 0
        mask, metadata = scope_mask(raw)
        self.assertFalse(mask.any())
        self.assertEqual(metadata["voiced_frames"],0)


if __name__ == "__main__":
    unittest.main()
