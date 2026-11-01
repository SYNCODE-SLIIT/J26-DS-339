import unittest
import numpy as np
import torch
from bpdr_input.modeling.joint_transformer import JointTransformer, predict_raw as original
from bpdr_input.modeling.activity_decoding import predict_raw


class ActivityDecodingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(339)
        self.model = JointTransformer().eval()
        self.raw = np.zeros((110, 71), np.float32); self.raw[:, 65:67] = 1.
        with torch.no_grad():
            self.model.regression.bias.fill_(.5)

    def test_expected_replays_original_and_binary_uses_conditional_amplitude(self):
        expected = original(self.model, self.raw)
        for old, new in zip(expected, predict_raw(self.model, self.raw, mode='expected')):
            np.testing.assert_array_equal(old, new)
        np.testing.assert_array_equal(predict_raw(self.model, self.raw)[0], 0.)
        with torch.no_grad():
            self.model.activity.weight.zero_(); self.model.activity.bias.fill_(0.)
        actual = predict_raw(self.model, self.raw)[0]
        self.assertTrue(np.all(abs(actual) <= 30.))
        np.testing.assert_allclose(actual, 30 * np.tanh(.5), atol=2e-6)

    def test_metadata_invariance_unvoiced_identity_and_invalid_mode(self):
        self.raw[:3, 66] = 0.
        with torch.no_grad():
            self.model.activity.bias.fill_(2.)
        a = predict_raw(self.model, self.raw)[0]
        modified = self.raw.copy(); modified[:, 69:71] = 999.
        np.testing.assert_array_equal(a, predict_raw(self.model, modified)[0])
        np.testing.assert_array_equal(a[:3], 0.)
        with self.assertRaises(ValueError):
            predict_raw(self.model, self.raw, mode='tuned')


if __name__ == '__main__':
    unittest.main()
