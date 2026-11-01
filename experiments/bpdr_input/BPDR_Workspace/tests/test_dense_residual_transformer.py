import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from bpdr_input.modeling.joint_transformer import JointTransformer, predict_raw as parent_predict
from bpdr_input.modeling.dense_residual_transformer import DenseResidualTransformer, predict_raw, save, load


class DenseResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(339)
        self.parent = JointTransformer().eval()
        with torch.no_grad():
            self.parent.regression.weight.normal_()
        self.model = DenseResidualTransformer(self.parent)
        self.raw = np.zeros((110, 71), np.float32)
        self.raw[:, 64] = np.sin(np.arange(110) / 12.) * 13.
        self.raw[:, 65:67] = 1.

    def test_zero_adapter_preserves_dense_function_and_metadata_invariance(self):
        expected = parent_predict(self.parent, self.raw)
        actual = predict_raw(self.model, self.raw)
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(a, b, atol=2e-6, rtol=1e-6)
        altered = self.raw.copy()
        altered[:, 69:71] = 999.
        np.testing.assert_array_equal(actual[0], predict_raw(self.model, altered)[0])

    def test_only_adapter_receives_gradients_and_parent_stays_eval(self):
        self.model.train()
        self.assertFalse(self.parent.training)
        features = torch.randn(8, 35)
        out, logit, _ = self.model.forward_embeddings(features, True)
        (out.square().mean() + logit.square().mean()).backward()
        self.assertTrue(all(p.grad is None for p in self.parent.parameters()))
        self.assertTrue(all(p.grad is not None for p in self.model.regression_delta.parameters()))
        self.assertTrue(all(p.grad is not None for p in self.model.activity_delta.parameters()))
        self.assertEqual(sum(p.numel() for p in self.model.parameters() if p.requires_grad), 72)

    def test_reload_bounds_and_unvoiced_identity(self):
        with torch.no_grad():
            self.model.regression_delta.weight.normal_()
            self.model.activity_delta.weight.normal_()
        self.raw[:5, 66] = 0.
        expected = predict_raw(self.model, self.raw)[0]
        self.assertTrue(np.all(abs(expected) <= 30.))
        np.testing.assert_array_equal(expected[:5], 0.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            save(path, self.model, dict(seed=339))
            restored, _ = load(path)
            np.testing.assert_array_equal(expected, predict_raw(restored, self.raw)[0])


if __name__ == '__main__':
    unittest.main()
