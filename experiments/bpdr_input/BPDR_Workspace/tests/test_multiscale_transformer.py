import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch

from bpdr_input.modeling.joint_transformer import JointConfig
from bpdr_input.modeling.multiscale_transformer import MultiscaleTransformer, multiscale_tokens, predict_raw, save, load
from bpdr_input.experiments.learning.multiscale_transformer_training import difficult_original_pool


class MultiscaleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_broad_context_retained_while_current_region_stays_stable(self):
        raw = np.zeros((501, 71), np.float32); raw[:, 65:67] = 1.; raw[:, 68] = -20.
        raw[200:301, 64] = 200.; raw[225:276, 64] += 20.
        changed = raw.copy(); changed[:200, 64] = 900.; changed[301:, 64] = 900.
        x, pad = multiscale_tokens(raw, [250]); y, other_pad = multiscale_tokens(changed, [250])
        self.assertEqual(x.shape, (1, 51, 13))
        self.assertFalse(pad.any())
        self.assertNotEqual(float(x[0, 25, 0]), float(y[0, 25, 0]))
        np.testing.assert_allclose(x[:, :, 5:], y[:, :, 5:], atol=1e-5)
        np.testing.assert_array_equal(pad, other_pad)
        self.assertEqual(x[0, 0, 3], 1.)
        self.assertEqual(x[0, 0, 8], 0.)

    def test_hard_original_pool_uses_only_reliable_frames(self):
        x = np.zeros((50, 3, 13), np.float32); x[:, 1, 5] = np.arange(50)
        mask = np.arange(50) < 30
        pool = difficult_original_pool(x, mask)
        np.testing.assert_array_equal(pool, np.arange(24, 30))
        self.assertTrue(mask[pool].all())
        with self.assertRaises(ValueError):
            difficult_original_pool(x, np.zeros(50, bool))

    def test_nonzero_checkpoint_roundtrip_and_batch_independence(self):
        model = MultiscaleTransformer(JointConfig(hidden=16, layers=1, feed_forward=32))
        with torch.no_grad():model.regression.bias.fill_(.2)
        raw = np.zeros((80, 71), np.float32); raw[:, 65:67] = 1.; raw[0, 66] = 0.
        first = predict_raw(model, raw, 13)[0]
        self.assertEqual(first[0], 0.); self.assertGreater(float(abs(first).max()), 0.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'; save(path, model, dict(training_groups=['s1']))
            restored, payload = load(path)
            np.testing.assert_allclose(first, predict_raw(restored, raw, 80)[0], atol=1e-5)
            self.assertEqual(payload['training_groups'], ['s1'])


if __name__ == '__main__':unittest.main()
