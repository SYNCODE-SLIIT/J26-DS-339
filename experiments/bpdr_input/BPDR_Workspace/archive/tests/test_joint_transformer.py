import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from bpdr_input.modeling.joint_transformer import JointConfig, JointTransformer, joint_tokens, predict_raw, save, load
from bpdr_input.experiments.learning.joint_transformer_training import augmented_pairs


class JointTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_continuous_context_survives_step_abstention_without_boundary_input(self):
        raw = np.zeros((110, 71), np.float32)
        raw[:, 65:67] = 1.
        raw[:, 64] = 5.*np.sin(np.arange(110)*.7)
        x, pad = joint_tokens(raw, [55])
        self.assertGreater(float(abs(x[:, :, 0]).max()), .01)
        self.assertEqual(float(abs(x[:, :, 5:]).max()), 0.)
        changed = raw.copy()
        changed[:, 64] += 1200.
        changed[:, 69:71] = 99.
        y, pad2 = joint_tokens(changed, [55])
        np.testing.assert_allclose(x, y, atol=1e-5)
        np.testing.assert_array_equal(pad, pad2)

    def test_augmented_pair_preserves_source_and_exact_signed_feature_target(self):
        raw = np.zeros((120, 71), np.float32)
        raw[:, 65:67] = 1.
        source = dict(sha256='training-source', singer_id='training-singer')
        anchor = dict(group='training-singer', source=source, A=dict(views=[raw]))
        one = augmented_pairs([anchor], 339)[0]
        self.assertEqual(one['source'], source)
        self.assertEqual(one['role'], 'training_features_only')
        np.testing.assert_allclose(one['shifted'][:, 64]-one['clean'][:, 64], -one['target'], atol=1e-5)
        np.testing.assert_array_equal(raw[:, 64], np.zeros(120))
        self.assertTrue(one['plateau'].any())

    def test_nonzero_checkpoint_roundtrip_and_batch_inference(self):
        model = JointTransformer(JointConfig(hidden=16, layers=1, feed_forward=32))
        with torch.no_grad():
            model.regression.weight.fill_(.03)
            model.regression.bias.fill_(.1)
        raw = np.zeros((80, 71), np.float32)
        raw[:, 65:67] = 1.
        raw[0, 66] = 0.
        a = predict_raw(model, raw, 13)[0]
        self.assertTrue(np.any(a != 0))
        self.assertEqual(a[0], 0.)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.pt'
            save(path, model, dict(training_groups=['s1'], training_sources=['hash']))
            restored, metadata = load(path)
            np.testing.assert_allclose(a, predict_raw(restored, raw, 80)[0], atol=1e-5)
            self.assertEqual(metadata['training_groups'], ['s1'])


if __name__ == '__main__':
    unittest.main()
