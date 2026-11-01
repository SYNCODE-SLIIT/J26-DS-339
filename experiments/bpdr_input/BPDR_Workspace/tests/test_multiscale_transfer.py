import unittest
import numpy as np
import torch

from bpdr_input.modeling.joint_transformer import JointConfig, JointTransformer, predict_raw as parent_predict
from bpdr_input.modeling.multiscale_transformer import multiscale_tokens, predict_raw
from bpdr_input.modeling.multiscale_transfer import expand_parent


class TransferTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_initial_function_preserved_and_local_weights_can_learn(self):
        parent = JointTransformer(JointConfig(hidden=16, layers=1, feed_forward=32)).eval()
        with torch.no_grad():
            parent.regression.weight.copy_(torch.linspace(-.03, .03, 16)[None])
            parent.regression.bias.fill_(.2)
        child = expand_parent(parent)
        raw = np.zeros((200, 71), np.float32); raw[:, 65:67] = 1.
        raw[75:125, 64] = 20.
        np.testing.assert_allclose(parent_predict(parent, raw)[0], predict_raw(child, raw)[0], atol=1e-5)
        self.assertTrue(torch.equal(child.project.weight[:, 5:10], torch.zeros((16, 5))))
        x, pad = multiscale_tokens(raw, [100])
        child(torch.from_numpy(x), torch.from_numpy(pad)).sum().backward()
        self.assertGreater(float(abs(child.project.weight.grad[:, 5:10]).sum()), 0.)
        self.assertIsNone(parent.project.weight.grad)
        self.assertNotEqual(child.encoder.layers[0].linear1.weight.data_ptr(), parent.encoder.layers[0].linear1.weight.data_ptr())

    def test_localized_parent_cannot_be_mislabelled_as_broad_parent(self):
        parent = JointTransformer(JointConfig(scope_same_run=True))
        with self.assertRaises(ValueError):expand_parent(parent)


if __name__ == '__main__':unittest.main()
