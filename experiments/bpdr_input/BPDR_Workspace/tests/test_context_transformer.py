"""Invariants that protect time alignment and single-recording inference."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bpdr_input.modeling.context_transformer import (ContextConfig, ContextTransformer,
    load_checkpoint, predict_features, save_checkpoint, window_at)


class ContextTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(339)
        self.config = ContextConfig(hidden=16, layers=1, feed_forward=32, window=11)

    def test_window_keeps_center_and_padding(self):
        x = np.arange(5*32, dtype=np.float32).reshape(5, 32)
        w, padding, positions = window_at(x, 0, 11)
        np.testing.assert_array_equal(w[5], x[0])
        np.testing.assert_array_equal(padding, [True]*5+[False]*5+[True])
        np.testing.assert_array_equal(positions, np.arange(-5, 6))

    def test_inference_matches_exact_center_windows(self):
        model = ContextTransformer(self.config).eval()
        torch.nn.init.normal_(model.encoder.head.weight, std=.01)
        x = np.random.default_rng(1).normal(size=(19, 32)).astype(np.float32)
        actual = predict_features(model, x, batch_size=7)
        expected = []
        with torch.inference_mode():
            for i in range(len(x)):
                w, pad, _ = window_at(x, i, 11)
                expected.append(model(torch.from_numpy(w[None]), torch.from_numpy(pad[None]))[0, 5].item())
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_checkpoint_roundtrip_and_signed_output(self):
        model = ContextTransformer(self.config).eval()
        model.encoder.head.bias.data.fill_(-.5)
        x = np.zeros((3, 32), dtype=np.float32)
        parent = Path(__file__).resolve().parents[1]/'results/transformer_recovery'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            path = Path(folder)/"candidate.pt"
            save_checkpoint(path, model, dict(eligible_for_demo=False))
            restored, payload = load_checkpoint(path)
            np.testing.assert_allclose(predict_features(restored, x), -15., atol=1e-5)
            self.assertFalse(payload['eligible_for_demo'])

    def test_rejects_invalid_features(self):
        model = ContextTransformer(self.config)
        with self.assertRaises(ValueError):
            predict_features(model, np.zeros((10, 31)))
        with self.assertRaises(ValueError):
            predict_features(model, np.full((10, 32), np.nan))


if __name__ == '__main__':
    unittest.main()
