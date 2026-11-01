import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from bpdr_input.modeling.joint_transformer import joint_tokens
from bpdr_input.modeling.sequence_alternatives import SequenceAlternative, predict_raw, save, load


class AlternativesTests(unittest.TestCase):
    def test_encoders_gradients_metadata_and_reload(self):
        torch.set_num_threads(2)
        raw = np.zeros((80, 71), np.float32)
        raw[:, 65:67] = 1.
        raw[:, 64] = np.sin(np.arange(80) / 10.) * 10.
        raw[:2, 66] = 0.
        for family in ('bigru', 'tcn'):
            with self.subTest(family=family):
                torch.manual_seed(339)
                model = SequenceAlternative(family)
                with torch.no_grad():
                    model.regression.weight.normal_(std=.01)
                x, pad = joint_tokens(raw, np.array([0, 40, 79]), model.config)
                output, logit, amp = model(torch.from_numpy(x), torch.from_numpy(pad), True)
                (output.square().mean() + logit.square().mean()).backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()))
                self.assertTrue(torch.all(amp.abs() <= 30.))
                expected = predict_raw(model, raw)[0]
                np.testing.assert_array_equal(expected[:2], 0.)
                modified = raw.copy(); modified[:, 69:71] = 999.
                np.testing.assert_array_equal(expected, predict_raw(model, modified)[0])
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'model.pt'
                    save(path, model, dict(seed=339))
                    restored, _ = load(path)
                    np.testing.assert_array_equal(expected, predict_raw(restored, raw)[0])


if __name__ == '__main__':
    unittest.main()
