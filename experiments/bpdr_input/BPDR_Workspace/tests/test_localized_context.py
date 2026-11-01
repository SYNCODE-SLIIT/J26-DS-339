from dataclasses import replace
import unittest
import numpy as np

from bpdr_input.modeling.joint_transformer import JointConfig, joint_tokens


class LocalizedContextTests(unittest.TestCase):
    def test_neighboring_note_cannot_reverse_current_note_context(self):
        raw = np.zeros((501, 71), np.float32)
        raw[:, 65:67] = 1.
        raw[:, 68] = -20.
        raw[200:301, 64] = 200.
        raw[225:276, 64] += 20.
        changed = raw.copy()
        changed[:200, 64] = 900.
        changed[301:, 64] = 900.
        config = replace(JointConfig(), scope_same_run=True)
        x, pad = joint_tokens(raw, [250], config)
        y, other_pad = joint_tokens(changed, [250], config)
        np.testing.assert_allclose(x, y, atol=1e-5)
        np.testing.assert_array_equal(pad, other_pad)
        self.assertAlmostEqual(float(x[0, 25, 0])*30., 20., delta=1e-4)
        broad, _ = joint_tokens(raw, [250])
        other_broad, _ = joint_tokens(changed, [250])
        self.assertNotEqual(np.sign(broad[0, 25, 0]), np.sign(other_broad[0, 25, 0]))


if __name__ == '__main__':
    unittest.main()
