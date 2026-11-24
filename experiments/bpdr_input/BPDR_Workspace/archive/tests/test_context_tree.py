import tempfile
import unittest
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from bpdr_input.modeling.context_tree import features, predict_raw, save, load


class ContextTreeTests(unittest.TestCase):
    def test_input_metadata_bounds_identity_and_reload(self):
        raw = np.zeros((80, 71), np.float32); raw[:, 65:67] = 1.
        raw[:, 64] = np.sin(np.arange(80) / 10.) * 10.
        centers = np.arange(80); x = features(raw, centers)
        self.assertEqual(x.shape, (80, 459))
        altered = raw.copy(); altered[:, 69:71] = 999.
        np.testing.assert_array_equal(x, features(altered, centers))
        model = ExtraTreesRegressor(n_estimators=4, random_state=339, n_jobs=1).fit(x, np.linspace(-1, 1, len(x)))
        raw[:3, 66] = 0.; expected = predict_raw(model, raw)[0]
        self.assertTrue(np.all(abs(expected) <= 30.))
        np.testing.assert_array_equal(expected[:3], 0.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'; save(path, model, dict(seed=339))
            restored, _ = load(path)
            np.testing.assert_array_equal(expected, predict_raw(restored, raw)[0])


if __name__ == '__main__':
    unittest.main()
