import unittest
import numpy as np
from bpdr_input.modeling.fcpe_measurement import align_native


class MeasurementAlignmentTests(unittest.TestCase):
    def test_real_frames_ties_and_no_extrapolation(self):
        times = np.array([.005, .015, .025, .035])
        output, valid, indices = align_native(times, [100., 200., 300., 400.], [0., .01, .02, .04])
        np.testing.assert_array_equal(valid, [False, True, True, False])
        np.testing.assert_array_equal(output[valid], [100., 200.])
        np.testing.assert_array_equal(indices[1:3], [0, 1])

    def test_unvoiced_and_out_of_range_remain_missing_without_clamping(self):
        output, valid, _ = align_native([.005, .015, .025, .035], [220., 0., 1600., 220.], [.005, .015, .025, .035])
        np.testing.assert_array_equal(valid, [True, False, False, True])
        self.assertTrue(np.isnan(output[1:3]).all())

    def test_invalid_grid_refused(self):
        with self.assertRaises(ValueError):
            align_native([.01, .01], [220., 220.], [.01])


if __name__ == '__main__':
    unittest.main()
