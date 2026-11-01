import unittest
import numpy as np
from bpdr_input.modeling.candidate_center_projection import project
from bpdr_input.modeling.step_candidates import measured_candidates


class CandidateCenterProjectionTests(unittest.TestCase):
    def test_uses_learned_center_only_in_input_measured_regions(self):
        raw=np.zeros((350,71),np.float32);raw[:,65:67]=1.;raw[100:170,64]=20.
        values=np.linspace(-20,20,350,dtype=np.float32);out=project(raw,values)
        covered=np.zeros(len(raw),bool)
        for start,stop in measured_candidates(raw):
            covered[start:stop]=True
            np.testing.assert_array_equal(out[start:stop],values[(start+stop-1)//2])
        self.assertTrue(covered.any());np.testing.assert_array_equal(out[~covered],values[~covered])
        altered=raw.copy();altered[:,69:71]=77.
        np.testing.assert_array_equal(out,project(altered,values))

    def test_no_candidates_identity_and_invalid_predictions(self):
        raw=np.zeros((100,71),np.float32);raw[:,65:67]=1.
        values=np.linspace(-10,10,100,dtype=np.float32)
        np.testing.assert_array_equal(project(raw,values),values)
        with self.assertRaises(ValueError):project(raw,np.full(100,31.))


if __name__=='__main__':unittest.main()
