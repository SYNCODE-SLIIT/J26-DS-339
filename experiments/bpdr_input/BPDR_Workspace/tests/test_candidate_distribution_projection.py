import unittest
import numpy as np
from bpdr_input.modeling.candidate_distribution_projection import project,representative_indices,apply_indices
from bpdr_input.modeling.step_candidates import measured_candidates


class DistributionReadoutTests(unittest.TestCase):
    def setUp(self):
        self.raw=np.zeros((350,71),np.float32);self.raw[:,65:67]=1.;self.raw[100:170,64]=20.

    def test_observed_ranks_sign_symmetry_and_dense_fallback(self):
        values=np.linspace(10,25,350,dtype=np.float32);covered=np.zeros(350,bool)
        for a,b in measured_candidates(self.raw):covered[a:b]=True
        self.assertTrue(covered.any())
        for mode in ('median','upper_quartile'):
            result=project(self.raw,values,mode)
            np.testing.assert_array_equal(result[~covered],values[~covered])
            np.testing.assert_array_equal(project(self.raw,-values,mode),-result)
            for a,b in measured_candidates(self.raw):
                rank=int(np.rint((.5 if mode=='median' else .75)*(b-a-1)))
                self.assertTrue(np.all(result[a:b]==np.sort(values[a:b])[rank]))
                self.assertIn(result[a],values[a:b])

    def test_associated_activity_and_input_metadata_invariance(self):
        values=np.linspace(10,25,350,dtype=np.float32);activity=np.linspace(0,1,350,dtype=np.float32)
        indices=representative_indices(self.raw,values,'upper_quartile');result=apply_indices(activity,indices)
        for a,b in measured_candidates(self.raw):self.assertTrue(np.all(result[a:b]==activity[indices[a]]))
        altered=self.raw.copy();altered[:,69:71]=99.
        np.testing.assert_array_equal(indices,representative_indices(altered,values,'upper_quartile'))

    def test_invalid_predictions_refused_without_clipping(self):
        with self.assertRaises(ValueError):project(self.raw,np.full(350,31.),'median')
        with self.assertRaises(ValueError):project(self.raw,np.zeros(350),'unknown')


if __name__=='__main__':unittest.main()
