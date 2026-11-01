import unittest
import numpy as np
from bpdr_input.modeling.step_candidates import measured_candidates,candidate_context
from bpdr_input.experiments.learning.candidate_labels import label_candidate


class StepCandidateTests(unittest.TestCase):
    def test_geometry_and_features_ignore_boundary_metadata(self):
        raw=np.zeros((350,71),np.float32);raw[:,65:67]=1.;raw[100:170,64]=20.
        raw[1:,67]=np.diff(raw[:,64])
        candidates=measured_candidates(raw);self.assertTrue(candidates)
        altered=raw.copy();altered[:,69:71]=np.random.default_rng(5).normal(size=(350,2))
        self.assertEqual(candidates,measured_candidates(altered))
        for interval in candidates:
            for a,b in zip(candidate_context(raw,interval),candidate_context(altered,interval)):
                np.testing.assert_array_equal(a,b)
            self.assertGreaterEqual(interval[1]-interval[0],43)

    def test_missing_mixed_and_unknown_labels_are_not_fabricated(self):
        target=np.full(60,20.);known=np.ones(60,bool)
        self.assertEqual(label_candidate((0,60),target,known),20.)
        target[:]=0.;self.assertEqual(label_candidate((0,60),target,known),0.)
        target[20:40]=20.;self.assertIsNone(label_candidate((0,60),target,known))
        target[:]=20.;known[:40]=False;self.assertIsNone(label_candidate((0,60),target,known))

    def test_unvoiced_identity_and_short_candidate_refusal(self):
        raw=np.zeros((100,71),np.float32)
        self.assertEqual(measured_candidates(raw),[])
        with self.assertRaises(ValueError):candidate_context(raw,(20,62))


if __name__=='__main__':unittest.main()
