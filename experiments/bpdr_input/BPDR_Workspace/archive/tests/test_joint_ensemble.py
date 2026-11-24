import unittest
import numpy as np
from bpdr_input.modeling.joint_ensemble import combine
from bpdr_input.repair.ensemble_policy import conditioned_proposal
from bpdr_input.repair.held_voice_policy import HeldPolicy


class EnsembleTests(unittest.TestCase):
    def test_median_envelope_permutation_and_identity(self):
        x=np.array([[0,20],[0,21],[0,-20]],np.float32)
        expected=combine(x)
        np.testing.assert_array_equal(expected,combine(x[[2,0,1]]))
        np.testing.assert_array_equal(expected[:,0],0.)
        np.testing.assert_array_equal(expected[:,1],[20,-20,21])

    def test_invalid_members_refused_without_clipping(self):
        for x in (np.zeros((2,10)),np.full((3,10),31.),np.full((3,10),np.nan)):
            with self.assertRaises(ValueError):combine(x)

    def test_full_range_veto_and_exact_primary_support(self):
        audio=np.zeros(220500,np.float32);times=np.arange(501)*.02
        raw=np.zeros((501,71),np.float32);raw[:,65:67]=1.
        pair=dict(audio_C=audio,C=dict(features={'times_seconds':times},views=[raw]*3))
        members=np.zeros((3,501),np.float32);members[:,80:140]=np.array([18,20,22])[:,None]
        curve,plan,_=conditioned_proposal(pair,'C',combine(members),HeldPolicy(activation_cents=8.),'median11')
        self.assertTrue(plan['permitted']);self.assertFalse(plan['identity'])
        np.testing.assert_array_equal(np.flatnonzero(curve),np.arange(80,140))
        members[:,80:140]=np.array([15,20,25])[:,None]
        _,bad,_=conditioned_proposal(pair,'C',combine(members),HeldPolicy(activation_cents=8.),'median11')
        self.assertFalse(bad['permitted']);self.assertIn('UNCERTAINTY_LIMIT',bad['reasons'])
        self.assertEqual(bad['intervals_seconds'],[])


if __name__=='__main__':unittest.main()
