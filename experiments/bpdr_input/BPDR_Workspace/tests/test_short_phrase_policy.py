import unittest
import numpy as np
from bpdr_input.repair.short_phrase_policy import proposal
from bpdr_input.repair.repair_controller import FixturePolicy


class ShortPolicyTest(unittest.TestCase):
    def inputs(self,width=25,cents=20.):
        audio=np.zeros(6*22050,np.float32);times=np.arange(301)*.02
        raw=np.zeros((301,71),np.float32);raw[:,65]=.9;raw[:,66]=1.;raw[:,68]=-20
        p=np.zeros((3,301));p[:,100:100+width]=cents
        return audio,times,[raw.copy() for _ in range(3)],p

    def test_short_policy_is_opt_in_and_bounded(self):
        args=self.inputs();r=proposal(*args)
        self.assertTrue(r['plan']['permitted']);self.assertFalse(r['production_eligible'])
        self.assertEqual(FixturePolicy().min_support_seconds,.86)
        self.assertLessEqual(r['plan']['affected_fraction'],.25)

    def test_user_protection_vetoes_entire_candidate(self):
        args=self.inputs();mask=np.zeros(301,bool);mask[110]=True
        r=proposal(*args,user_protected=mask)
        self.assertTrue(r['plan']['identity']);self.assertEqual(r['candidates'][0]['reason'],'PROTECTED_REGION')
        np.testing.assert_array_equal(r['curve'],0.)

    def test_overlimit_and_disagreement_are_never_clipped(self):
        args=self.inputs(cents=31.);r=proposal(*args)
        self.assertFalse(r['plan']['permitted']);self.assertIn('CORRECTION_LIMIT',r['plan']['reasons'])
        args=self.inputs();args[3][1,100:125]+=6
        r=proposal(*args);self.assertFalse(r['plan']['permitted'])
        self.assertIn('BOUNDARY_DISAGREEMENT',r['plan']['reasons'])

    def test_no_padding_of_short_or_unreliable_supports(self):
        r=proposal(*self.inputs(width=10));self.assertTrue(r['plan']['identity'])
        args=self.inputs();args[2][0][110,66]=0
        r=proposal(*args);self.assertTrue(r['plan']['identity'])


if __name__=='__main__':unittest.main()
