import unittest
import numpy as np
from bpdr_input.repair.experimental_short_audit import audit_short_curve
from bpdr_input.repair.interventions import AuditPolicy


class ExperimentalShortAuditTest(unittest.TestCase):
    def fixture(self):
        times=np.arange(151)*.02;n=len(times);base=np.full(n,220.)
        request=np.where((times>=1.)&(times<1.24),20.,0.)
        shifted=base*2**(request/1200)
        before=dict(times_seconds=times,f0_hz=base,voiced_mask=np.ones(n,bool))
        after=dict(times_seconds=times,f0_hz=shifted,voiced_mask=np.ones(n,bool))
        return [before,after,base,shifted,base.copy(),shifted.copy(),np.ones(n,bool),[(1.,1.24)],request]

    def test_requires_three_corroborated_observations(self):
        r=audit_short_curve(*self.fixture())
        self.assertTrue(r['passed']);self.assertEqual(r['cc']['frames'],3)
        self.assertFalse(r['production_eligible']);self.assertEqual(AuditPolicy().min_audit_frames,10)

    def test_third_tracker_opposition_is_rejected(self):
        args=self.fixture();args[5]=args[4]*2**(-args[8]/1200)
        r=audit_short_curve(*args)
        self.assertFalse(r['passed']);self.assertIn('CC_DIRECTION',r['reasons'])

    def test_missing_third_tracker_does_not_reduce_the_denominator(self):
        args=self.fixture();times=args[0]['times_seconds'];args[5][(times>=1.09)&(times<=1.15)]=np.nan
        r=audit_short_curve(*args)
        self.assertFalse(r['passed']);self.assertIn('CC_COMMON_FRAMES',r['reasons']);self.assertIn('CC_COVERAGE',r['reasons'])

    def test_wrong_requested_direction_and_zero_change_are_rejected(self):
        args=self.fixture();args[8]=-args[8]
        self.assertFalse(audit_short_curve(*args)['passed'])
        args=self.fixture();args[1]['f0_hz']=args[0]['f0_hz'];args[3]=args[2];args[5]=args[4]
        self.assertFalse(audit_short_curve(*args)['passed'])

    def test_grid_mismatch_is_rejected(self):
        args=self.fixture();args[5]=args[5][:-1]
        with self.assertRaises(ValueError):audit_short_curve(*args)


if __name__=='__main__':unittest.main()
