import unittest
import numpy as np
from bpdr_input.framewise_audit import audit_curve


class FramewiseAuditTests(unittest.TestCase):
    def setUp(self):
        self.times=np.arange(0.,4.02,.02);n=len(self.times)
        self.before=dict(times_seconds=self.times,f0_hz=np.full(n,220.),voiced_mask=np.ones(n,dtype=bool))
        self.curve=np.where((self.times>=1.)&(self.times<2.),-30.+60.*(self.times-1.),0.)
        self.yin=np.full(n,220.)

    def measure(self,curve):
        after=dict(times_seconds=self.times,f0_hz=220.*2**(curve/1200),voiced_mask=np.ones(len(self.times),dtype=bool))
        return audit_curve(self.before,after,self.yin,after['f0_hz'],self.before['voiced_mask'],[(1.,2.)],self.curve)

    def test_per_frame_target_accepts_matching_sign_changing_curve(self):
        result=self.measure(self.curve)
        self.assertTrue(result['passed']);self.assertGreater(result['direction_checked_frames'],0)
        self.assertAlmostEqual(result['fine']['p90_error_cents'],0.,places=8)

    def test_constant_shift_cannot_hide_wrong_direction_in_changing_curve(self):
        wrong=np.where(self.curve!=0,20.,0.)
        result=self.measure(wrong)
        self.assertFalse(result['passed']);self.assertGreater(result['fine_wrong_direction_frames'],0)


if __name__=='__main__':unittest.main()
