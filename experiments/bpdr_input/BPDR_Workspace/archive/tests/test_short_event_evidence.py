import unittest
import numpy as np
from bpdr_input.modeling.short_event_evidence import evidence


class ShortEventEvidenceTest(unittest.TestCase):
    def raw(self,pitch):
        r=np.zeros((len(pitch),71),np.float32);r[:,64]=pitch;r[:,66]=1;return r

    def test_flat_and_linear_glides_do_not_create_events(self):
        for pitch in [np.zeros(80),np.linspace(-25,25,80),np.full(80,200.)]:
            np.testing.assert_array_equal(evidence(self.raw(pitch)),0.)

    def test_short_signed_event_is_measured_without_a_reference(self):
        for sign in [-1,1]:
            pulse=np.zeros(40);pulse[14:26]=1
            pitch=sign*20*np.convolve(pulse,[.1,.2,.4,.2,.1],mode='same')
            f,report=evidence(self.raw(pitch),details=True)
            self.assertTrue(report['fits']);self.assertFalse(report['permission_granted'])
            self.assertAlmostEqual(float(f[20,0]*30),-sign*20,delta=2.)
            self.assertLessEqual(report['fits'][0]['pulse_stop']-report['fits'][0]['pulse_start'],14)

    def test_periodic_vibrato_is_not_a_single_local_pulse(self):
        t=np.arange(100)*.02
        np.testing.assert_array_equal(evidence(self.raw(20*np.sin(2*np.pi*5.5*t))),0.)

    def test_no_missing_pitch_is_exposed_as_reliable_evidence(self):
        r=self.raw(np.r_[np.zeros(14),np.full(12,20),np.zeros(14)])
        r[20,66]=0
        np.testing.assert_array_equal(evidence(r)[20],0.)

    def test_metadata_and_transposition_do_not_change_evidence(self):
        r=self.raw(np.r_[np.zeros(14),np.full(12,20),np.zeros(14)])
        other=r.copy();other[:,64]+=300;other[:,:64]=7;other[:,69:]=3
        np.testing.assert_allclose(evidence(r),evidence(other),atol=1e-6,rtol=0)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):evidence(np.zeros((0,71)))
        r=self.raw(np.zeros(40));r[0,64]=np.nan
        with self.assertRaises(ValueError):evidence(r)


if __name__=='__main__':unittest.main()
