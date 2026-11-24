import unittest
import numpy as np
from bpdr_input.experiments.learning.virtual_short_examples import make_pair


class VirtualShortExamplesTest(unittest.TestCase):
    def raw(self):
        r=np.zeros((300,71),np.float32);r[:,66]=1;r[:,65]=.9;r[:,64]=np.linspace(-70,70,300);r[10:20,66]=0
        r[10:20,64]=0.;return r

    def test_counterfactual_difference_and_shared_expression(self):
        r=self.raw();a,c,m=make_pair(r,left_seconds=2.,width_seconds=.24,cents=20.,vibrato_amplitude_cents=15.,vibrato_hz=6.,phase=.4,drift_cents=10.)
        voiced=r[:,66]>.5;d=c[:,64]-a[:,64];d-=np.median(d[m['zero']])
        np.testing.assert_allclose(d[voiced],20*m['pulse'][voiced],atol=2e-5,rtol=0)
        self.assertFalse(m['physical_waveform']);self.assertFalse(np.any(m['positive']&m['zero']))
        np.testing.assert_array_equal(m['target'][m['positive']],-20.)

    def test_unvoiced_and_other_channels_are_not_fabricated(self):
        r=self.raw();a,c,m=make_pair(r,left_seconds=2.,width_seconds=.5,cents=-30.)
        for x in [a,c]:
            np.testing.assert_array_equal(x[r[:,66]<=.5,64],0.)
            np.testing.assert_array_equal(x[:,[65,66,68,69,70]],r[:,[65,66,68,69,70]])
        self.assertTrue(np.isfinite(a).all() and np.isfinite(c).all())

    def test_invalid_virtual_support_is_rejected(self):
        with self.assertRaises(ValueError):make_pair(self.raw(),left_seconds=5.9,width_seconds=.5,cents=20.)
        with self.assertRaises(ValueError):make_pair(self.raw(),left_seconds=1.,width_seconds=.5,cents=50.)


if __name__=='__main__':unittest.main()
