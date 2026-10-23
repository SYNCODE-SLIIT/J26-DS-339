import unittest
import numpy as np
from bpdr_input.phase_feasibility import analytic_fundamental,mixing_arrays,ramp_metrics


class PhaseDiagnosticTests(unittest.TestCase):
    def test_phasor_controls_detect_cancellation_and_alignment(self):
        z=np.ones(3,dtype=complex);w=np.full(3,.5)
        for offset,ratio in ((0.,1.),(np.pi,0.),(np.pi/2,np.sqrt(.5))):
            other=z*np.exp(1j*offset)
            result=mixing_arrays(z,other,(1-w)*z+w*other,w)
            np.testing.assert_allclose(result['algebraic_ratio'],ratio,rtol=0,atol=1e-12)
            np.testing.assert_allclose(result['measured_ratio'],ratio,rtol=0,atol=1e-12)

    def test_known_harmonic_signal_has_stable_fundamental_phase_difference(self):
        sr=22050;t=np.arange(sr*4)/sr;phase=2*np.pi*220*t
        source=np.sin(phase)+.4*np.sin(2*phase);offset=.6
        other=np.sin(phase+offset)+.4*np.sin(2*(phase+offset))
        a=analytic_fundamental(source,sr,220);b=analytic_fundamental(other,sr,220)
        interior=(t>1.)&(t<3.)
        np.testing.assert_allclose(np.angle(b[interior]*np.conj(a[interior])),offset,rtol=0,atol=.001)

    def test_silent_branches_are_withheld_not_reported_as_phase_support(self):
        t=np.linspace(1.,1.08,100);z=np.zeros(len(t),dtype=complex)
        arrays=mixing_arrays(z,z,z,np.full(len(t),.5))
        result=ramp_metrics(arrays,t,1.,2.,'entry',.01)
        self.assertTrue(result['withheld']);self.assertEqual(result['valid_samples'],0)
        self.assertTrue(np.isnan(arrays['measured_ratio']).all())

    def test_invalid_shapes_weights_and_known_center_refuse(self):
        z=np.ones(3,dtype=complex)
        for w in (np.ones(2),np.full(3,1.2),np.full(3,np.nan)):
            with self.assertRaises(ValueError):mixing_arrays(z,z,z,w)
        with self.assertRaises(ValueError):analytic_fundamental(np.ones(100),22050,20000)
