import unittest
from unittest.mock import patch
import numpy as np
from bpdr_input.phase_renderer import polar_ramp,render_phase_candidate


class PhaseRendererTests(unittest.TestCase):
    def setUp(self):
        self.sr=22050;t=np.arange(4*self.sr)/self.sr
        self.audio=(.1*np.sin(2*np.pi*220*t)).astype(np.float32)
        self.times=np.arange(len(self.audio)//441+1)*.02
        self.curve=np.where((self.times>=1.)&(self.times<2.),20.,0.)

    def test_polar_join_removes_opposed_phasor_amplitude_cancellation(self):
        z=np.ones(101,dtype=complex);w=np.linspace(0,1,101)
        joined,_=polar_ramp(z,-z,w)
        np.testing.assert_allclose(abs(joined),1,rtol=0,atol=1e-12)
        self.assertAlmostEqual(abs(((1-w)*z+w*(-z))[50]),0.)

    def test_relative_phase_wrap_does_not_create_a_discontinuous_join(self):
        phase=np.linspace(3.,3.3,100);z=np.ones(100,dtype=complex)
        joined,gap=polar_ramp(z,np.exp(1j*phase),np.full(100,.5))
        self.assertLess(np.max(abs(np.diff(gap))),.01)
        self.assertLess(np.max(abs(np.diff(joined))),.01)

    def test_zero_edit_is_exact_and_bypasses_native_and_phase_backends(self):
        with (patch('bpdr_input.phase_renderer.private_resynthesis',side_effect=AssertionError('No backend')),
                patch('bpdr_input.phase_renderer.hilbert',side_effect=AssertionError('No phase processing'))):
            output,diag,branches=render_phase_candidate(self.audio,self.sr,self.times,np.zeros(len(self.times)),[])
        np.testing.assert_array_equal(output,self.audio);self.assertEqual(branches,{})
        self.assertTrue(diag['identity']);self.assertFalse(diag['production_eligible'])

    def test_original_over_limit_and_outside_requests_are_not_hidden(self):
        with patch('bpdr_input.phase_renderer.private_resynthesis') as backend:
            for curve in (self.curve*2,np.full(len(self.times),20.)):
                with self.assertRaises(ValueError):render_phase_candidate(self.audio,self.sr,self.times,curve,[(1.,2.)])
            backend.assert_not_called()

    def test_actual_render_copies_outside_and_resynthesized_interior_exactly(self):
        output,diag,b=render_phase_candidate(self.audio,self.sr,self.times,self.curve,[(1.,2.)])
        np.testing.assert_array_equal(output[b['blend']==0],self.audio[b['blend']==0])
        np.testing.assert_array_equal(output[b['blend']==1],b['baseline_local'][b['blend']==1])
        self.assertEqual(output.shape,self.audio.shape);self.assertTrue(diag['interior_samples_exact'])
        self.assertFalse(diag['duration_tier_modified']);self.assertIsNone(diag['accepted_audio'])

    def test_zero_amplitude_and_invalid_weights_refuse_phase_processing(self):
        z=np.ones(3,dtype=complex)
        for source,weight in ((z*0,np.full(3,.5)),(z,np.full(3,np.nan)),(z,np.full(3,1.1))):
            with self.assertRaises(ValueError):polar_ramp(source,z,weight)
