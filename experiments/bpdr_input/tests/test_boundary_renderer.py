import unittest
from unittest.mock import patch
import numpy as np
from bpdr_input.boundary_renderer import render_boundary_candidate


class BoundaryRendererTests(unittest.TestCase):
    def setUp(self):
        self.sr=22050;t=np.arange(self.sr*4)/self.sr
        self.audio=(.1*np.sin(2*np.pi*220*t)).astype(np.float32)
        self.times=np.arange(len(self.audio)//441+1)*.02
        self.curve=np.where((self.times>=1.)&(self.times<2.),20.,0.)

    def test_original_outside_and_over_limit_requests_cannot_be_hidden_by_taper(self):
        outside=self.curve.copy();outside[0]=20.
        over=self.curve.copy();over[50]=60. # Taper is zero exactly at this support start.
        with patch('bpdr_input.boundary_renderer.render_candidate') as backend:
            for curve in (outside,over):
                with self.assertRaises(ValueError):render_boundary_candidate(self.audio,self.sr,self.times,curve,[(1.,2.)])
            backend.assert_not_called()

    def test_zero_request_bypasses_backend_and_retains_exact_samples(self):
        with patch('parselmouth.Sound',side_effect=AssertionError('Zero edit must not resynthesize')):
            output,diag,tapered=render_boundary_candidate(self.audio,self.sr,self.times,np.zeros(len(self.times)),[])
        np.testing.assert_array_equal(output,self.audio);self.assertFalse(tapered.any())
        self.assertTrue(diag['identity']);self.assertIsNone(diag['accepted_audio'])

    def test_measured_render_retains_target_in_guarded_interior_and_outside_samples(self):
        output,diag,tapered=render_boundary_candidate(self.audio,self.sr,self.times,self.curve,[(1.,2.)])
        interior=(self.times>=1.18)&(self.times<=1.82)
        np.testing.assert_array_equal(tapered[interior],self.curve[interior])
        self.assertEqual(tapered[50],0.);self.assertGreater(tapered[51],0.);self.assertLess(tapered[51],20.)
        outside=(np.arange(len(self.audio))/self.sr<=1.)|(np.arange(len(self.audio))/self.sr>=2.)
        np.testing.assert_array_equal(output[outside],self.audio[outside])
        self.assertEqual(output.shape,self.audio.shape)
        self.assertEqual(diag['source_time_origin_seconds'],diag['output_time_origin_seconds'])
        self.assertFalse(diag['duration_tier_modified']);self.assertFalse(diag['production_eligible'])

    def test_invalid_original_grid_nonfinite_or_supports_refuse_before_rendering(self):
        with patch('bpdr_input.boundary_renderer.render_candidate') as backend:
            for times,curve,support in ((self.times+.01,self.curve,[(1.,2.)]),
                    (self.times,np.full(len(self.times),np.nan),[(1.,2.)]),
                    (self.times,self.curve,[(1.,1.5)]),
                    (self.times,self.curve,[(1.,2.),(1.5,2.5)])):
                with self.assertRaises(ValueError):render_boundary_candidate(self.audio,self.sr,times,curve,support)
            backend.assert_not_called()
