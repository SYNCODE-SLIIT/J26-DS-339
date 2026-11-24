"""Candidate renderer mechanics: identity, limits, support and timing preservation."""
import unittest
import numpy as np
from bpdr_input.framewise_renderer import render_candidate


class FramewiseRendererTests(unittest.TestCase):
    def setUp(self):
        self.sr=22050;t=np.arange(4*self.sr)/self.sr
        self.audio=(.1*np.sin(2*np.pi*220*t)).astype(np.float32)
        self.times=np.arange(len(self.audio)//441+1)*.02
        self.curve=np.where((self.times>=1.)&(self.times<1.9),20.,0.)

    def test_zero_curve_is_exact_identity(self):
        result,diagnostic=render_candidate(self.audio,self.sr,self.times,np.zeros(len(self.times)),[])
        self.assertTrue(np.array_equal(result,self.audio));self.assertTrue(diagnostic['identity'])
        self.assertNotIn('praat_version',diagnostic)

    def test_over_limit_nonfinite_and_outside_curve_are_refused(self):
        for curve in (self.curve*2,np.full(len(self.times),np.nan),np.full(len(self.times),20.)):
            with self.assertRaises(ValueError):render_candidate(self.audio,self.sr,self.times,curve,[(1.,1.9)])

    def test_short_overlapping_and_out_of_domain_support_are_refused(self):
        for supports in ([(1.,1.5)],[(1.,2.),(1.5,2.5)],[(-1.,1.)],[(3.,5.)]):
            with self.assertRaises(ValueError):render_candidate(self.audio,self.sr,self.times,self.curve,supports)

    def test_actual_backend_preserves_count_origin_and_outside_samples(self):
        result,diagnostic=render_candidate(self.audio,self.sr,self.times,self.curve,[(1.,1.9)])
        outside=(np.arange(len(self.audio))/self.sr<1.)|(np.arange(len(self.audio))/self.sr>=1.9)
        self.assertEqual(result.shape,self.audio.shape)
        self.assertTrue(np.array_equal(result[outside],self.audio[outside]))
        self.assertEqual(diagnostic['source_time_origin_seconds'],diagnostic['output_time_origin_seconds'])
        self.assertFalse(diagnostic['duration_tier_modified']);self.assertFalse(diagnostic['production_eligible'])


if __name__=='__main__':unittest.main()
