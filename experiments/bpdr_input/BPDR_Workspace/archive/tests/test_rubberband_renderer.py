import tempfile
from pathlib import Path
import unittest
import numpy as np
from bpdr_input.repair.rubberband_renderer import render_candidate,verify_dependency
from bpdr_input.repair.rule_correction import fine_hz


class RubberBandRendererTests(unittest.TestCase):
    def setUp(self):
        self.audio=(.1*np.sin(2*np.pi*220*np.arange(110250)/22050)).astype(np.float32)
        self.times=np.arange(251)*.02

    def test_actual_both_directions_and_exact_outside(self):
        before=fine_hz(self.audio,22050,self.times)
        sample_times=np.arange(len(self.audio))/22050
        outside=(sample_times<2.)|(sample_times>=3.2)
        scored=(self.times>2.3)&(self.times<2.9)
        for cents in (-20.,20.):
            curve=np.where((self.times>=2.)&(self.times<3.2),cents,0.)
            output,report=render_candidate(self.audio,22050,self.times,curve,[(2.,3.2)])
            self.assertEqual(output.shape,self.audio.shape)
            np.testing.assert_array_equal(output[outside],self.audio[outside])
            measured=1200*np.log2(fine_hz(output,22050,self.times)[scored]/before[scored])
            self.assertAlmostEqual(float(np.median(measured)),cents,delta=3.)
            self.assertTrue(report['auto_gain_disabled']);self.assertFalse(report['production_eligible'])

    def test_identity_bad_curve_and_unpinned_dependency(self):
        output,_=render_candidate(self.audio,22050,self.times,np.zeros(251),[])
        np.testing.assert_array_equal(output,self.audio)
        for curve,intervals in [(np.full(251,31.),[(0.,5.)]),
                (np.where((self.times>=2.)&(self.times<3.2),20.+np.sin(np.arange(251)),0.),[(2.,3.2)]),
                (np.where((self.times>=2.)&(self.times<2.4),20.,0.),[(2.,2.4)])]:
            with self.assertRaises(ValueError):render_candidate(self.audio,22050,self.times,curve,intervals)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'rubberband-r3.exe').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_dependency(path)


if __name__=='__main__':unittest.main()
