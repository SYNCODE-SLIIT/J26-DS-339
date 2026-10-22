"""Validate independent pitch timing and low-tone behavior."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from melody_understanding.audio import PreparedAudio
from melody_understanding.review import map_pitch_frames,praat_pitch

# Grouping Related Contract Checks
class PitchReviewTests(unittest.TestCase):
    # Leave unsupported edges and unvoiced source frames missing after nearest-frame mapping.
    def test_mapping_preserves_unvoiced_gaps_and_edges(self):
        hz,support=map_pitch_frames(np.array([.1,.2,.3]),np.array([110.,0.,220.]),np.array([0.,.1,.2,.3,.4]),.05)
        np.testing.assert_array_equal(support,[False,True,True,True,False])
        self.assertTrue(np.isnan(hz[[0,2,4]]).all())
        np.testing.assert_allclose(hz[[1,3]],[110,220])

    # Check Praat recovers a known 110 Hz tone away from analysis-window edges.
    def test_independent_estimator_recovers_low_tone(self):
        sr=22050;t=np.arange(sr)/sr
        audio=PreparedAudio(.2*np.sin(2*np.pi*110*t),sr,1.,0.,{})
        targets=np.arange(0,sr,256)/sr
        pitch,supported=praat_pitch(audio,targets)
        mask=supported&(targets>.2)&(targets<.8)&pitch['voiced']
        self.assertGreater(mask.sum(),40)
        self.assertLess(abs(1200*np.log2(np.median(pitch['f0_hz'][mask])/110)),10)
