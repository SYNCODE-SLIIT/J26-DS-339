"""Check label semantics, uncertainty masks and overlap rejection."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from melody_understanding.targets import score_targets

# Grouping Related Contract Checks
class ScoreTargetTests(unittest.TestCase):
    # Check shared pitch-class labels across octaves, explicit rests and excluded note edges.
    def test_octave_invariance_rest_and_boundaries(self):
        t=np.array([.1,.3,.5,.7,.9,1.1,1.3,1.5,1.7,1.9])
        labels,keep=score_targets([(.2,.8,60),(1.2,1.8,72)],t,2.,.05)
        np.testing.assert_array_equal(labels,[0,1,1,1,0,0,1,1,1,0])
        self.assertTrue(keep.all())
        _,keep=score_targets([(.2,.8,60)],np.array([.2,.21,.5,.79,.8]),1.,.1)
        np.testing.assert_array_equal(keep,[False,False,True,False,False])

    # Reject reference intervals that assign two notes to the same supervision frame.
    def test_rejects_overlapping_reference(self):
        with self.assertRaises(ValueError):score_targets([(0,.8,60),(.5,1,62)],np.array([.6]),1.)
