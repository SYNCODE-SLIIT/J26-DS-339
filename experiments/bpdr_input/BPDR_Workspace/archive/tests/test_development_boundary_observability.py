import unittest
import numpy as np
from bpdr_input.development_boundary_observability import changed_boundary_frames
from bpdr_input.shared_pair_audit import game_boundary_track


class BoundaryObservabilityTests(unittest.TestCase):
    def test_removed_close_edge_can_be_invisible_on_sampled_gaussian_grid(self):
        t=np.arange(0,1,.02)
        a=game_boundary_track(t,[.4001,.4002,.4003]);b=game_boundary_track(t,[.4001,.4003])
        self.assertEqual(changed_boundary_frames(a,b),0)

    def test_observable_merge_is_kept(self):
        t=np.arange(0,1,.02);a=game_boundary_track(t,[.2,.4,.6]);b=game_boundary_track(t,[.2,.6])
        self.assertGreater(changed_boundary_frames(a,b),0)

    def test_precision_only_metadata_change_is_not_promoted(self):
        self.assertEqual(changed_boundary_frames(np.array([.5]),np.array([.5+1e-12])),0)

    def test_no_mutation(self):
        a=np.array([0.,1.,0.]);b=np.array([0.,.5,0.]);old=a.copy()
        self.assertEqual(changed_boundary_frames(a,b),1);np.testing.assert_array_equal(a,old)

    def test_bad_shapes_and_nonfinite_values_refused(self):
        for a,b in [([],[]),([1],[1,2]),([np.nan],[0]),([[0]],[[1]])]:
            with self.assertRaises(ValueError):changed_boundary_frames(a,b)
