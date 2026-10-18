import copy
import unittest
import numpy as np
from bpdr_input.audit_diagnostics import failure_reasons, pitch_metrics, window_comparison


def passing():
    return dict(audit_frame_count=20,plateau_coverage_fraction=1.,outside_audit_frame_count=20,outside_coverage_fraction=1.,
                independent_yin=dict(median_shift_cents=20.,median_error_cents=1.,p90_error_cents=2.),
                pesto=dict(median_shift_cents=33.,median_error_cents=13.,p90_error_cents=20.),requested_cents=20.,
                outside_median_drift_cents=0.,outside_p90_drift_cents=0.,identity_pass=True,same_sample_count=True,outside_support_exact=True)


class DiagnosticTests(unittest.TestCase):
    def test_good_pitch_cannot_hide_insufficient_audit_coverage(self):
        m=passing()
        self.assertEqual(failure_reasons(m),[])
        m['plateau_coverage_fraction']=.71875
        self.assertEqual(failure_reasons(m),['PLATEAU_COVERAGE'])

    def test_multiple_causes_are_retained_without_rewriting_measurements(self):
        m=passing()
        m['independent_yin']['median_error_cents']=6.
        m['independent_yin']['p90_error_cents']=14.
        m['outside_support_exact']=False
        prior=copy.deepcopy(m)
        self.assertEqual(set(failure_reasons(m)),{'FINE_MEDIAN_ERROR','FINE_P90_ERROR','OUTSIDE_SUPPORT_EXACT'})
        self.assertEqual(m,prior)

    def test_all_frame_diagnostic_exposes_errors_hidden_by_a_subset(self):
        before=np.full(4,440.)
        after=before*2**(np.array([20.,20.,70.,70.])/1200)
        subset=pitch_metrics(before,after,np.array([True,True,False,False]),20.)
        all_frames=pitch_metrics(before,after,np.ones(4,dtype=bool),20.)
        self.assertLess(subset['p90_error_cents'],.0001)
        self.assertGreater(all_frames['p90_error_cents'],49.)

    def test_window_crossing_ramp_is_not_called_an_interior_match(self):
        sr=22050
        full=np.ones(4*sr,dtype=np.float32)
        from bpdr_input.interventions import envelope
        local=envelope(np.arange(len(full))/sr,(1.,3.)).astype(np.float32)
        center_only=window_comparison(local,full,np.array([2.]),np.array([True]),sr,(1.,3.))
        edge=window_comparison(local,full,np.array([1.081]),np.array([True]),sr,(1.,3.))
        self.assertTrue(center_only['yin_windows_fully_in_unit_blend'])
        self.assertEqual(center_only['max_window_sample_difference'],0.)
        self.assertFalse(edge['yin_windows_fully_in_unit_blend'])
        self.assertGreater(edge['max_window_sample_difference'],0.)
