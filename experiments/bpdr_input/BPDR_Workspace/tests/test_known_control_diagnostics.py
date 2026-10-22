import unittest
import numpy as np
from bpdr_input.known_control_diagnostics import residual_summary, strict_sample_law, native_context


class KnownControlDiagnosticTests(unittest.TestCase):
    def test_intended_change_is_not_error_and_zero_change_is_identity(self):
        mask=np.ones(3,dtype=bool);law=np.array([-20.,0.,20.])
        metrics,residual=residual_summary(law,law,mask)
        np.testing.assert_array_equal(residual,0.)
        self.assertEqual(metrics['max_abs_center_law_residual_cents'],0.)
        zero,_=residual_summary(np.zeros(3),np.zeros(3),mask)
        self.assertEqual(zero['max_abs_center_law_residual_cents'],0.)

    def test_raw_peak_and_residual_peak_can_be_different_frames(self):
        delta=np.array([30.,8.,np.nan]);law=np.array([30.,-10.,0.])
        metrics,r=residual_summary(delta,law,np.ones(3,dtype=bool))
        self.assertEqual(metrics['residual_peak_frame'],1)
        self.assertEqual(metrics['max_abs_center_law_residual_cents'],18.)
        self.assertEqual(metrics['valid_frames'],2)
        self.assertTrue(np.isnan(r[2]))

    def test_empty_valid_support_and_changed_grids_refuse(self):
        for delta,law,mask in (([np.nan],[0.],np.ones(1,dtype=bool)),
                               ([1.,2.],[0.],np.ones(2,dtype=bool)),
                               ([1.],[0.],np.array([1]))):
            with self.assertRaises(ValueError):residual_summary(delta,law,mask)

    def test_strict_sample_law_preserves_zero_outside_and_exact_timestamps(self):
        t=np.arange(8)*.01;a=dict(times_seconds=t[2:6],sign_preserving_cents=np.array([0.,-10.,10.,0.]))
        law=strict_sample_law(t,[(.02,.06,a)])
        np.testing.assert_array_equal(law,[0.,0.,0.,-10.,10.,0.,0.,0.])
        with self.assertRaises(ValueError):strict_sample_law(t,[(.02,.06,{**a,'times_seconds':a['times_seconds']+.001})])

    def test_no_relaxed_fallback_or_overlapping_supports(self):
        t=np.arange(8)*.01;a=dict(times_seconds=t[2:6],sign_preserving_cents=np.zeros(4))
        with self.assertRaises(ValueError):strict_sample_law(t,[(.02,.06,a),(.02,.06,a)])
        with self.assertRaises(KeyError):strict_sample_law(t,[(.02,.06,dict(times_seconds=t[2:6],relaxed_cents=np.zeros(4)))])

    def test_native_alignment_retains_actual_later_frame_on_tie(self):
        f=dict(times_seconds=np.array([.01,.019]),native_pitch_times_seconds=np.array([0.,.02]),
               native_pitch_hz=np.array([220.,221.]),native_pitch_confidence=np.array([.9,.95]),f0_hz=np.array([221.,221.]))
        self.assertEqual(native_context(f,0)['native_frame'],1)
        self.assertEqual(native_context(f,1)['native_time_seconds'],.02)
        self.assertEqual(native_context(f,1)['native_hz'],221.)
