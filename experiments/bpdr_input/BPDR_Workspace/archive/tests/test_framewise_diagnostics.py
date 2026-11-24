import unittest
import numpy as np
from bpdr_input.framewise_diagnostics import direction_details, relative_peak, transition_mask
from bpdr_input.framewise_audit import audit_curve


class DiagnosticTests(unittest.TestCase):
    def test_tiny_opposing_change_remains_rejected(self):
        t=np.arange(201)*.02;n=len(t);f=np.full(n,220.,dtype=np.float32)
        before=dict(times_seconds=t,f0_hz=f,voiced_mask=np.ones(n,dtype=bool),confidence=np.ones(n))
        curve=np.where((t>=1.)&(t<2.),-20.,0.)
        after={**before,'f0_hz':f.copy()};after['f0_hz'][70]=np.nextafter(f[70],np.float32(np.inf))
        fine=f.astype(float)*2**(curve/1200.)
        audit=audit_curve(before,after,f,fine,before['voiced_mask'],[(1.,2.)],curve)
        rows,arrays=direction_details(before,after,f,fine,before['voiced_mask'],[(1.,2.)],curve)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['size_bin'],'<=.001')
        self.assertGreater(rows[0]['frequency_difference_hz'],0.)
        self.assertGreater(rows[0]['float64_ratio_coarse_cents'],0.)
        self.assertFalse(audit['passed']);self.assertIn('COARSE_DIRECTION',audit['reasons'])
        self.assertTrue(arrays['opposing'][70])

    def test_large_opposing_change_is_separate_descriptive_bin(self):
        t=np.arange(201)*.02;n=len(t)
        before=dict(times_seconds=t,f0_hz=np.full(n,220.),voiced_mask=np.ones(n,dtype=bool),confidence=np.ones(n))
        after={**before,'f0_hz':before['f0_hz']*2**(20/1200)};curve=np.full(n,-20.)
        rows,_=direction_details(before,after,before['f0_hz'],before['f0_hz']*2**(-20/1200),before['voiced_mask'],[(1.,2.)],curve)
        self.assertTrue(rows);self.assertEqual({r['size_bin'] for r in rows},{'>10'})

    def test_transition_windows_exclude_guarded_interior_and_include_both_supports(self):
        t=np.array([.9,1.,1.18,1.5,1.82,2.,2.1,5.,5.5,6.])
        np.testing.assert_array_equal(transition_mask(t,[(1.,2.),(5.,6.)]),[1,1,0,0,0,1,1,1,0,1])

    def test_peak_ignores_invalid_frequencies_and_empty_mask_refuses(self):
        result,delta=relative_peak(np.array([220.,0.,220.]),np.array([440.,220.,np.nan]),np.ones(3,dtype=bool))
        self.assertEqual(result['peak_abs_cents'],1200.);self.assertEqual(result['valid_frames'],1)
        self.assertTrue(np.isnan(delta[1:]).all())
        with self.assertRaises(ValueError):relative_peak(np.ones(2),np.ones(2),np.zeros(2,dtype=bool))
