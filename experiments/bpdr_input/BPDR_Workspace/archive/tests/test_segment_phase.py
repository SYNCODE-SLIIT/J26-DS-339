import unittest
from unittest.mock import patch
import numpy as np
from bpdr_input.source_estimator import nearest_indices, guarded_support
from bpdr_input.segment_phase import estimate_segment_phase, interval_summary


def toy(length=22050, gap=False, hz=173.):
    t = np.arange(0, length/22050+.02, .02)
    keep = np.ones(len(t), dtype=bool)
    if gap: keep[(t>=.4)&(t<.6)] = False
    st = np.arange(length)/22050
    mask = keep[nearest_indices(t, st)]; guard = guarded_support(mask, 22050)
    f = np.full(len(t), hz)
    a = dict(frame_times=t, frame_pesto_hz=f, frame_yin_hz=f.copy(), frame_confidence=np.ones(len(t)),
             frame_reliable=keep, sample_times=st, sample_reliable=mask, guarded_support=guard,
             retained_support=guard.copy(), estimated_amplitude=np.ones(length))
    # Model a low-confidence gap without changing either surrounding frequency.
    a['frame_confidence'][~keep] = 0
    audio=(.12*np.sin(2*np.pi*hz*st+.31)).astype(np.float32)
    return audio, a


class SegmentPhaseTests(unittest.TestCase):
    def test_independent_tone_has_expected_interior_frequency_and_no_acceptance(self):
        x,a=toy();r=estimate_segment_phase(x,22050,a)
        self.assertEqual(r['phase_feasible_intervals'],1)
        f=r['arrays']['local_frequency_hz'][a['retained_support']]
        self.assertLess(float(np.median(abs(1200*np.log2(f/173.)))),1.)
        self.assertIsNone(r['accepted_audio']);self.assertFalse(r['production_eligible'])

    def test_no_gap_bridging_and_masks_unchanged(self):
        x,a=toy(gap=True);before={k:v.copy() for k,v in a.items()};r=estimate_segment_phase(x,22050,a)
        self.assertEqual(len(r['raw_islands']),2)
        self.assertTrue(np.isnan(r['arrays']['local_frequency_hz'][~a['sample_reliable']]).all())
        for k in a:np.testing.assert_array_equal(a[k],before[k])
        for k in ('sample_reliable','guarded_support','retained_support'):np.testing.assert_array_equal(r['arrays'][k],a[k])

    def test_short_island_is_recorded_without_calling_filter(self):
        x,a=toy(length=4410)
        with patch('bpdr_input.segment_phase.sosfiltfilt') as filtering:r=estimate_segment_phase(x,22050,a)
        filtering.assert_not_called();self.assertEqual(r['short_islands'],1)
        self.assertEqual(r['retained_samples'],0)

    def test_first_allowed_raw_length_keeps_one_guarded_sample(self):
        x,a=toy(length=4411);r=estimate_segment_phase(x,22050,a)
        self.assertEqual(r['computed_islands'],1);self.assertEqual(r['retained_samples'],1)
        self.assertEqual(r['phase_feasible_long_intervals'],0)

    def test_one_bad_sample_fails_whole_interval_without_splitting(self):
        mask=np.ones(20000,dtype=bool);f=np.full(20000,173.);f[10000]=-1
        r=interval_summary(f,mask)
        self.assertEqual(r['retained_intervals'],1);self.assertEqual(r['phase_feasible_samples'],0)
        self.assertEqual(r['nonpositive_on_retained'],1);self.assertEqual(r['retained_samples'],20000)

    def test_invalid_local_band_has_no_fallback(self):
        x,a=toy(hz=10000.);r=estimate_segment_phase(x,22050,a)
        self.assertEqual(r['raw_islands'][0]['reason'],'INVALID_LOCAL_BAND')
        self.assertEqual(r['nonfinite_on_retained'],r['retained_samples'])

    def test_filter_error_is_recorded_without_retry(self):
        x,a=toy()
        with patch('bpdr_input.segment_phase.sosfiltfilt',side_effect=ValueError('toy failure')) as filtering:
            r=estimate_segment_phase(x,22050,a)
        self.assertEqual(filtering.call_count,1);self.assertEqual(r['raw_islands'][0]['reason'],'LOCAL_FILTER_ERROR')

    def test_nonfinite_analytic_signal_is_not_salvaged(self):
        x,a=toy()
        with patch('bpdr_input.segment_phase.hilbert',return_value=np.full(len(x),complex(np.nan))):
            r=estimate_segment_phase(x,22050,a)
        self.assertEqual(r['raw_islands'][0]['reason'],'NONFINITE_LOCAL_ANALYTIC_SIGNAL')
        self.assertEqual(r['phase_feasible_samples'],0)

    def test_changed_support_or_audio_format_is_rejected(self):
        x,a=toy();a['retained_support'][0]=True
        with self.assertRaises(ValueError):estimate_segment_phase(x,22050,a)
        x,a=toy()
        with self.assertRaises(ValueError):estimate_segment_phase(x.astype(np.float64),22050,a)
        with self.assertRaises(ValueError):estimate_segment_phase(x,16000,a)

    def test_local_median_uses_unique_represented_frames_not_global_center(self):
        x,a=toy(gap=True)
        select=a['frame_times']>=.6
        a['frame_yin_hz'][select]=311.;a['frame_pesto_hz'][select]=311.
        r=estimate_segment_phase(x,22050,a)
        self.assertEqual([v['local_center_hz'] for v in r['raw_islands']],[173.,311.])
