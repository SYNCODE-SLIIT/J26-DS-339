import unittest
from unittest.mock import patch
import numpy as np
from scipy.signal import butter,hilbert,sosfiltfilt
from test_segment_phase import toy
from bpdr_input.segment_phase import estimate_segment_phase
from bpdr_input.context_ablation import estimate_contexts
from bpdr_input.context_ablation_scoring import score_four


class ContextAblationTests(unittest.TestCase):
    def test_whole_clip_island_makes_all_contexts_identical(self):
        x,a=toy();c=estimate_segment_phase(x,22050,a);r=estimate_contexts(x,22050,a,c['raw_islands'])
        for key,new in [('sine_phase','local_sine_phase'),('amplitude','local_amplitude'),('frequency_hz','local_frequency_hz')]:
            np.testing.assert_array_equal(r['arrays']['D_'+key],c['arrays'][new])
            np.testing.assert_array_equal(r['arrays']['E_'+key],c['arrays'][new])
        self.assertIsNone(r['accepted_audio']);self.assertFalse(r['production_eligible'])

    def test_same_full_filter_is_shared_while_hilbert_domains_differ(self):
        x,a=toy(gap=True);c=estimate_segment_phase(x,22050,a)
        with patch('bpdr_input.context_ablation.sosfiltfilt',wraps=sosfiltfilt) as filtered:r=estimate_contexts(x,22050,a,c['raw_islands'])
        self.assertEqual(filtered.call_count,2)
        z=sosfiltfilt(butter(4,[.7*173,1.3*173],btype='bandpass',fs=22050,output='sos'),x.astype(np.float64),padtype='odd',padlen=27)
        full=np.gradient(np.unwrap(np.angle(hilbert(z))))*22050/(2*np.pi)
        for i in c['raw_islands']:
            lo,hi=i['start_sample'],i['end_sample']
            np.testing.assert_array_equal(r['arrays']['D_frequency_hz'][lo:hi],full[lo:hi])
            own=np.gradient(np.unwrap(np.angle(hilbert(z[lo:hi]))))*22050/(2*np.pi)
            np.testing.assert_array_equal(r['arrays']['E_frequency_hz'][lo:hi],own)

    def test_short_islands_have_no_filter_call(self):
        x,a=toy(length=4410);c=estimate_segment_phase(x,22050,a)
        with patch('bpdr_input.context_ablation.sosfiltfilt') as f:r=estimate_contexts(x,22050,a,c['raw_islands'])
        f.assert_not_called();self.assertEqual(r['D']['raw_islands'][0]['reason'],'SHORT_RAW_ISLAND')

    def test_filter_failure_is_recorded_for_both_arms_no_retry(self):
        x,a=toy();c=estimate_segment_phase(x,22050,a)
        with patch('bpdr_input.context_ablation.sosfiltfilt',side_effect=ValueError('toy')) as f:
            r=estimate_contexts(x,22050,a,c['raw_islands'])
        self.assertEqual(f.call_count,1)
        for arm in ('D','E'):self.assertEqual(r[arm]['raw_islands'][0]['reason'],'FULL_FILTER_ERROR')

    def test_saved_center_is_used_and_invalid_center_has_no_fallback(self):
        x,a=toy();c=estimate_segment_phase(x,22050,a);c['raw_islands'][0]['local_center_hz']=10000.
        r=estimate_contexts(x,22050,a,c['raw_islands'])
        self.assertEqual(r['D']['raw_islands'][0]['reason'],'INVALID_SAVED_LOCAL_BAND')

    def test_input_masks_and_gap_nan_are_preserved(self):
        x,a=toy(gap=True);c=estimate_segment_phase(x,22050,a);saved={k:v.copy() for k,v in a.items()}
        r=estimate_contexts(x,22050,a,c['raw_islands'])
        for k in a:np.testing.assert_array_equal(saved[k],a[k])
        for arm in ('D','E'):self.assertTrue(np.isnan(r['arrays'][arm+'_frequency_hz'][~a['sample_reliable']]).all())
        for k in ('sample_reliable','guarded_support','retained_support'):np.testing.assert_array_equal(a[k],r['arrays'][k])

    def test_four_arm_common_domain_does_not_hide_invalid_predictions(self):
        fs={a:np.full(4,200.) for a in 'BCDE'};fs['E'][1]=np.nan;fs['D'][2]=-1
        r=score_four(fs,np.ones(4,dtype=bool),np.full(4,200.),np.ones(4,dtype=bool),np.full(4,3000))
        self.assertEqual(r['retained_denominator'],4);self.assertEqual(r['common_four_arm_voiced_samples'],2)
        self.assertEqual(r['arms']['D']['invalid_on_retained'],1)
        self.assertEqual(r['arms']['B']['scored_voiced_samples'],4)
        self.assertEqual(r['arms']['B']['common_four_arm_absolute_error_cents']['samples'],2)

    def test_synthetic_silence_is_counted_but_not_scored_as_pitch(self):
        fs={a:np.full(2,200.) for a in 'BCDE'}
        r=score_four(fs,np.ones(2,dtype=bool),np.full(2,200.),np.array([True,False]),np.full(2,3000))
        self.assertEqual(r['retained_silent_samples'],1)
        self.assertEqual(r['arms']['B']['edge_bins'][1]['common_voiced_samples'],1)
