import unittest
from unittest.mock import patch
import numpy as np
from bpdr_input.source_estimator import estimate_source, guarded_support, nearest_indices
from bpdr_input.source_estimation_experiment import score_prediction
from bpdr_input.framewise_fixtures import constructed, signal


class SourceEstimationTests(unittest.TestCase):
    def source(self):
        sr=22050;t=np.arange(2*sr)/sr;audio=(.12*np.sin(2*np.pi*197.*t+.37)).astype(np.float32)
        frames=np.arange(len(audio)//441+1)*.02
        features=dict(times_seconds=frames,f0_hz=np.full(len(frames),197.),confidence=np.ones(len(frames)))
        return audio,sr,t,features

    def test_unlabelled_unseen_frequency_estimates_phase_without_oracle_offset(self):
        audio,sr,t,features=self.source()
        with patch('bpdr_input.source_estimator.extract',return_value=(features,{})),patch('bpdr_input.source_estimator.independent_pitch',return_value=np.full(len(features['times_seconds']),197.)):
            result=estimate_source(audio,sr)
        a=result['arrays'];mask=a['retained_support']
        phase_error=abs(np.angle(np.exp(1j*(a['estimated_sine_phase']-(2*np.pi*197*t+.37)))))
        self.assertLess(np.quantile(phase_error[mask],.9),.02)
        self.assertLess(np.median(abs(1200*np.log2(a['estimated_frequency_hz'][mask]/197.))),1.)
        self.assertFalse(result['production_eligible']);self.assertIsNone(result['accepted_audio'])
        np.testing.assert_array_equal(audio,self.source()[0])
        with self.assertRaises(TypeError):estimate_source(audio,sr,profile='unknown')

    def test_no_reliable_source_frames_refuses_without_filter_fallback(self):
        audio,sr,t,features=self.source();features['confidence'][:]=0.
        with patch('bpdr_input.source_estimator.extract',return_value=(features,{})),patch('bpdr_input.source_estimator.independent_pitch',return_value=np.full(len(features['times_seconds']),197.)):
            result=estimate_source(audio,sr)
        self.assertEqual(result['status'],'refused');self.assertEqual(result['reasons'],['NO_RELIABLE_SOURCE_FRAMES'])
        self.assertNotIn('estimated_sine_phase',result['arrays'])

    def test_nonpositive_edge_frequency_refuses_even_when_outside_retained_support(self):
        audio,sr,t,features=self.source();phase=2*np.pi*197*t;phase[0]=phase[1]
        with patch('bpdr_input.source_estimator.extract',return_value=(features,{})),patch('bpdr_input.source_estimator.independent_pitch',return_value=np.full(len(features['times_seconds']),197.)),patch('bpdr_input.source_estimator.hilbert',return_value=np.exp(1j*phase)):
            result=estimate_source(audio,sr)
        self.assertEqual(result['status'],'refused');self.assertIn('NONPOSITIVE_ESTIMATED_FREQUENCY',result['reasons'])
        self.assertGreater(result['nonpositive_frequency_outside_retained_samples'],0)
        self.assertEqual(result['nonpositive_frequency_on_retained_samples'],0)

    def test_predicted_gaps_and_clip_edges_get_guards_without_truth_input(self):
        sr=22050;reliable=np.ones(sr,dtype=bool);reliable[9000:12000]=False
        mask=guarded_support(reliable,sr)
        self.assertFalse(mask[:2205].any());self.assertFalse(mask[-2205:].any())
        self.assertFalse(mask[9000-2205:12000+2205].any());self.assertTrue(mask[3000])
        np.testing.assert_array_equal(nearest_indices(np.array([0.,.02]),np.array([.009,.01,.019])),[0,1,1])

    def test_false_silence_support_is_reported_and_never_scored_as_voiced(self):
        profile=dict(hz=220,kind='rest',seconds=8);t,f,amp=constructed(profile);audio=signal(f,amp)
        n=len(t);arrays=dict(sample_reliable=np.ones(n,dtype=bool),guarded_support=np.ones(n,dtype=bool),retained_support=np.ones(n,dtype=bool),
            estimated_frequency_hz=f.copy(),estimated_sine_phase=2*np.pi*np.cumsum(f)/22050,estimated_amplitude=amp.copy())
        scores,_=score_prediction(profile,audio,22050,arrays)
        self.assertEqual(scores['retained_false_support_in_silence'],44100)
        self.assertEqual(scores['joint_scored_samples'],int((amp>0).sum()))
        self.assertEqual(scores['frequency_error_cents']['maximum'],0.)

    def test_global_phase_offset_is_measured_not_fitted_away(self):
        profile=dict(hz=177.7,kind='constant',seconds=1);t,f,amp=constructed(profile);n=len(t)
        arrays=dict(retained_support=np.ones(n,dtype=bool),estimated_frequency_hz=f,
            estimated_sine_phase=2*np.pi*np.cumsum(f)/22050+.4,estimated_amplitude=amp)
        scores,_=score_prediction(profile,signal(f,amp),22050,arrays)
        self.assertAlmostEqual(scores['absolute_circular_phase_error_radians']['median'],.4,places=10)

    def test_invalid_joint_frequency_is_not_silently_removed_from_metric(self):
        profile=dict(hz=177.7,kind='constant',seconds=1);t,f,amp=constructed(profile);n=len(t);bad=f.copy();bad[4000]=0.
        arrays=dict(retained_support=np.ones(n,dtype=bool),estimated_frequency_hz=bad,
            estimated_sine_phase=2*np.pi*np.cumsum(f)/22050,estimated_amplitude=amp)
        scores,_=score_prediction(profile,signal(f,amp),22050,arrays)
        self.assertIsNone(scores['frequency_error_cents']);self.assertEqual(scores['joint_scored_samples'],n)
        self.assertIn('frequency_score_unavailable_reason',scores)
