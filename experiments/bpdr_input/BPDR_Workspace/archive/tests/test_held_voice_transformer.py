import unittest
import numpy as np
import torch
from unittest.mock import patch
from bpdr_input.modeling.held_voice_transformer import HeldConfig,HeldVoiceTransformer,tokens,predict_raw
from bpdr_input.repair.held_voice_policy import proposal,HeldPolicy,eligibility


def raw_input(n=500):
    raw=np.zeros((n,71),np.float32);raw[:,65:67]=1.;raw[:,68]=-20.
    return raw


class HeldVoiceTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_translation_invariant_input_no_chromatic_snap(self):
        raw=raw_input();raw[200:250,64]=20.
        shifted=raw.copy();shifted[:,64]+=137.5
        a,pad=tokens(raw,[0,220,499]);b,pad2=tokens(shifted,[0,220,499])
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(pad,pad2)
        self.assertEqual(a.shape,(3,51,7));self.assertEqual(a[1,25,0],np.float32(2/3))

    def test_exact_center_and_batch_equivalence(self):
        model=HeldVoiceTransformer(HeldConfig(hidden=16,layers=1,feed_forward=32)).eval()
        torch.nn.init.normal_(model.regression.weight,std=.1)
        raw=raw_input(23)
        a,_=predict_raw(model,raw,batch_size=7)
        b,_=predict_raw(model,raw,batch_size=23)
        np.testing.assert_allclose(a,b,atol=1e-5)
        self.assertTrue(np.isfinite(a).all());self.assertTrue(np.all(abs(a)<=30))

    def test_linear_drift_is_not_a_local_pitch_step(self):
        raw=raw_input(601);raw[:,64]=np.arange(601)*.08
        config=HeldConfig(radius=150,stride=6,centering='linear_flanks')
        data,_=tokens(raw,[300],config)
        self.assertLess(abs(data[0,25,0]),1e-6)
        raw[260:341,64]+=20.
        changed,_=tokens(raw,[300],config)
        self.assertAlmostEqual(float(changed[0,25,0]),2/3,places=5)

    def test_original_without_reliable_frames_cannot_supply_anchor_labels(self):
        from bpdr_input.experiments.learning.held_voice_training import anchor_pool
        row=dict(source=dict(singer_id='s1'),working_audio='unvoiced.wav')
        with patch('bpdr_input.experiments.learning.model_recovery.from_cached',return_value=(None,dict(reliable=np.zeros(5,bool)))):
            self.assertEqual(anchor_pool([row]),[])

    def test_neighboring_notes_do_not_erase_small_local_shift(self):
        raw=raw_input(701);raw[300:,64]=400.
        config=HeldConfig(radius=150,stride=6,centering='linear_flanks',scope_same_run=True,flank_samples=2)
        a,padding=tokens(raw,[390],config)
        shifted=raw.copy();shifted[350:431,64]+=20.
        c,_=tokens(shifted,[390],config)
        self.assertAlmostEqual(float(a[0,25,0]),0.,places=6)
        self.assertAlmostEqual(float(c[0,25,0]-a[0,25,0]),2/3,places=5)
        self.assertTrue(padding[0,0])

    def test_unvoiced_context_stays_finite_and_requests_identity(self):
        raw=raw_input(11);raw[:,66]=0.
        model=HeldVoiceTransformer(HeldConfig(scope_same_run=True))
        output,activity=predict_raw(model,raw)
        np.testing.assert_array_equal(output,np.zeros(11));np.testing.assert_array_equal(activity,np.zeros(11))

    def test_protected_overlap_vetoes_whole_candidate(self):
        raw=raw_input();times=np.arange(500)*.02;audio=np.zeros(499*441+100,np.float32)
        pred=np.zeros((3,500));pred[:,200:250]=-20.
        protection=np.zeros(500,bool);protection[225]=True
        result=proposal(audio,times,[raw]*3,pred,HeldPolicy(activation_cents=12),protection)
        self.assertFalse(np.any(result['curve']))
        self.assertIn('PROTECTED_REGION',[c['reason'] for c in result['candidates']])

    def test_multiple_spans_retained_and_unreliable_gap_not_bridged(self):
        raw=raw_input();times=np.arange(500)*.02;audio=np.zeros(499*441+100,np.float32)
        pred=np.zeros((3,500));pred[:,100:145]=-20.;pred[:,300:345]=20.
        result=proposal(audio,times,[raw]*3,pred,HeldPolicy(activation_cents=12,gap_frames=4))
        self.assertEqual(len(result['plan']['intervals_seconds']),2)
        self.assertTrue(result['plan']['permitted'])
        raw[120:122,66]=0.
        result=proposal(audio,times,[raw]*3,pred,HeldPolicy(activation_cents=12,gap_frames=4))
        self.assertFalse(np.any(result['curve'][100:145]))

    def test_small_pitch_boundary_is_not_an_acoustic_transition(self):
        raw=raw_input();raw[200:250,64]=20.;raw[200,70]=1.
        allowed,reliable,protected=eligibility(raw)
        self.assertTrue(allowed[220]);self.assertFalse(protected[200])
        raw[200:250,64]=100.
        _,_,protected=eligibility(raw)
        self.assertTrue(protected[200])

    def test_version_two_keeps_separate_continuous_held_pitch_regions(self):
        raw=raw_input(600);raw[200:400,64]=300.;raw[400:,64]=600.
        old,_,_=eligibility(raw)
        new,_,protected=eligibility(raw,region_split_cents=70.)
        self.assertFalse(old.any())
        self.assertTrue(new[100] and new[300] and new[500])
        self.assertTrue(protected[200] and protected[400])

    def test_measured_shape_guard_vetoes_the_whole_neural_candidate(self):
        raw=raw_input();times=np.arange(500)*.02;audio=np.zeros(499*441+100,np.float32)
        pred=np.zeros((3,500));pred[:,200:250]=-20.
        cue=np.zeros((500,3),np.float32);cue[200:250,1]=.70
        policy=HeldPolicy(version='held-voice-input-policy-2',activation_cents=8.,region_split_cents=70.,minimum_step_fit_gain=.75)
        with patch('bpdr_input.modeling.step_context.evidence',return_value=cue):
            result=proposal(audio,times,[raw]*3,pred,policy)
        self.assertFalse(np.any(result['curve']))
        self.assertIn('STEP_SHAPE_EVIDENCE',[c['reason'] for c in result['candidates']])
        cue[200:250,1]=.90
        with patch('bpdr_input.modeling.step_context.evidence',return_value=cue):
            result=proposal(audio,times,[raw]*3,pred,policy)
        self.assertTrue(result['plan']['permitted']);self.assertTrue(np.any(result['curve']))


if __name__=='__main__':unittest.main()
