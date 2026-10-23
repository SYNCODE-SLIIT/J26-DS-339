"""No annotation leakage, no guessed uncertainty, no bridging of unsafe supports."""
import unittest
import numpy as np
from bpdr_input.prediction_adapter import VIEWS, infer_views, propose, runtime_views, validate_features


class PredictionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.audio=np.full(8*22050,.1,dtype=np.float32)
        self.times=np.arange(len(self.audio)//441+1)*.02
        n=len(self.times)
        frame=np.zeros((n,70),dtype=np.float32); frame[:,65]=.9; frame[:,66]=1.
        self.features=dict(times_seconds=self.times,confidence=np.full(n,.9),voiced_mask=np.ones(n,dtype=bool),
            provisional_boundary_score=np.zeros(n),frame_features_70=frame,
            boundary_features_71=np.column_stack([frame,np.zeros(n)]).astype(np.float32))
        self.curve=np.where((self.times>=1.)&(self.times<1.9),20.,0.)

    def preview(self,curve=None,**kwargs):
        curve=self.curve if curve is None else curve
        predictions={v:curve.copy() for v in VIEWS}
        predictions.update(kwargs.pop('predictions',{}))
        return propose(self.audio,22050,self.features,predictions,**kwargs)[0]

    def test_runtime_never_guesses_uncertainty_or_returns_fixture_permission(self):
        result=self.preview()
        self.assertFalse(result['edit_permission'])
        self.assertIn('UNCERTAINTY_UNAVAILABLE',result['blockers'])
        self.assertIn('EXPRESSIVITY_PROTECTION_UNAVAILABLE',result['blockers'])
        self.assertIsNone(result['fixture_permission_plan'])
        # Even supplied low values cannot be described as calibrated uncertainty.
        result=self.preview(uncertainty=np.ones(len(self.times)),expressive_mask=np.zeros(len(self.times),dtype=bool))
        self.assertFalse(result['edit_permission'])
        self.assertIn('UNCERTAINTY_UNCALIBRATED',result['blockers'])

    def test_subthreshold_predictions_do_not_imply_source_musical_suitability(self):
        result=self.preview(np.full(len(self.times),.2))
        self.assertEqual(result['outcome'],'NO_EDIT_PROPOSED')
        self.assertEqual(result['candidate_intervals'],[])
        self.assertFalse(result['production_eligible'])
        self.assertIsNone(result['accepted_audio'])

    def test_boundary_views_change_only_the_71st_column_and_do_not_mutate_source(self):
        self.features['boundary_features_71'][60,-1]=1.
        original=self.features['boundary_features_71'].copy()
        views=runtime_views(self.features)
        for values in views.values(): self.assertTrue(np.array_equal(values[:,:70],original[:,:70]))
        self.assertEqual(np.argmax(views['earlier_40ms'][:,-1]),58)
        self.assertEqual(np.argmax(views['later_40ms'][:,-1]),62)
        self.assertTrue(np.array_equal(original,self.features['boundary_features_71']))

    def test_targets_and_misaligned_feature_columns_are_rejected(self):
        self.features['correction_target_cents']=np.zeros(len(self.times))
        with self.assertRaises(ValueError): validate_features(self.features,self.audio,22050)
        del self.features['correction_target_cents']
        self.features['boundary_features_71'][0,0]=1.
        with self.assertRaises(ValueError): validate_features(self.features,self.audio,22050)

    def test_sign_changes_are_separate_supported_intervals(self):
        curve=self.curve.copy();curve[(self.times>=1.9)&(self.times<2.8)]=-20.
        result=self.preview(curve,mode='synthetic_fixture',uncertainty=np.ones(len(self.times)),
                            expressive_mask=np.zeros(len(self.times),dtype=bool))
        self.assertTrue(result['edit_permission'])
        self.assertEqual([i['direction'] for i in result['candidate_intervals']],['raise','lower'])
        self.assertEqual(len(result['usable_intervals_seconds']),2)

    def test_sign_changes_cannot_hide_short_supports_in_a_long_combined_run(self):
        curve=self.curve.copy();curve[(self.times>=1.45)&(self.times<1.9)]=-20.
        result=self.preview(curve,mode='synthetic_fixture',uncertainty=np.ones(len(self.times)),
                            expressive_mask=np.zeros(len(self.times),dtype=bool))
        self.assertFalse(result['edit_permission'])
        self.assertIn('SIGN_OR_GAP_SPLIT_SUPPORT_TOO_SHORT',result['blockers'])

    def test_worst_alternate_and_protected_transition_are_not_ignored(self):
        alternate=self.curve.copy(); alternate[60]=27.
        result=self.preview(mode='synthetic_fixture',uncertainty=np.ones(len(self.times)),
                            expressive_mask=np.zeros(len(self.times),dtype=bool),predictions={'later_40ms':alternate})
        self.assertIn('BOUNDARY_DISAGREEMENT',result['blockers'])
        self.features['provisional_boundary_score'][60]=1.
        self.features['boundary_features_71'][60,-1]=1.
        result=self.preview(mode='synthetic_fixture',uncertainty=np.ones(len(self.times)),
                            expressive_mask=np.zeros(len(self.times),dtype=bool))
        self.assertIn('PROTECTED_REGION',result['blockers'])

    def test_nonfinite_or_incomplete_predictions_cannot_enter_controller(self):
        with self.assertRaises(ValueError): self.preview(predictions={'main':np.full(len(self.times),np.nan)})
        with self.assertRaises(ValueError): propose(self.audio,22050,self.features,{'main':self.curve})

    def test_inference_uses_saved_normalizer_eval_and_only_acoustic_inputs(self):
        import torch
        from types import SimpleNamespace
        from bpdr_input.examples import Normalizer
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.));self.config=SimpleNamespace(features=71)
            def forward(self,x): return x[:,:,65]*self.weight
        model=Probe()
        normalizer=Normalizer(np.zeros(71),np.full(71,2.))
        with self.assertRaises(ValueError): infer_views(model,normalizer,self.features)
        model.eval();result=infer_views(model,normalizer,self.features)
        self.assertTrue(all(np.allclose(v,.45) for v in result.values()))
        self.assertTrue(np.all(normalizer.scale==2.))


if __name__=='__main__': unittest.main()
