from copy import deepcopy
import unittest
import numpy as np
from bpdr_input.expanded_experiment_preparation import array_difference,locked_protocol,preservation_qualified,select_checkpoint,resolve_training_catalog


class ExpandedExperimentPreparationTests(unittest.TestCase):
    def summary(self,error=20):
        return dict(pairs=304,missing_datasets=[],dataset_balanced_macro=dict(repair_error_cents=error),
            by_dataset={ds:dict(metrics=dict(original_plateau_alteration_cents=.5,outside_alteration_cents=.5)) for ds in ['vocadito','jamendo']},
            original_preservation=dict(originals=88,by_dataset={ds:dict(originals=n,voiced_originals=v,mean_alteration_cents=.5) for ds,n,v in [('vocadito',14,13),('jamendo',74,74)]}))

    def test_float_ulp_distance_handles_positive_negative_and_exact_nan(self):
        for value in [np.float32(1),np.float32(-1)]:
            a=np.array([value]);b=np.nextafter(a,np.float32(np.inf))
            self.assertEqual(array_difference(a,b)['maximum_ulp_distance'],1)
        self.assertTrue(array_difference(np.array([np.nan]),np.array([np.nan]))['exact'])

    def test_shape_dtype_or_different_nonfinite_cache_refused(self):
        for a,b in [(np.ones(2),np.ones(3)),(np.ones(2),np.ones(2,np.float32)),(np.array([np.nan]),np.array([1.]))]:
            with self.assertRaises(ValueError):array_difference(a,b)

    def test_preservation_is_per_dataset_and_missing_is_not_zero(self):
        s=self.summary();self.assertTrue(preservation_qualified(s))
        s['by_dataset']['vocadito']['metrics']['outside_alteration_cents']=1.01
        self.assertFalse(preservation_qualified(s))
        s=self.summary();s['original_preservation']['by_dataset']['vocadito']['mean_alteration_cents']=None
        self.assertFalse(preservation_qualified(s))

    def test_original_control_inventory_cannot_be_silently_reduced(self):
        s=self.summary();s['original_preservation']['by_dataset']['vocadito']['voiced_originals']=14
        self.assertFalse(preservation_qualified(s))

    def test_selection_uses_plateau_and_earliest_tie_with_zero_fallback(self):
        c=[dict(update=u,summary=self.summary(e)) for u,e in [(75,20),(25,21),(50,20)]]
        self.assertEqual(select_checkpoint(c,24.6),50)
        self.assertIsNone(select_checkpoint(c,19))
        c[0]['summary']['by_dataset']['jamendo']['metrics']['outside_alteration_cents']=2
        self.assertEqual(select_checkpoint(c,24.6),50)

    def test_checkpoint_budget_cannot_expand_or_duplicate(self):
        for c in [[dict(update=400,summary=self.summary())],[dict(update=25,summary=self.summary())]*2]:
            with self.assertRaises(ValueError):select_checkpoint(c,24.6)

    def test_protocol_keeps_loss_budget_and_prepared_status_explicit(self):
        p=locked_protocol();self.assertEqual(p['fit']['updates'],300);self.assertEqual(p['fit']['warmup'],100)
        self.assertEqual(p['total_planned_optimizer_updates'],900);self.assertEqual(p['training_updates_executed'],0)
        self.assertFalse(p['objective_changed']);self.assertEqual(p['checkpoint_selection']['preservation_limit_cents'],1.)

    def test_quarantine_is_phrase_specific_and_preserves_other_same_recording_phrase(self):
        def r(i,ds,pid):return dict(dataset=ds,phrase_id=pid,source_id='vocadito_37' if ds=='vocadito' else 'j'+str(i),singer_id='s',split='train',audit_passed=True,exploratory=True,training_eligible=False,cents=float(i),corruption='split')
        records=[r(i,'vocadito','vocadito_37_p00') for i in range(18)]+[r(i,'vocadito','vocadito_37_p01') for i in range(77)]+[r(i,'jamendo','j'+str(i)) for i in range(918)]
        identity=dict(passed=False,acoustic_arrays_exact={'f0_hz':True,'log_mel_db':False},sample_passthrough_exact=True,float_wav_roundtrip_exact=True,source_yin_exact=True,checkpoint_exact=True)
        before=deepcopy(records);selected,q=resolve_training_catalog(records,identity)
        self.assertEqual((len(selected),len(q)),(995,18));self.assertEqual(records,before)
        self.assertEqual(sum(r['phrase_id']=='vocadito_37_p01' for r in selected),77)
