import copy
import unittest
from pathlib import Path
import tempfile
import numpy as np
import torch

from bpdr_input.comparison_preparation import comparison_protocol
from bpdr_input.matched_comparison import aggregate, check_catalog, evaluate_no_change, macro, matched_model, metric_row
from bpdr_input.examples import Normalizer
from bpdr_input.losses import objective


def record(dataset='vocadito',group='s',source='p',cents=20.,corruption='split'):
    return dict(dataset=dataset,group_id=group,source_id=source,phrase_id=source,cents=cents,corruption=corruption,
        group_kind='fixture',reference_type='fixture',split='development',audit_passed=True,
        exploratory=True,training_eligible=False,data_status='exploratory_not_research')


class MatchedComparisonTests(unittest.TestCase):
    def test_hierarchical_macro_does_not_weight_many_variants_more(self):
        rows=[dict(record(corruption=k),metric=0.) for k in ['split','merge','jitter']]
        rows += [dict(record(group='t',source='q'),metric=20.),dict(record(dataset='jamendo',group='a',source='j'),metric=30.)]
        self.assertEqual(macro(rows,'metric'),20.)

    def test_missing_dataset_reported_unavailable_not_zero(self):
        r=record(dataset='jamendo')
        for key in ['correction_mae_cents','repair_error_cents','original_audited_alteration_cents','outside_alteration_cents',
                    'boundary_sensitivity_cents','counter_detuning_error_cents','original_joint_voiced_no_change_cents']:r[key]=3.
        result=aggregate([r])
        self.assertEqual(result['missing_datasets'],['vocadito'])
        self.assertIsNone(result['full_two_dataset_macro'])
        self.assertEqual(result['selection_criterion'],3.)

    def arrays(self):
        x={v:np.zeros((20,71),dtype=np.float32) for v in 'ABCD'}
        for v in x:x[v][:,66]=1
        mask=np.ones(20,dtype=bool);mask[:2]=False
        detuning=np.zeros(20);detuning[6:11]=20
        return x,mask,detuning

    def test_perfect_relative_correction_and_zero_baseline_are_distinct(self):
        x,mask,e=self.arrays()
        p={v:np.zeros(20) if v in 'AB' else -e for v in 'ABCD'}
        result=metric_row(record(),p,x,mask,e)
        self.assertEqual(result['correction_mae_cents'],0.)
        self.assertEqual(result['repair_error_cents'],0.)
        z=metric_row(record(),{v:np.zeros(20) for v in 'ABCD'},x,mask,e)
        self.assertEqual(z['repair_error_cents'],20.)
        self.assertGreater(z['correction_mae_cents'],0.)

    def test_no_change_diagnostic_includes_voiced_frames_outside_target_mask(self):
        x,mask,e=self.arrays();p={v:np.zeros(20) for v in 'ABCD'}
        p['A'][0]=20.;p['B'][0]=20.
        result=metric_row(record(),p,x,mask,e)
        self.assertEqual(result['original_audited_alteration_cents'],0.)
        self.assertEqual(result['original_joint_voiced_no_change_cents'],1.)
        self.assertEqual(result['joint_voiced_no_change_frames'],20)

    def test_catalog_partition_and_approval_claims_rejected(self):
        with self.assertRaisesRegex(ValueError,'No usable'):check_catalog([],'development')
        with self.assertRaisesRegex(ValueError,'Wrong partition'):check_catalog([record()],'train')
        r=record();r['training_eligible']=True
        with self.assertRaises(ValueError):check_catalog([r],'development')

    def test_unpaired_preservation_needs_no_detuning_target_archive(self):
        class Constant(torch.nn.Module):
            def forward(self,x):return torch.ones(x.shape[:2])*4
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.npz';x=np.zeros((10,71),dtype=np.float32);x[:6,66]=1
            np.savez_compressed(path,boundary_features_71=x)
            r=record();r['input_path']=str(path)
            result=evaluate_no_change(Constant(),[r],Normalizer(np.zeros(71),np.ones(71)),71)
            self.assertEqual(result['by_dataset']['vocadito']['mean_alteration_cents'],4.)
            self.assertFalse(result['musical_approval'])

    def test_matched_initialization_and_rng_after_width_change(self):
        p=comparison_protocol();p['model']=dict(hidden=8,layers=1,heads=2,feed_forward=16,dropout=.1)
        a=matched_model(p,71);rng_a=torch.get_rng_state()
        f=matched_model(p,70);rng_f=torch.get_rng_state()
        self.assertTrue(torch.equal(rng_a,rng_f))
        for key,value in a.state_dict().items():
            expected=value[:,:70] if key=='project.weight' else value
            self.assertTrue(torch.equal(expected,f.state_dict()[key]),key)

    def test_microbatch_gradient_average_matches_batch_loss_without_dropout(self):
        p=comparison_protocol();p['model']=dict(hidden=8,layers=1,heads=2,feed_forward=16,dropout=0.)
        a=matched_model(p,71);b=matched_model(p,71)
        x=torch.randn(2,12,71)
        batch=dict(inputs={v:x.clone() for v in 'ABCD'},padding=torch.zeros(2,12,dtype=torch.bool),
            mask=torch.ones(2,12,dtype=torch.bool),detuning=torch.ones(2,12)*20,
            voiced={v:torch.ones(2,12,dtype=torch.bool) for v in 'AC'})
        pred={v:a(t,batch['padding']) for v,t in batch['inputs'].items()}
        objective(pred,batch,'bpdr')[0].backward()
        for i in range(2):
            single={k:({v:t[i:i+1] for v,t in value.items()} if isinstance(value,dict) else value[i:i+1]) for k,value in batch.items()}
            pred={v:b(t,single['padding']) for v,t in single['inputs'].items()}
            (objective(pred,single,'bpdr')[0]/2).backward()
        for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa.grad,pb.grad,rtol=2e-4,atol=2e-5)


if __name__=='__main__':unittest.main()
