from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from bpdr_input.frozen_development_evaluation import frozen_model_inventory,response_row,summarize,frozen_forward,SELECTED_UPDATES


class FrozenDevelopmentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.record=dict(dataset='vocadito',group_id='vocadito:S1',source_id='v1',phrase_id='v1_p0',cents=30.,corruption='split',
            reference_type='human_annotation',group_kind='annotated_singer_id')
        x=np.zeros((30,71),np.float32);x[:,66]=1
        self.pair=dict(inputs={v:x.copy() for v in 'ABCD'},mask=np.ones(30,bool),detuning=np.r_[np.full(10,30,np.float32),np.zeros(20,np.float32)])
        self.zero={v:np.zeros(30,np.float32) for v in 'ABCD'}

    def test_zero_baseline_exposes_no_correction_despite_perfect_preservation(self):
        r=response_row(self.record,self.zero,self.pair)
        self.assertEqual(r['repair_error_cents'],30);self.assertEqual(r['plateau_counter_detuning_error_cents'],30)
        self.assertEqual(r['plateau_response_gain'],0);self.assertEqual(r['original_plateau_alteration_cents'],0)

    def test_perfect_response_has_correct_sign_and_gain(self):
        p=deepcopy(self.zero)
        for v in 'CD':p[v][:10]=-30
        r=response_row(self.record,p,self.pair)
        self.assertEqual(r['repair_error_cents'],0);self.assertEqual(r['plateau_response_gain'],1)
        self.assertEqual(r['plateau_counter_detuning_error_cents'],0);self.assertEqual(r['outside_alteration_cents'],0)

    def test_constant_bias_is_not_counter_detuning_response(self):
        p={v:np.full(30,-5,np.float32) for v in 'ABCD'};r=response_row(self.record,p,self.pair)
        self.assertEqual(r['repair_error_cents'],25);self.assertEqual(r['plateau_response_gain'],0)
        self.assertEqual(r['plateau_counter_detuning_error_cents'],30);self.assertEqual(r['original_plateau_alteration_cents'],5)

    def test_negative_injection_keeps_gain_sign(self):
        self.record['cents']=-30;self.pair['detuning'][:10]=-30
        p=deepcopy(self.zero)
        for v in 'CD':p[v][:10]=30
        self.assertEqual(response_row(self.record,p,self.pair)['plateau_response_gain'],1)

    def test_dataset_macro_is_not_dominated_by_many_variants(self):
        a=response_row(self.record,self.zero,self.pair);b=dict(a,dataset='jamendo',group_id='jamendo:A1',source_id='j1',repair_error_cents=10)
        rows=[a]+[dict(b,corruption=str(k)) for k in range(20)]
        self.assertEqual(summarize(rows,[])['dataset_balanced_macro']['repair_error_cents'],20)

    def test_missing_dataset_is_unavailable_not_zero(self):
        r=summarize([response_row(self.record,self.zero,self.pair)],[])
        self.assertIsNone(r['full_two_dataset_macro']);self.assertEqual(r['missing_datasets'],['jamendo'])

    def test_checkpoint_reselection_refused(self):
        m=dict(status='verified',production_eligible=False,models={k:dict(selected_update=v,existing_cpu_loader_passed=True,all_state_tensors_cpu=True) for k,v in SELECTED_UPDATES.items()})
        self.assertEqual(len(frozen_model_inventory(m)),3);m['models']['bpdr']['selected_update']=275
        with self.assertRaises(ValueError):frozen_model_inventory(m)

    def test_training_mode_or_live_gradient_parameters_refused(self):
        class Toy(torch.nn.Module):
            def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.zeros(1));self.config=SimpleNamespace(features=71)
            def forward(self,x):return x[:,:,0]+self.bias
        model=Toy();raw=np.zeros((3,71),np.float32)
        with self.assertRaises(ValueError):frozen_forward(model,lambda x:x,raw)
        model.eval()
        with self.assertRaises(ValueError):frozen_forward(model,lambda x:x,raw)
        model.requires_grad_(False);self.assertTrue(np.array_equal(frozen_forward(model,lambda x:x,raw),np.zeros(3)))
        self.assertIsNone(model.bias.grad)
