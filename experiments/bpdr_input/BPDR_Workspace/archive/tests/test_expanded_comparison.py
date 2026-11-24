import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from bpdr_input.examples import Normalizer, collate
from bpdr_input.expanded_comparison import accumulated_step, assess, same_vectors, validate_protocol
from bpdr_input.expanded_experiment_preparation import locked_protocol
from bpdr_input.losses import LossWeights, objective


class ExpandedExecutionTests(unittest.TestCase):
    def protocol(self):
        p=locked_protocol();p['fit']['weight_decay']=.01
        return p

    def test_budget_device_and_condition_drift_are_refused(self):
        base=self.protocol();s=np.zeros((300,8),dtype=np.int64)
        validate_protocol(base,s,995)
        for key,value in [('updates',301),('warmup',99),('weight_decay',0.),('batch_size',4)]:
            p=copy.deepcopy(base);p['fit'][key]=value
            with self.assertRaises(ValueError):validate_protocol(p,s,995)
        p=copy.deepcopy(base);p['execution']['device']='cpu'
        with self.assertRaises(ValueError):validate_protocol(p,s,995)
        p=copy.deepcopy(base);p['conditions']=dict(reversed(list(p['conditions'].items())))
        with self.assertRaises(ValueError):validate_protocol(p,s,995)
        with self.assertRaises(ValueError):validate_protocol(base,s,994)
        s[0,0]=995
        with self.assertRaises(ValueError):validate_protocol(base,s,995)

    def test_accumulation_matches_equal_example_full_batch_with_unequal_lengths(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.layer=torch.nn.Linear(71,1)
            def forward(self,x,padding=None):return self.layer(x).squeeze(-1)
        torch.manual_seed(14);first=Tiny();second=copy.deepcopy(first)
        pairs=[]
        for i in range(8):
            length=5+i
            inputs={v:np.random.default_rng(i+ord(v)).normal(size=(length,71)).astype(np.float32) for v in 'ABCD'}
            for x in inputs.values():x[:,66]=1.
            pairs.append(dict(inputs=inputs,mask=np.ones(length,dtype=bool),detuning=np.full(length,10.,dtype=np.float32),pair_id=str(i),singer_id=str(i)))
        norm=Normalizer(np.zeros(71),np.ones(71));weights=LossWeights()
        a=torch.optim.SGD(first.parameters(),lr=.01);b=torch.optim.SGD(second.parameters(),lr=.01)
        row=accumulated_step(first,a,pairs,norm,71,'bpdr',weights,True,'cpu')
        batch=collate(pairs,norm,71);pred={v:second(x,batch['padding']) for v,x in batch['inputs'].items()}
        loss,_,_=objective(pred,batch,'bpdr',weights,response=True)
        loss.backward();torch.nn.utils.clip_grad_norm_(second.parameters(),1.);b.step()
        self.assertAlmostEqual(row['loss'],float(loss.detach()),places=4)
        for x,y in zip(first.parameters(),second.parameters()):self.assertTrue(torch.allclose(x,y,atol=1e-7,rtol=1e-6))
        with self.assertRaises(ValueError):accumulated_step(first,a,pairs[:7],norm,71,'bpdr',weights,True,'cpu')

    def test_zero_control_retains_voiceless_unavailable_and_plateau_error(self):
        raw=np.zeros((6,71),np.float32);raw[:3,66]=1.
        pair=dict(inputs={v:raw.copy() for v in 'ABCD'},mask=np.ones(6,bool),detuning=np.array([0,10,10,0,0,0],np.float32))
        r=dict(dataset='vocadito',group_id='singer',source_id='s',phrase_id='p',cents=10.,corruption='split',
               reference_type='human_annotation',group_kind='annotated_singer_id')
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'original.npz';np.savez(path,boundary_features_71=np.zeros((6,71),np.float32))
            original=dict(r,input_path=str(path))
            with patch('bpdr_input.expanded_comparison.read_pair',return_value=pair):
                result=assess(None,[r],[original],Normalizer(np.zeros(71),np.ones(71)),71,Path(temporary)/'out')
            self.assertEqual(result['summary']['dataset_balanced_macro']['repair_error_cents'],10.)
            self.assertEqual(result['summary']['dataset_balanced_macro']['plateau_response_gain'],0.)
            self.assertIsNone(result['summary']['original_preservation']['by_dataset']['vocadito']['mean_alteration_cents'])
            self.assertEqual(result['summary']['original_preservation']['originals'],1)
            self.assertEqual(result['prediction_vectors'],5)
            self.assertEqual(result['unique_forward_inputs'],0)

    def test_reload_refuses_vector_or_numerical_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            a=Path(temporary)/'a.npz';b=Path(temporary)/'b.npz'
            np.savez(a,p=np.zeros(4,np.float32));np.savez(b,p=np.full(4,1e-6,np.float32))
            self.assertFalse(same_vectors(a,b)['exact'])
            np.savez(b,p=np.full(4,1e-3,np.float32))
            with self.assertRaises(ValueError):same_vectors(a,b)
            np.savez(b,other=np.zeros(4,np.float32))
            with self.assertRaises(ValueError):same_vectors(a,b)
