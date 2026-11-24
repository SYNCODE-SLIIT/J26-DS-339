import copy
import unittest
import torch
from bpdr_input.losses import LossWeights,objective as historical
from bpdr_input.plateau_objective import objective,regions,locked_protocol,SPEC
from bpdr_input.expanded_experiment_preparation import locked_protocol as previous_protocol


def case(length=100,nonzero=1,cents=20.):
    detuning=torch.zeros(1,length);detuning[:,:nonzero]=cents
    b=dict(mask=torch.ones(1,length,dtype=torch.bool),detuning=detuning,padding=torch.zeros(1,length,dtype=torch.bool),
           voiced={v:torch.ones(1,length,dtype=torch.bool) for v in 'AC'})
    return {v:torch.zeros_like(detuning,requires_grad=True) for v in 'ABCD'},b


class PlateauObjectiveTests(unittest.TestCase):
    def test_sparse_support_is_not_diluted_by_added_zero_target_frames(self):
        for length in [4,100,1000]:
            r,b=case(length)
            new,parts,_=objective(r,b,'augmentation_only',LossWeights(smoothness=0))
            old,_,_=historical(r,b,'augmentation_only',LossWeights(smoothness=0))
            self.assertEqual(float(new),5.)
            self.assertAlmostEqual(float(old),10/length,places=6)
            self.assertEqual(float(parts['original_anchor']),0.)
            self.assertEqual(float(parts['outside_preservation']),0.)

    def test_original_and_outside_protection_remain_nonzero_penalties(self):
        r,b=case();r={v:torch.full_like(x,2.) for v,x in r.items()}
        _,parts,_=objective(r,b,'bpdr',LossWeights(smoothness=0))
        self.assertEqual(float(parts['original_anchor']),2.)
        self.assertEqual(float(parts['outside_preservation']),2.)
        # A common bias cannot satisfy the paired detuning constraint.
        self.assertEqual(float(parts['plateau_response']),20.)
        self.assertEqual(float(parts['detuning']),10.)
        r,b=case();r['C']=torch.full_like(r['C'],-20.);r['D']=r['C'].clone()
        _,parts,_=objective(r,b,'bpdr',LossWeights(smoothness=0))
        self.assertEqual(float(parts['plateau_response']),0.)
        self.assertEqual(float(parts['outside_response']),20.)
        self.assertEqual(float(parts['outside_preservation']),20.)

    def test_compensating_sign_oracle_is_zero_but_same_direction_is_not(self):
        for sign in [-1,1]:
            r,b=case(cents=sign*20.)
            r={v:(torch.zeros_like(b['detuning']) if v in 'AB' else -b['detuning']) for v in 'ABCD'}
            loss,_,_=objective(r,b,'bpdr',LossWeights(smoothness=0))
            self.assertEqual(float(loss),0.)
            r['C']=-r['C'];r['D']=-r['D']
            self.assertGreater(float(objective(r,b,'bpdr',LossWeights(smoothness=0))[0]),0.)

    def test_padding_and_unequal_lengths_preserve_equal_example_reduction(self):
        records=[case(10,1),case(100,20)]
        for i,(r,b) in enumerate(records):
            r['C']=r['C']+i;r['D']=r['D']+i
        individual=[objective(r,b,'augmentation_only',LossWeights(smoothness=0))[0] for r,b in records]
        pad=lambda x:torch.nn.functional.pad(x,(0,100-x.shape[1]))
        pred={v:torch.cat([pad(r[v]) for r,b in records]) for v in 'ABCD'}
        batch={k:torch.cat([pad(b[k]) for r,b in records]) for k in ['mask','detuning','padding']}
        batch['padding'][0,10:]=True
        batch['voiced']={v:torch.cat([pad(b['voiced'][v]) for r,b in records]) for v in 'AC'}
        loss,_,_=objective(pred,batch,'augmentation_only',LossWeights(smoothness=0))
        self.assertTrue(torch.allclose(loss,torch.stack(individual).mean()))
        pred['C'][0,10:]=1e6;pred['D'][0,10:]=-1e6
        self.assertTrue(torch.allclose(objective(pred,batch,'augmentation_only',LossWeights(smoothness=0))[0],loss))

    def test_correction_gradients_are_finite_and_have_compensating_direction(self):
        for sign in [-1,1]:
            r,b=case(cents=sign*20.)
            loss,_,_=objective(r,b,'augmentation_only',LossWeights(smoothness=0));loss.backward()
            self.assertAlmostEqual(float(r['C'].grad[0,0]),sign*.125,places=6)
            self.assertTrue(torch.isfinite(r['C'].grad).all())
            self.assertEqual(float(r['C'].grad[0,1:].abs().sum()),0.)

    def test_missing_regions_bad_masks_and_nonfinite_predictions_stop(self):
        for nonzero in [0,100]:
            r,b=case(nonzero=nonzero)
            with self.assertRaises(ValueError):objective(r,b,'bpdr')
        r,b=case();b['padding'][0,0]=True
        with self.assertRaises(ValueError):regions(b)
        r,b=case();r['A']=torch.full_like(r['A'],float('nan'))
        with self.assertRaises(ValueError):objective(r,b,'bpdr')

    def test_warmup_and_condition_term_wiring(self):
        for condition in ['augmentation_only','frame_only','bpdr']:
            for response in [False,True]:
                r,b=case();_,parts,_=objective(r,b,condition,response=response)
                self.assertEqual('detuning' in parts,condition=='bpdr' and response)
                self.assertEqual('boundary' in parts,condition=='bpdr' and response)

    def test_protocol_changes_only_declared_objective_family(self):
        base=previous_protocol();before=copy.deepcopy(base)
        p=locked_protocol(base,'reference')
        self.assertEqual(base,before)
        for key in ['artifacts','fit','model','conditions','execution','checkpoint_selection','loss_weights']:
            if key in base:self.assertEqual(p[key],base[key])
        self.assertEqual(p['objective'],SPEC)
        self.assertEqual(p['total_planned_optimizer_updates'],900)
        self.assertEqual(p['training_updates_executed'],0)
