import unittest
import torch
import numpy as np
from bpdr_input.multisource_control import one_loss,weighted_loss,select_records,prediction_rows


def example(n=10):
    e=torch.zeros(1,n);e[:,2:4]=-30
    return dict(inputs={v:torch.zeros(1,n,71) for v in 'ABCD'},mask=torch.ones(1,n,dtype=torch.bool),
        padding=torch.zeros(1,n,dtype=torch.bool),detuning=e,voiced={v:torch.ones(1,n,dtype=torch.bool) for v in 'AC'})


class MultiSourceTests(unittest.TestCase):
    def test_protection_pushes_displaced_original_toward_zero(self):
        b=example();r={v:torch.full_like(b['detuning'],-5 if v in 'AB' else 25,requires_grad=True) for v in 'ABCD'}
        total,_=one_loss(r,b,'preservation');total.backward()
        self.assertTrue(bool((r['A'].grad[:,2:4]<0).all()))
        self.assertTrue(bool((r['B'].grad[:,2:4]<0).all()))

    def test_extra_protection_is_zero_when_original_is_zero(self):
        b=example();r={v:torch.zeros_like(b['detuning']) for v in 'ABCD'}
        a,_=one_loss(r,b,'balanced');c,p=one_loss(r,b,'preservation')
        self.assertEqual(float(a),float(c));self.assertEqual(float(p['original_protection']),0)

    def test_weighted_loss_matches_independent_examples(self):
        a=example();b=example();b['detuning']=-b['detuning']
        stacked={k:torch.cat([a[k],b[k]]) for k in ['mask','padding','detuning']}
        for k in ['inputs','voiced']:stacked[k]={v:torch.cat([a[k][v],b[k][v]]) for v in a[k]}
        pred={v:torch.ones(2,10) for v in 'ABCD'}
        for arm in ['existing','balanced','preservation']:
            expected=.25*one_loss({v:x[:1] for v,x in pred.items()},a,arm)[0]+.75*one_loss({v:x[1:] for v,x in pred.items()},b,arm)[0]
            self.assertAlmostEqual(float(weighted_loss(pred,stacked,[.25,.75],arm)[0]),float(expected),5)

    def record(self,ds,artist,source,cents):
        return dict(dataset=ds,group_id=artist,source_id=source,phrase_id=source,split='train',audit_passed=True,
            exploratory=True,training_eligible=False,data_status='exploratory_not_research',cents=cents,corruption='split')

    def test_catalog_order_and_distinct_artists_with_both_shifts(self):
        r=[]
        for ds,artist,source in [('vocadito','v','v1'),('jamendo','a','a1'),('jamendo','a','a2'),('jamendo','b','b1'),('jamendo','c','c1')]:
            r.extend(self.record(ds,artist,source,c) for c in [-30.,30.])
        selected=select_records(r)
        self.assertEqual([x['source_id'] for x in selected],['v1','v1','a1','a1','b1','b1','c1','c1'])

    def test_incomplete_both_direction_coverage_refused(self):
        with self.assertRaisesRegex(ValueError,'Insufficient'):select_records([self.record('vocadito','v','v',-30.)])

    def test_wrong_partition_is_refused(self):
        r=self.record('vocadito','v','v',-30.);r['split']='development'
        with self.assertRaisesRegex(ValueError,'Wrong partition'):select_records([r])

    def test_response_gain_cannot_hide_original_alteration(self):
        b=example();r=self.record('vocadito','v','v',-30.)
        r.update(group_kind='singer',reference_type='human_annotation')
        inputs={v:np.ones((10,71),dtype=np.float32) for v in 'ABCD'}
        pair=dict(inputs=inputs,mask=b['mask'][0].numpy(),detuning=b['detuning'][0].numpy())
        for original_shift,detuned_shift,passes in [(0.,0.,False),(0.,30.,True),(-13.,17.,False)]:
            p={f'0_{v}':np.zeros(10,dtype=np.float32) for v in 'ABCD'}
            for v in 'ABCD':p[f'0_{v}'][2:4]=original_shift if v in 'AB' else detuned_shift
            row=prediction_rows([r],[pair],p)[0]
            self.assertEqual(row['bounded_response_and_preservation_pass'],passes)
            if original_shift==-13.:self.assertAlmostEqual(row['plateau_response_gain'],1.)
