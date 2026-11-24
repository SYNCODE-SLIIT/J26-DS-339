import copy
import unittest
import torch
from bpdr_input.feature_weighting_control import scale_standardized_batch,TRANSFORM,check_protocol


def batch():
    return dict(inputs={v:torch.randn(2,10,71) for v in 'ABCD'},mask=torch.ones(2,10,dtype=torch.bool),
        detuning=torch.zeros(2,10),padding=torch.zeros(2,10,dtype=torch.bool))


class FeatureWeightingTests(unittest.TestCase):
    def test_only_spectral_columns_are_scaled(self):
        b=batch();original={v:x.clone() for v,x in b['inputs'].items()};scaled=scale_standardized_batch(b)
        for v,x in scaled['inputs'].items():
            self.assertTrue(torch.equal(x[:,:,:64],original[v][:,:,:64]/8))
            self.assertTrue(torch.equal(x[:,:,64:],original[v][:,:,64:]))
            self.assertTrue(torch.equal(b['inputs'][v],original[v]))
            self.assertEqual(x.dtype,b['inputs'][v].dtype)

    def test_targets_and_masks_are_untouched(self):
        b=batch();scaled=scale_standardized_batch(b)
        for key in ['detuning','mask','padding']:self.assertIs(scaled[key],b[key])

    def test_transform_cannot_be_applied_twice(self):
        b=scale_standardized_batch(batch());self.assertEqual(b['bpdr_input_transform'],TRANSFORM)
        with self.assertRaisesRegex(ValueError,'already'):scale_standardized_batch(b)

    def test_padding_zero_and_shape_are_preserved(self):
        b=batch()
        for x in b['inputs'].values():x[:,-2:]=0
        result=scale_standardized_batch(b)
        for v,x in result['inputs'].items():
            self.assertEqual(x.shape,b['inputs'][v].shape);self.assertEqual(int(torch.count_nonzero(x[:,-2:])),0)

    def test_transform_preserves_gradient_flow(self):
        b=batch();b['inputs']['A'].requires_grad_();scaled=scale_standardized_batch(b)
        scaled['inputs']['A'].sum().backward();g=b['inputs']['A'].grad
        self.assertTrue(torch.equal(g[:,:,:64],torch.full_like(g[:,:,:64],.125)))
        self.assertTrue(torch.equal(g[:,:,64:],torch.ones_like(g[:,:,64:])))

    def test_invalid_width_and_dtype_are_refused(self):
        for x in [torch.zeros(2,10,70),torch.zeros(2,10,71,dtype=torch.int64)]:
            b=batch();b['inputs']['A']=x
            with self.assertRaisesRegex(ValueError,'features'):scale_standardized_batch(b)

    def test_incompatible_fit_protocol_cannot_reuse_control(self):
        p=dict(seed=339,updates_per_arm=100,batch_size=8,warmup=0,learning_rate=.001,weight_decay=.01,gradient_clip=1.,
            weights=[.125]*8,records=[dict(split='train',audit_passed=True,training_eligible=False) for _ in range(8)])
        check_protocol(p)
        for key,value in [('seed',340),('updates_per_arm',200),('warmup',100),('learning_rate',.01)]:
            bad=copy.deepcopy(p);bad[key]=value
            with self.assertRaisesRegex(ValueError,'mismatch'):check_protocol(bad)
